"""Structured prompting for the FOCUS frame track: a chain-of-thought prompt
that still yields an exact-match-parseable answer.

Base VLMs here are not blind, they are non-compliant: they name the right object
in prose that the deterministic parsers in `focus.data.formats` then reject. The
prompt targets that gap. Three constraints shaped it:

1. **Format-agnostic.** `Request` carries no answer format -- `_format` lives
   only on the ground-truth `Reference`, so branching on it would work offline
   and be impossible at submission time. One prompt for every question, keyed
   off the question's own wording ("how many", "yes or no", listed options).

2. **Chain of thought without `enable_thinking`.** Reasoning is a plain
   `REASONING:` line, not a native thought channel, so it behaves the same on
   Qwen and Gemma. `extract_answer()` keeps only the `ANSWER:` line. Capped at
   ~25 words -- every extra token costs latency against the 5s/question budget.

3. **OOD-safe.** Nothing names an anatomy, procedure, or dataset. The one closed
   vocabulary is the challenge's own FO class registry, read live from
   `FOType.names()` so that classes added as test-phase metadata are picked up
   for free. That turns open generation into constrained selection without
   teaching a procedure-specific prior.
"""

from __future__ import annotations

import re

from focus.foreign_objects import FO_DEFINITION, FO_DEFINITIONS_FILE, FOType

# ── the prompt ──────────────────────────────────────────────────────────────

# Answer-shape rules are phrased against what the QUESTION asks, never against a
# format field, and mirror the parsers in focus.data.formats exactly:
#   Binary.verify  -> text.strip().lower() in ("yes", "no")   ("Yes." FAILS)
#   Number.verify  -> text.strip().isdigit()                  ("2 clips" FAILS)
#   FOClass.verify -> comma-separated registry names, or a lone "none"
#   Time.verify    -> hh:mm:ss (comma-separated if several)
#   OpenEnded      -> <= 300 chars, LLM-judged
_ANSWER_RULES = """\
Rules for the answer:
- Write the value only. No sentence, no explanation, no units, no trailing
  period, and never repeat the question.
- Asks yes or no -> write exactly: yes   or   no
- Asks how many / for a count -> write digits only, e.g. 0 or 1 or 2.
- Asks which foreign object class(es) -> write class names exactly as spelled
  in the list above, comma-separated (e.g. Clip, Sponge), or exactly: none
  Never answer with a generic description such as "surgical instrument".
- Asks for a time -> write hh:mm:ss.
- Lists options to choose from -> copy exactly one of those options, verbatim.
- Anything else -> a short phrase, at most a few words.

If you are unsure, still commit to your single best answer in the required
form. An empty, hedged, or explanatory answer is scored as wrong."""

# SEGMENT takes a clip rather than a frame, and its format mix differs: `time` is
# its largest bucket at 38.5%, and `percentage` occurs here but never in FRAME, so
# both get explicit rules. A separate literal rather than something assembled from
# _ANSWER_RULES, so the FRAME prompt stays byte-identical to what every existing
# checkpoint was trained on.
_ANSWER_RULES_SEGMENT = """\
Rules for the answer:
- Write the value only. No sentence, no explanation, no units, no trailing
  period, and never repeat the question.
- Asks yes or no -> write exactly: yes   or   no
- Asks how many / for a count -> write digits only, e.g. 0 or 1 or 2.
- Asks which foreign object class(es) -> write class names exactly as spelled
  in the list above, comma-separated (e.g. Clip, Sponge), or exactly: none
  Never answer with a generic description such as "surgical instrument".
- Asks when something happens -> write hh:mm:ss, read off the frame timestamps.
  Those timestamps count from the START OF THE VIDEO, not from the start of this
  clip, so <1234.5 seconds> is 00:20:34. Convert carefully: 3600 seconds is one
  hour, 60 seconds is one minute. If the event falls between two frames, give
  your best estimate between their timestamps rather than snapping to one.
  For several time points, separate them with commas: 00:09:19, 00:12:44
- Asks for a percentage or a share -> write the number only, e.g. 40
- Lists options to choose from -> copy exactly one of those options, verbatim.
- Anything else -> a short phrase, at most a few words.

If you are unsure, still commit to your single best answer in the required
form. An empty, hedged, or explanatory answer is scored as wrong."""

# `direct` differs from `structured` in exactly one thing -- no reasoning line --
# so comparing them isolates whether thinking out loud helps or only costs tokens.
# It is also the natural partner for --enable-thinking: the model reasons in its
# native <think> channel, stripped before scoring, while the prompt governs only
# the shape of the final answer.
_DIRECT_SHAPE = """\
Reply with the answer and nothing else -- no reasoning, no preamble, no
explanation, no restating the question. A single short line."""

_RESPONSE_SHAPE = """\
Reply with exactly two lines, nothing before or after:

REASONING: <name the foreign objects you can see and roughly where they are;
            if the question is about something else, state the specific detail
            in the frame that decides it. Under 40 words.>
ANSWER: <the final answer only>"""


# The clip is fed as a sequence of frames, each preceded by its absolute position
# in the source video (`<559.8 seconds>`), which Qwen3-VL's processor injects from
# the VideoMetadata we supply. Naming that convention explicitly is what turns the
# markers from noise into the anchor `time` answers are measured against.
_SEGMENT_INTRO = (
    "You are a surgical video analysis assistant. You are shown a short clip "
    "from a laparoscopic procedure and asked a single question about it.\n\n"
    "The clip is given as a sequence of frames in chronological order. Each frame "
    "is preceded by its timestamp in the source video, written as "
    "<SECONDS seconds> -- a frame marked <1234.5 seconds> was taken 1234.5 seconds "
    "(00:20:34) after the video began. Consecutive frames may be several seconds "
    "apart, so an event can happen between two of them."
)

_FRAME_INTRO = (
    "You are a surgical video analysis assistant. You are shown one frame "
    "from a laparoscopic procedure and asked a single question about it."
)


def build_system_prompt(include_definitions: bool = False, style: str = "structured",
                        track: str = "frame") -> str:
    """The structured prompt, as a system message.

    Parameters
    ----------
    style
        ``"structured"`` asks for a REASONING line then an ANSWER line (visible
        chain of thought); ``"direct"`` asks for the bare answer only. The two
        share every other byte.
    include_definitions
        Append the per-class descriptions from ``FO_DEFINITIONS_FILE`` (~2.9 kB,
        roughly 700 extra prompt tokens per question). Class *names* are always
        included; this adds what each class looks like, trading latency for
        recognition.
    track
        ``"frame"`` (default) reproduces the FRAME prompt byte-for-byte.
        ``"segment"`` describes the timestamped-clip input and adds the `time`
        and `percentage` rules that track needs.
    """
    if style not in ("structured", "direct"):
        raise ValueError(f"style must be 'structured' or 'direct', got {style!r}")
    if track not in ("frame", "segment"):
        raise ValueError(f"track must be 'frame' or 'segment', got {track!r}")
    class_names = ", ".join(FOType.names())

    if include_definitions:
        objects_block = FO_DEFINITIONS_FILE.read_text().strip()
    else:
        objects_block = (
            f"{FO_DEFINITION.strip()}\n\n"
            f"The foreign object classes are exactly: {class_names}."
        )

    intro = _SEGMENT_INTRO if track == "segment" else _FRAME_INTRO
    rules = _ANSWER_RULES_SEGMENT if track == "segment" else _ANSWER_RULES

    return (
        f"{intro}\n\n"
        f"{objects_block}\n\n"
        f"{_RESPONSE_SHAPE if style == 'structured' else _DIRECT_SHAPE}\n\n"
        f"{rules}\n"
    )


# ── answer extraction ───────────────────────────────────────────────────────

# Tolerates markdown emphasis around the marker but requires the delimiter --
# making the colon optional lets this fire inside prose ("The answer is 2 clips"
# -> "is 2 clips"), the exact base-model shape it has to survive. `.*` stops at
# the newline, so rambling after the ANSWER line is not captured.
_ANSWER_RE = re.compile(r"[*_`\s]*ANSWER[*_`\s]*[:\-][^\S\n]*(.*)", re.IGNORECASE)
_REASONING_RE = re.compile(r"^[*_`\s]*REASONING[*_`\s]*[:\-].*$", re.IGNORECASE | re.MULTILINE)
# A dangling "Answer 00:10:12" on the fallback path -- marker without delimiter.
_LEADING_ANSWER_RE = re.compile(r"^answer\b[:\-]?\s*", re.IGNORECASE)

# Quotes/emphasis the model may wrap the value in, plus the trailing period that
# alone is enough to fail Binary.verify / Number.verify.
_STRIP_CHARS = " \t\n\"'`*_.:"

_OPEN_ENDED_MAX_LEN = 300  # focus.data.formats.OpenEnded default


def extract_answer(raw: str) -> str:
    """Reduce a structured generation to the bare value to be scored.

    Takes the text after the LAST ``ANSWER:`` marker (last, not first, so a
    model that restates the template before filling it in still parses). Falls
    back to the last non-empty line that is not the reasoning -- which is the
    right guess when the model answers without the marker, and no worse than the
    raw string when it was truncated mid-reasoning.
    """
    text = (raw or "").strip()
    if not text:
        return ""

    matches = list(_ANSWER_RE.finditer(text))
    if matches:
        answer = matches[-1].group(1)
    else:
        without_reasoning = _REASONING_RE.sub("", text)
        lines = [ln for ln in without_reasoning.splitlines() if ln.strip()]
        answer = _LEADING_ANSWER_RE.sub("", lines[-1] if lines else text)

    answer = answer.strip().strip(_STRIP_CHARS).strip()

    # An over-long answer fails OpenEnded.verify outright, i.e. is guaranteed
    # wrong; truncating leaves the judge something scoreable instead.
    if len(answer) > _OPEN_ENDED_MAX_LEN:
        answer = answer[:_OPEN_ENDED_MAX_LEN].rstrip()
    return answer
