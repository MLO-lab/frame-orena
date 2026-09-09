# Supplementary annotations — MLO-Lab, ORena FOCUS 2026 FRAME track

The annotations come in two halves.

| half | contents | where |
|---|---|---|
| public | HeiCo + hernia questions, hernia sheets, hernia frames | [`Machine-Learning-Oncology/orena-frame-annotations`](https://huggingface.co/datasets/Machine-Learning-Oncology/orena-frame-annotations) |
| private | LapChole questions and sheets | available **on request** |

```bash
# public half
python annotations/fetch.py

# private half, once you have the archive
mkdir -p annotations/private
tar xzf orena-frame-lapchole-annotations_*.tar.gz \
    -C annotations/private --strip-components=1
```

Neither half is committed to git. Of the 34,006 training rows, 20,000 are the official
examples and 14,006 are ours: 8,958 public and 5,048 LapChole.

> **Licence.** The hernia frames under `public/frames/` are shared for **non-commercial
> research only**, by agreement with the clinical collaborator who supplied the videos.
> The raw videos are **not** redistributed. The LapChole material under `private/` is
> covered by the LapChole-FOCUS data usage agreement and must not be made public until the
> organisers release that dataset.

**The split is per row, not per file.** Our generators mix datasets — questions about one
dataset use frames from another as in-domain negatives — so every row is routed by its own
`source_dataset`. 481 of the public HeiCo rows were produced by the LapChole and hernia
generators but describe HeiCo frames. Run `check_no_lapchole_in_public.py` to verify that
nothing under `public/` is LapChole-derived.

---

## 1. Generation approach

Four generators, all constrained to emit a question only where an annotation logically
entails the answer.

| generator | rows | what it does |
|---|---:|---|
| `derived` | 7,022 | re-asks what the official per-frame labels already determine — class sets, counts, co-occurrence |
| `interpolated` | 3,700 | propagates a label between two annotated anchors that agree, across the gap between them |
| `lapchole_annotation` | 1,211 | from our LapChole sheets (Gallstone, Absorbable Hemostatic Agent) |
| `mesh_annotation` | 1,350 | from our hernia sheets (Mesh) |
| `*_in_domain_negative` | 800 | negatives drawn from other datasets' frames, so a class-specific question is not answerable from context alone |

### New frame-level annotations

Human annotators marked foreign objects on frames sampled every 10 s:

* **LapChole-FOCUS** — 69 videos, 520 annotated frames, targeting `Gallstone` and
  `Absorbable Hemostatic Agent`.
* **Hernia (YouTube)** — 12 videos, 232 annotated frames, targeting `Mesh`.

Counts and quadrants are used; the free-text `other_objects` column is **not** treated as an
exhaustive class list, so it does not produce `fo_class` questions on LapChole.

---

## 2. Structural format

### 2.1 Raw annotation sheets — CSV

One row per annotated frame per target class:

```csv
"image","video","timestamp","reviewer","target_class","count","quadrants","points_xy","other_objects"
```

| column | type | description |
|---|---|---|
| `image` | string | frame filename, `<video>__HH-MM-SS.jpg` |
| `video` | string | source video stem |
| `timestamp` | `HH:MM:SS` | position in the source video |
| `reviewer` | string | annotator id (may be empty) |
| `target_class` | string | FO class being annotated |
| `count` | integer | visible instances of `target_class` |
| `quadrants` | string | `top`/`bottom` + `/left`/`right`, `;`-separated; `center` = within a radius-0.18 disc of the frame centre |
| `points_xy` | string | `x,y` per instance, `;`-separated, normalised to `[0,1]` on the **full frame** |
| `other_objects` | string | other FO classes noticed; **not exhaustive** |

A header-only file means the video was reviewed and the target class never appeared — those
are negatives, not failures.

**`crop_boxes.json`** (private) records the geometry of the frames annotators saw. Some were
auto-cropped to remove letterboxing, so `points_xy` is normalised to the cropped image for
those videos; this file maps back to full-frame coordinates.

### 2.2 VQA pairs — JSON Lines

One JSON object per line:

```json
{
  "qID": "gen-mesh-000042",
  "videoID": "hernia_yt/05 - Inguinal - Full Length",
  "source_dataset": "hernia_yt",
  "procedure_type": "Hernia Repair",
  "frame": "05 - Inguinal - Full Length__00-12-00.jpg",
  "timestamp": "00:12:00",
  "time_s": 720,
  "question": "List all foreign objects that are visible in this frame.",
  "answer": "Mesh",
  "format": "fo_class",
  "primary_capability": "object_recognition",
  "secondary_capabilities": [],
  "provenance": "mesh_annotation"
}
```

| field | type | description |
|---|---|---|
| `qID` | string | unique within this release |
| `videoID` | string | `<dataset>/<video>`, matching FOCUS |
| `source_dataset` | string | `heico`, `lapchole`, `hernia_yt` |
| `frame` | string | the image filename this question is about |
| `frame_index` | int | HeiCo/LapChole: absolute frame number in the source video |
| `timestamp`, `time_s` | string, int | hernia: position in the source video |
| `question`, `answer` | string | the pair |
| `format` | string | `fo_class`, `binary`, `number`, `multiple_choice` |
| `primary_capability` | string | FOCUS capability leaf |
| `provenance` | string | which generator produced it |
| `label_confidence` | float | interpolated rows only |

### 2.3 Composition

| file | rows | fo_class | binary | number | multiple_choice |
|---|---:|---:|---:|---:|---:|
| `public/vqa/heico_derived.jsonl` | 7,608 | 2,709 | 2,170 | 2,678 | 51 |
| `public/vqa/hernia_mesh.jsonl` | 1,350 | 472 | 564 | 200 | 114 |
| `private/vqa/lapchole.jsonl` | 5,048 | 1,932 | 1,654 | 1,245 | 217 |
| **total** | **14,006** | **5,113** | **4,388** | **4,123** | **382** |

### 2.4 Frames

**HeiCo and LapChole frames are not redistributed.** Each row carries `frame_index`, the
absolute frame number in the source video, so the image is resolved from your own copy of
FOCUS at `<root>/<dataset>/frames/<video>/frame<NNNNNNN>.jpg`.

**Hernia frames ARE included**, because the videos are not part of FOCUS. All 232 are frames
the annotation sheets mark up directly, so this is the annotated set exactly — no frame
beyond what was annotated is shipped, and no video in any form.

`check_no_lapchole_in_public.py` asserts both properties.
