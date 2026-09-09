"""Emit the provenance of the hernia videos into the public release.

The frames we redistribute are stills from YouTube videos that carry the standard YouTube
licence -- NOT an open licence. Redistribution rests on a direct agreement with the channel
owner, so the release must name every source and state the terms rather than implying the
material is openly licensed.

    python annotations/make_sources.py --manifest <mesh/manifest.json>
"""
import argparse, json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, default=HERE / "public/SOURCES.md")
    ap.add_argument("--out-json", type=Path, default=HERE / "public/sources.json")
    a = ap.parse_args()

    man = json.loads(a.manifest.read_text())
    used = {k: v for k, v in man.items() if v.get("folder")}
    rows = sorted(used.items(), key=lambda kv: kv[1]["folder"])

    out = {}
    for vid, v in rows:
        out[v["folder"]] = {
            "youtube_id": vid, "url": v["url"], "title": v["title"],
            "uploader": v["uploader"], "channel_url": v.get("channel_url"),
            "upload_date": v.get("upload_date"), "duration_s": v.get("duration_s"),
            "license": v.get("license") or "Standard YouTube License",
            "open_licence": bool(v.get("open_licence")),
        }
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(out, indent=2) + "\n")

    channels = sorted({v["uploader"] for _, v in rows})
    md = ["# Hernia video sources",
          "",
          "The hernia questions and frames in this release derive from publicly available",
          "YouTube videos. **These videos carry the standard YouTube licence, not an open**",
          "**licence.** We redistribute only annotated frames — never the video — under a",
          "direct agreement with the channel owner, limited to **non-commercial research**.",
          "",
          f"All {len(rows)} videos come from " + " and ".join(f"*{c}*" for c in channels) + ".",
          "",
          "| # | our name | title | link |",
          "|---|---|---|---|"]
    for vid, v in rows:
        n = v["folder"].split(" - ")[0]
        # titles contain '|' (e.g. "... (Full Surgery) | Dr. Todd Harris"), which would
        # break the table
        title = v["title"].replace("|", "\\|")
        md.append(f"| {n} | `{v['folder']}` | {title} | [{vid}]({v['url']}) |")
    md += ["",
           "## Per-video detail",
           "",
           "| our name | uploaded | duration | licence |",
           "|---|---|---|---|"]
    for vid, v in rows:
        d = v.get("duration_s") or 0
        ud = v.get("upload_date") or ""
        ud = f"{ud[:4]}-{ud[4:6]}-{ud[6:]}" if len(ud) == 8 else ud
        md.append(f"| `{v['folder']}` | {ud} | {d//60:d}m{d%60:02d}s | "
                  f"{v.get('license') or 'Standard YouTube License'} |")
    md += ["",
           "`sources.json` carries the same information in machine-readable form.",
           ""]
    a.out_md.write_text("\n".join(md))
    print(f"{a.out_md}  ({len(rows)} videos)")
    print(f"{a.out_json}")


if __name__ == "__main__":
    main()
