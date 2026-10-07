"""Run the real model over tests/images/* and write tests/out/report.md + annotated images.
    python -m tests.run_eval                # uses env config
    REPAIR_PROMPT=v2 python -m tests.run_eval
Optional tests/expected.json:
{"bike1.jpg": {"object": "bicycle", "keywords": ["derailleur"], "bbox": [0.6, 0.4, 0.2, 0.2], "insufficient": false}}
Manual columns (hallucination / safe action) are for you to fill in after looking at the annotated images."""
import json
import pathlib
import sys

from PIL import Image, ImageDraw

from app import config, pipeline

ROOT = pathlib.Path(__file__).parent
IMG_DIR, OUT = ROOT / "images", ROOT / "out"


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def main():
    OUT.mkdir(exist_ok=True)
    exp = json.loads((ROOT / "expected.json").read_text()) if (ROOT / "expected.json").exists() else {}
    files = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not files:
        sys.exit(f"Put 5-10 test images in {IMG_DIR}")
    rows = ["| image | status | object | problem | conf | risk | secs | obj ok | kw ok | IoU | leaks | insuff ok |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in files:
        res = pipeline.analyze(p.read_bytes())
        (OUT / f"{p.stem}.json").write_text(json.dumps(res, indent=2))
        im = Image.open(p).convert("RGB")
        if res["location"]:
            L, (W, H) = res["location"], im.size
            ImageDraw.Draw(im).rectangle([L["x"] * W, L["y"] * H, (L["x"] + L["width"]) * W, (L["y"] + L["height"]) * H],
                                         outline="red", width=max(3, W // 150))
        im.save(OUT / f"{p.stem}_boxed.jpg")
        e = exp.get(p.name, {})
        text = (res["problem"] + " " + " ".join(res["observations"])).lower()
        obj_ok = "" if "object" not in e else str(e["object"].lower() in res["object"].lower())
        kw_ok = "" if "keywords" not in e else str(any(k.lower() in text for k in e["keywords"]))
        loc = res["location"]
        i = "" if not (e.get("bbox") and loc) else f'{iou(e["bbox"], [loc["x"], loc["y"], loc["width"], loc["height"]]):.2f}'
        ins = "" if "insufficient" not in e else str((res["status"] == pipeline.INSUFFICIENT) == e["insufficient"])
        leaks = sum(1 for o in res["observations"] if pipeline.INFERENCE_RE.search(o))
        rows.append(f"| {p.name} | {res['status']} | {res['object']} | {res['problem']} | {res['confidence']} | {res['risk']} | "
                    f"{res['meta']['seconds']} | {obj_ok} | {kw_ok} | {i} | {leaks} | {ins} |")
        print(rows[-1])
    (OUT / f"report_{config.MODEL.replace(':', '_').replace('/', '_')}_{config.PROMPT_VERSION}.md").write_text("\n".join(rows))


if __name__ == "__main__":
    main()
