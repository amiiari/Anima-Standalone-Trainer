"""End-of-training report: loss chart on top, then one row per Compare sample (original, no LoRA, each epoch).

    python tools/report.py <report.json> <out.png>     (report.json is written by lib/compare.js)
    python tools/report.py --thumbs <image_dir> <out_dir>   256px thumbnails for dataset snapshots
    python tools/report.py --chart <report.json> <out.png>  just the loss chart
    python tools/report.py --zip <manifest.json> <out.zip>  zip {arcname: source path}; images/LoRAs stored as-is
"""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

BG, PANEL, GRID, TEXT, MUTED = "#0d1117", "#161b22", "#30363d", "#e6edf3", "#8b949e"
CHECK, TRAIN, BEST = "#3987e5", "#d95926", "#3fb950"
CELL_H = 256


def font(size):
    return ImageFont.load_default(size=size)


def ema(values, k=0.9):
    out, acc = [], None
    for v in values:
        acc = v if acc is None else acc * k + v * (1 - k)
        out.append(acc)
    return out


def best_epoch(epochs):
    """Where the fixed-noise check flattens: first epoch with 90% of its total drop done
    (its lowest point is usually just the last epoch, since it's measured on training images)."""
    checked = [e for e in epochs if e["check"] is not None]
    if len(checked) < 3:
        return None
    low = min(e["check"] for e in checked)
    drop = checked[0]["check"] - low
    return next(e["epoch"] for e in checked if e["check"] - low <= drop * 0.1)


def draw_chart(losses, width, height=420):
    img = Image.new("RGB", (width, height), PANEL)
    d = ImageDraw.Draw(img)
    cur, epochs = losses["current"], losses["epochs"]
    if not cur:
        d.text((20, 20), "No loss logged (needs log_with = tensorboard)", fill=MUTED, font=font(18))
        return img
    l, r, t, b = 70, 90, 40, 40
    smooth = ema([v for _, v in cur])
    ys = smooth + [e["train"] for e in epochs] + [e["check"] for e in epochs if e["check"] is not None]
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.1 or 0.01
    lo, hi = lo - pad, hi + pad
    max_step = cur[-1][0] or 1
    x = lambda s: l + s / max_step * (width - l - r)
    y = lambda v: t + (1 - (v - lo) / (hi - lo)) * (height - t - b)

    for i in range(5):
        v = lo + (hi - lo) * i / 4
        d.line([(l, y(v)), (width - r, y(v))], fill=GRID)
        d.text((l - 8, y(v)), f"{v:.3f}", fill=MUTED, font=font(13), anchor="rm")
    step = max(1, len(epochs) // 15)
    for i, e in enumerate(epochs):
        if i % step == 0 or i == len(epochs) - 1:
            d.text((x(e["step"]), height - b + 14), str(e["epoch"]), fill=MUTED, font=font(13), anchor="mm")

    best = best_epoch(epochs)
    if best is not None:
        bx = x(next(e["step"] for e in epochs if e["epoch"] == best))
        for yy in range(t, height - b, 10):
            d.line([(bx, yy), (bx, yy + 5)], fill=BEST, width=2)

    d.line([(x(s), y(v)) for (s, _), v in zip(cur, smooth)], fill=TRAIN, width=2)
    pts = [(x(e["step"]), y(e["check"])) for e in epochs if e["check"] is not None]
    if len(pts) > 1:
        d.line(pts, fill=CHECK, width=2)
    for px, py in pts:
        d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=CHECK, outline=PANEL, width=2)

    d.text((width - r, 12), "x axis: epoch", fill=MUTED, font=font(13), anchor="ra")
    d.text((l, 12), "Train loss (smoothed)", fill=TRAIN, font=font(15))
    d.text((l + 200, 12), "Fixed-noise check", fill=CHECK, font=font(15))
    if best is not None:
        d.text((l + 370, 12), f"learning flattens: epoch {best}", fill=BEST, font=font(15))
    return img


def fit(path, h):
    if not os.path.exists(path):
        return None
    im = Image.open(path).convert("RGB")
    return im.resize((max(1, round(im.width * h / im.height)), h), Image.LANCZOS)


def build(report, out):
    epochs = report["losses"]["epochs"]
    best = best_epoch(epochs)
    final_epoch = epochs[-1]["epoch"] if epochs else None

    def label(name):
        if name == "__base__":
            return "No LoRA"
        tail = name.rsplit("-", 1)[-1]
        return f"E{int(tail)}" if tail.isdigit() and len(tail) == 6 else f"Final{f' (E{final_epoch})' if final_epoch else ''}"

    def epoch_of(name):
        tail = name.rsplit("-", 1)[-1]
        return int(tail) if tail.isdigit() and len(tail) == 6 else (final_epoch if name != "__base__" else None)

    rows = []
    for s in report["samples"]:
        cells = [fit(s["original"], CELL_H)] + [fit(p, CELL_H) for p in s["renders"]]
        rows.append(cells)
    col_w = max([c.width for row in rows for c in row if c] + [CELL_H])
    headers = ["Original"] + [label(n) for n in report["columns"]]
    sheet_w = 20 + len(headers) * (col_w + 8)
    width = max(1600, sheet_w)

    chart = draw_chart(report["losses"], width - 40)
    title_h, head_h = 60, 34
    height = title_h + chart.height + 30 + (head_h + len(rows) * (CELL_H + 8) if rows else 0) + 20
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    d.text((20, 18), report["title"], fill=TEXT, font=font(28))
    img.paste(chart, (20, title_h))

    y0 = title_h + chart.height + 30
    for i, h in enumerate(headers):
        cx = 20 + i * (col_w + 8)
        is_best = best is not None and i > 0 and epoch_of(report["columns"][i - 1]) == best
        d.text((cx + col_w / 2, y0 + 14), h + (" ★" if is_best else ""), fill=BEST if is_best else MUTED, font=font(16), anchor="mm")
        for r, row in enumerate(rows):
            cell = row[i]
            cy = y0 + head_h + r * (CELL_H + 8)
            if cell:
                img.paste(cell, (cx + (col_w - cell.width) // 2, cy))
            else:
                d.rectangle([cx, cy, cx + col_w, cy + CELL_H], fill=PANEL)
            if is_best:
                d.rectangle([cx - 2, cy - 2, cx + col_w + 1, cy + CELL_H + 1], outline=BEST, width=3)
    img.save(out)


def thumbs(src, dst):
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            im = Image.open(os.path.join(src, f)).convert("RGB")
            im.thumbnail((256, 256))
            im.save(os.path.join(dst, os.path.splitext(f)[0] + ".jpg"), quality=85)


def chart_only(report, out):
    epochs = report["losses"]["epochs"]
    img = Image.new("RGB", (1600, 520), BG)
    d = ImageDraw.Draw(img)
    d.text((20, 18), f"{report['title']}: loss per epoch", fill=TEXT, font=font(26))
    img.paste(draw_chart(report["losses"], 1560, 440), (20, 66))
    img.save(out)


def make_zip(manifest, out):
    import zipfile
    with zipfile.ZipFile(out + ".part", "w", allowZip64=True) as z:
        for arc, src in manifest.items():
            packed = src.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".safetensors"))
            z.write(src, arc, compress_type=zipfile.ZIP_STORED if packed else zipfile.ZIP_DEFLATED)
    os.replace(out + ".part", out)


if __name__ == "__main__":
    if sys.argv[1] == "--thumbs":
        thumbs(sys.argv[2], sys.argv[3])
    elif sys.argv[1] in ("--chart", "--zip"):
        with open(sys.argv[2], encoding="utf-8") as fh:
            data = json.load(fh)
        (chart_only if sys.argv[1] == "--chart" else make_zip)(data, sys.argv[3])
    else:
        with open(sys.argv[1], encoding="utf-8") as fh:
            build(json.load(fh), sys.argv[2])
