"""
Generates real image files for a sample of projects so the OpenCV-based
vision-consistency engine has genuine pixel data to work on (not just
metadata rows). Images are procedural renderings that are visually
*distinct per work category* (so cross-category mismatch is detectable)
and *consistent within a category* (so within-category comparisons are
meaningful) — plus a handful of exact-duplicate files to validate the
perceptual-hash duplicate detector.

The renderer draws a shaded sky/ground scene with soft lighting, per-
category motifs, and a burned-in geotag/date corner stamp (mimicking the
kind of field-app photo real MPLADS site-visit photos carry) rather than
flat noise-over-shapes — it's still a cheap procedural image, but one
that looks intentional instead of like a placeholder, and the geotag
stamp is a nice bit of realism: a *reused* photo (see
`is_seeded_duplicate` below) still carries its original project's
coordinates baked into the pixels, which is exactly the kind of mismatch
a real investigator would notice.

Run: python data/generate_photos.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sentinel.config import DATA_DIR, PHOTO_DIR

np.random.seed(7)

# Each category gets a distinct sky/ground palette + motif so a real visual
# "does this look like a <category>" signature exists in the pixels, and a
# trained classifier has genuine (not fabricated) signal to learn from.
CATEGORY_STYLE = {
    "Road Construction": {
        "sky": ((133, 178, 214), (203, 217, 222)), "ground": ((70, 70, 76), (48, 48, 54)),
        "accent": (232, 198, 64), "motif": "road",
    },
    "Community Hall": {
        "sky": ((151, 190, 224), (213, 222, 219)), "ground": ((109, 148, 89), (86, 125, 72)),
        "accent": (150, 88, 58), "roof": (117, 58, 42), "motif": "building",
    },
    "Borewell / Handpump": {
        "sky": ((160, 200, 221), (211, 226, 220)), "ground": ((140, 149, 109), (117, 132, 92)),
        "accent": (72, 92, 113), "motif": "handpump",
    },
    "Street Lighting": {
        "sky": ((41, 36, 79), (222, 133, 75)), "ground": ((35, 35, 46), (23, 23, 30)),
        "accent": (255, 221, 140), "motif": "lamp",
    },
    "School Building Repair": {
        "sky": ((146, 191, 227), (214, 224, 224)), "ground": ((118, 152, 95), (96, 133, 80)),
        "accent": (225, 205, 176), "roof": (88, 108, 138), "motif": "building",
    },
    "Drainage System": {
        "sky": ((169, 190, 200), (213, 217, 216)), "ground": ((116, 116, 111), (95, 95, 92)),
        "accent": (86, 130, 128), "motif": "drainage",
    },
    "Drinking Water Supply": {
        "sky": ((150, 195, 226), (206, 223, 226)), "ground": ((128, 163, 138), (106, 147, 117)),
        "accent": (43, 122, 178), "motif": "watertank",
    },
    "Public Toilet Complex": {
        "sky": ((156, 197, 222), (215, 222, 216)), "ground": ((124, 152, 112), (103, 136, 96)),
        "accent": (213, 213, 202), "roof": (92, 120, 120), "motif": "building",
    },
    "Sports Infrastructure": {
        "sky": ((138, 195, 231), (200, 221, 231)), "ground": ((82, 152, 71), (58, 128, 54)),
        "accent": (255, 255, 255), "motif": "field",
    },
    "Library / Reading Room": {
        "sky": ((150, 193, 220), (219, 224, 205)), "ground": ((122, 149, 98), (100, 133, 84)),
        "accent": (231, 216, 186), "roof": (121, 80, 55), "motif": "library",
    },
}

FINAL_SIZE = (320, 240)
SS = 2  # supersample factor - render bigger, downsize for free anti-aliasing
W, H = FINAL_SIZE[0] * SS, FINAL_SIZE[1] * SS
HORIZON = int(H * 0.58)

try:
    _FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 13 * SS)
except OSError:
    _FONT = ImageFont.load_default()


def _vgrad(w, h, c_top, c_bottom):
    t = np.linspace(0, 1, max(h, 1))[:, None, None]
    row = np.array(c_top, dtype=np.float32) * (1 - t) + np.array(c_bottom, dtype=np.float32) * t
    return np.repeat(row, w, axis=1)


def _scene_base(style):
    sky = _vgrad(W, HORIZON, *style["sky"])
    ground = _vgrad(W, H - HORIZON, *style["ground"])
    arr = np.vstack([sky, ground])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def _add_glow(img, cx, cy, r, color, strength=160):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*color, strength))
    layer = layer.filter(ImageFilter.GaussianBlur(r * 0.6))
    return Image.alpha_composite(img, layer)


def _shadow(draw, cx, gy, rx, ry):
    draw.ellipse([cx - rx, gy - ry, cx + rx, gy + ry], fill=(0, 0, 0, 55))


def _shade(color, factor):
    return tuple(int(np.clip(c * factor, 0, 255)) for c in color)


def render_image(category, seed, lat=None, lon=None, capture_date=None):
    rng = np.random.RandomState(seed)
    style = CATEGORY_STYLE[category]
    motif = style["motif"]

    def jit(c, spread=14):
        return tuple(int(np.clip(v + rng.uniform(-spread, spread), 0, 255)) for v in c)

    jstyle = dict(style)
    jstyle["sky"] = (jit(style["sky"][0], 16), jit(style["sky"][1], 14))
    jstyle["ground"] = (jit(style["ground"][0], 16), jit(style["ground"][1], 14))

    img = _scene_base(jstyle).convert("RGBA")

    if motif != "lamp":
        for _ in range(rng.randint(2, 4)):
            cx, cy = rng.randint(0, W), rng.randint(int(H * 0.08), int(HORIZON * 0.55))
            r = rng.randint(int(30 * SS), int(60 * SS))
            layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ImageDraw.Draw(layer).ellipse([cx - r, cy - r * 0.5, cx + r, cy + r * 0.5], fill=(255, 255, 255, 70))
            layer = layer.filter(ImageFilter.GaussianBlur(r * 0.35))
            img = Image.alpha_composite(img, layer)
        sun_x = rng.randint(int(W * 0.1), int(W * 0.9))
        sun_y = rng.randint(int(H * 0.05), int(HORIZON * 0.4))
        img = _add_glow(img, sun_x, sun_y, int(34 * SS), (255, 250, 220), strength=90)
    else:
        img = _add_glow(img, int(W * rng.uniform(0.1, 0.4)), int(HORIZON * rng.uniform(0.8, 1.0)),
                         int(rng.uniform(45, 65) * SS), (255, 170, 90), strength=130)

    draw = ImageDraw.Draw(img, "RGBA")
    accent = jit(style["accent"], 12)
    ox = rng.randint(int(-30 * SS), int(30 * SS))
    scale = rng.uniform(0.85, 1.2) * SS

    # ambient ground clutter — a handful of randomly placed/sized/coloured
    # blobs (grass tufts, rubble, shadow patches). Purely cosmetic, but
    # also gives every render enough *low-frequency* per-instance
    # structure (unlike high-frequency grain, this survives the engine's
    # 64x64 downsampling) that two independently-generated photos of the
    # same category never accidentally land within the duplicate-detector's
    # near-identical pixel threshold.
    clutter_base = style["ground"][1] if motif != "lamp" else (18, 18, 24)
    for _ in range(rng.randint(4, 8)):
        ccx = rng.randint(0, W)
        ccy = rng.randint(HORIZON + int(4 * SS), H - int(4 * SS))
        r = rng.randint(int(5 * SS), int(16 * SS))
        cc = _shade(clutter_base, rng.uniform(0.55, 1.6))
        draw.ellipse([ccx - r, ccy - r * 0.45, ccx + r, ccy + r * 0.45], fill=(*cc, int(rng.randint(110, 210))))

    if motif == "road":
        vx = W // 2 + ox // 2
        top_w, bot_w = 30 * SS, 240 * SS
        draw.polygon([(vx - top_w, HORIZON), (vx + top_w, HORIZON),
                      (vx + bot_w + ox, H), (vx - bot_w + ox, H)], fill=(58, 58, 63))
        n = 7
        for i in range(n):
            f0, f1 = i / n, i / n + 0.5 / n

            def lerp(f):
                wgt = top_w + (bot_w - top_w) * f
                y = HORIZON + (H - HORIZON) * f
                return vx + ox * f, y, wgt * 0.06

            x0, y0, w0 = lerp(f0)
            x1, y1, w1 = lerp(f1)
            draw.line([(x0, y0), (x1, y1)], fill=accent, width=max(int((w0 + w1) / 2), 2))
        for _ in range(rng.randint(4, 8)):
            x0 = rng.randint(0, W)
            y0 = rng.randint(HORIZON, H)
            w = rng.randint(int(10 * SS), int(30 * SS))
            h = rng.randint(int(6 * SS), int(16 * SS))
            draw.rectangle([x0, y0, x0 + w, y0 + h], fill=_shade(style["ground"][1], rng.uniform(0.8, 1.15)))

    elif motif == "building":
        bw = int((120 + rng.uniform(-15, 25)) * scale)
        bh = int((78 + rng.uniform(-12, 18)) * scale)
        bx = W // 2 - bw // 2 + ox
        by = HORIZON - int(6 * SS)
        roof_c = style.get("roof", _shade(accent, 0.6))
        _shadow(draw, bx + bw // 2, by + bh + int(6 * SS), bw * 0.62, int(10 * SS))
        draw.rectangle([bx, by, bx + bw, by + bh], fill=accent, outline=_shade(accent, 0.55), width=max(int(SS), 1))
        draw.rectangle([bx + bw // 2, by, bx + bw, by + bh], fill=_shade(accent, 0.88))
        roof_peak_y = by - int(38 * SS)
        draw.polygon([(bx - int(10 * SS), by), (bx + bw // 2, roof_peak_y), (bx + bw + int(10 * SS), by)],
                      fill=roof_c)
        draw.polygon([(bx + bw // 2, roof_peak_y), (bx + bw + int(10 * SS), by), (bx + bw // 2, by)],
                      fill=_shade(roof_c, 0.75))
        dw, dh = int(bw * 0.16), int(bh * 0.55)
        dx, dy = bx + bw // 2 - dw // 2, by + bh - dh
        draw.rectangle([dx, dy, dx + dw, by + bh], fill=_shade(roof_c, 0.4))
        rows, cols = 2, 3
        margin = int(bw * 0.1)
        cell_w = (bw - 2 * margin) / cols
        for r in range(rows):
            for c in range(cols):
                wx = bx + margin + c * cell_w + cell_w * 0.15
                wy = by + int(bh * 0.16) + r * (bh * 0.32)
                ww, wh = cell_w * 0.7, bh * 0.2
                if dx - cell_w * 0.3 < wx < dx + dw + cell_w * 0.3 and wy + wh > dy:
                    continue
                draw.rectangle([wx, wy, wx + ww, wy + wh], fill=(255, 244, 200), outline=_shade(roof_c, 0.4))
        draw.polygon([(bx + bw * 0.3, by + bh), (bx + bw * 0.7, by + bh),
                      (bx + bw * 0.85, H), (bx + bw * 0.15, H)], fill=_shade(style["ground"][1], 1.12))

    elif motif == "handpump":
        base_x = W // 2 + ox
        base_y = HORIZON + int(30 * SS)
        _shadow(draw, base_x, base_y + int(8 * SS), int(34 * SS), int(9 * SS))
        draw.rounded_rectangle([base_x - int(40 * SS), base_y, base_x + int(40 * SS), base_y + int(10 * SS)],
                                radius=int(4 * SS), fill=(196, 191, 178))
        pipe_w = int(9 * SS)
        draw.rectangle([base_x - pipe_w, base_y - int(70 * SS), base_x + pipe_w, base_y], fill=accent)
        draw.rectangle([base_x - pipe_w, base_y - int(70 * SS), base_x, base_y], fill=_shade(accent, 1.15))
        draw.line([(base_x, base_y - int(65 * SS)), (base_x + int(45 * SS), base_y - int(78 * SS))],
                   fill=_shade(accent, 0.85), width=int(7 * SS))
        draw.ellipse([base_x - int(6 * SS), base_y - int(71 * SS), base_x + int(6 * SS), base_y - int(59 * SS)],
                      fill=_shade(accent, 0.7))
        draw.line([(base_x - pipe_w, base_y - int(30 * SS)), (base_x - int(24 * SS), base_y - int(18 * SS))],
                   fill=_shade(accent, 0.9), width=int(6 * SS))
        splash = (90, 165, 220)
        draw.ellipse([base_x - int(30 * SS), base_y - int(4 * SS), base_x - int(16 * SS), base_y + int(3 * SS)],
                      fill=splash)
        draw.ellipse([base_x - int(50 * SS), base_y + int(4 * SS), base_x + int(10 * SS), base_y + int(14 * SS)],
                      fill=(*splash, 130))

    elif motif == "lamp":
        n_poles = rng.randint(2, 4)
        pole_spacing = rng.uniform(0.26, 0.36)
        for k in range(n_poles):
            x = int(W * (0.16 + k * pole_spacing)) + ox
            pole_top = HORIZON - int(rng.uniform(70, 95) * SS)
            _shadow(draw, x, HORIZON + int(4 * SS), int(14 * SS), int(4 * SS))
            draw.line([(x, HORIZON), (x, pole_top)], fill=(45, 45, 52), width=int(5 * SS))
            draw.line([(x, pole_top), (x + int(18 * SS), pole_top - int(10 * SS))], fill=(45, 45, 52), width=int(4 * SS))
            bulb_x, bulb_y = x + int(18 * SS), pole_top - int(10 * SS)
            img = _add_glow(img, bulb_x, bulb_y, int(rng.uniform(16, 24) * SS), accent, strength=170)
            draw = ImageDraw.Draw(img, "RGBA")
            draw.ellipse([bulb_x - int(5 * SS), bulb_y - int(5 * SS), bulb_x + int(5 * SS), bulb_y + int(5 * SS)],
                         fill=(255, 250, 230))
        # scattered distant window/house lights along the horizon — extra
        # per-instance low-frequency variation for a scene that otherwise
        # has the least structure of any category.
        for _ in range(rng.randint(2, 6)):
            wx = rng.randint(0, W)
            wy = HORIZON - rng.randint(0, int(12 * SS))
            img = _add_glow(img, wx, wy, int(rng.uniform(5, 10) * SS), (255, 200, 120), strength=110)
        draw = ImageDraw.Draw(img, "RGBA")

    elif motif == "field":
        x0, y0 = int(W * 0.08) + ox // 3, HORIZON + int(6 * SS)
        x1, y1 = int(W * 0.92) + ox // 3, H - int(6 * SS)
        draw.rectangle([x0, y0, x1, y1], outline=accent, width=int(3 * SS))
        draw.line([((x0 + x1) // 2, y0), ((x0 + x1) // 2, y1)], fill=accent, width=int(2 * SS))
        cr = int((y1 - y0) * 0.16)
        draw.ellipse([(x0 + x1) // 2 - cr, (y0 + y1) // 2 - cr, (x0 + x1) // 2 + cr, (y0 + y1) // 2 + cr],
                      outline=accent, width=int(2 * SS))
        gh = int(24 * SS)
        draw.rectangle([x0 - int(2 * SS), y0 - gh, x0 + int(30 * SS), y0 - gh + int(2 * SS)], fill=(255, 255, 255))
        draw.rectangle([x1 - int(30 * SS), y0 - gh, x1 + int(2 * SS), y0 - gh + int(2 * SS)], fill=(255, 255, 255))

    elif motif == "drainage":
        cx = W // 2 + ox
        trench_y = HORIZON + int(20 * SS)
        draw.rectangle([cx - int(70 * SS), trench_y, cx + int(70 * SS), trench_y + int(34 * SS)],
                        fill=_shade(style["ground"][1], 0.7))
        draw.rectangle([cx - int(70 * SS), trench_y, cx + int(70 * SS), trench_y + int(6 * SS)],
                        fill=_shade(style["ground"][0], 1.08))
        pipe_y = trench_y + int(14 * SS)
        draw.rectangle([cx - int(55 * SS), pipe_y - int(12 * SS), cx + int(55 * SS), pipe_y + int(12 * SS)],
                        fill=(150, 150, 145))
        draw.ellipse([cx - int(60 * SS), pipe_y - int(13 * SS), cx - int(40 * SS), pipe_y + int(13 * SS)],
                      fill=(110, 110, 106))
        draw.line([(cx - int(45 * SS), pipe_y), (cx + int(45 * SS), pipe_y)], fill=(*accent, 160), width=int(3 * SS))

    elif motif == "watertank":
        base_x = W // 2 + ox
        base_y = HORIZON + int(10 * SS)
        _shadow(draw, base_x, base_y + int(66 * SS), int(30 * SS), int(8 * SS))
        for dx in (-int(22 * SS), int(22 * SS)):
            draw.line([(base_x + dx, base_y + int(20 * SS)), (base_x + dx * 1.3, base_y + int(64 * SS))],
                       fill=(90, 90, 95), width=int(4 * SS))
        tw, th = int(46 * SS), int(52 * SS)
        draw.rounded_rectangle([base_x - tw, base_y - th, base_x + tw, base_y + int(18 * SS)],
                                radius=int(10 * SS), fill=accent)
        draw.rounded_rectangle([base_x, base_y - th, base_x + tw, base_y + int(18 * SS)],
                                radius=int(10 * SS), fill=_shade(accent, 0.82))
        draw.ellipse([base_x - tw, base_y - th - int(8 * SS), base_x + tw, base_y - th + int(8 * SS)],
                      fill=_shade(accent, 1.15))
        draw.line([(base_x, base_y + int(18 * SS)), (base_x, base_y + int(30 * SS))], fill=(70, 70, 75), width=int(4 * SS))
        draw.ellipse([base_x - int(3 * SS), base_y + int(34 * SS), base_x + int(3 * SS), base_y + int(42 * SS)],
                      fill=(90, 170, 225))

    elif motif == "library":
        bw = int(110 * scale)
        bh = int(70 * scale)
        bx = W // 2 - bw // 2 + ox
        by = HORIZON - int(4 * SS)
        roof_c = style.get("roof", _shade(accent, 0.6))
        _shadow(draw, bx + bw // 2, by + bh + int(6 * SS), bw * 0.6, int(9 * SS))
        draw.rectangle([bx, by, bx + bw, by + bh], fill=accent, outline=_shade(accent, 0.55))
        draw.rectangle([bx + bw // 2, by, bx + bw, by + bh], fill=_shade(accent, 0.88))
        draw.polygon([(bx - int(8 * SS), by), (bx + bw // 2, by - int(28 * SS)), (bx + bw + int(8 * SS), by)],
                      fill=roof_c)
        dw, dh = int(bw * 0.18), int(bh * 0.5)
        dx, dy = bx + bw // 2 - dw // 2, by + bh - dh
        draw.rectangle([dx, dy, dx + dw, by + bh], fill=_shade(roof_c, 0.4))
        book_colors = [(178, 60, 60), (60, 110, 160), (210, 165, 60)]
        bx0 = bx - int(28 * SS)
        by0 = by + bh
        for i, bc in enumerate(book_colors):
            bwi = int(26 * SS) - i * int(2 * SS)
            bhi = int(6 * SS)
            draw.rectangle([bx0 - bwi // 2, by0 - (i + 1) * bhi, bx0 + bwi // 2, by0 - i * bhi], fill=bc)

    # subtle vignette for a photographic feel
    yy, xx = np.mgrid[0:H, 0:W]
    cx0, cy0 = W / 2, H / 2
    dist = np.sqrt(((xx - cx0) / (W / 2)) ** 2 + ((yy - cy0) / (H / 2)) ** 2)
    vig = np.clip(1 - 0.16 * np.clip(dist - 0.55, 0, None), 0.8, 1.0)
    rgb = np.array(img.convert("RGB")).astype(np.float32) * vig[:, :, None]
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")

    # light photographic grain + gentle softening
    noise = (rng.randn(H, W, 3) * 6).astype(np.int16)
    arr = np.array(img).astype(np.int16) + noise
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(0.5))

    # burned-in geotag/date stamp, like a real field-app site photo
    if lat is not None and lon is not None:
        draw = ImageDraw.Draw(img, "RGBA")
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        label = f"{abs(lat):.4f}\u00b0{ns} {abs(lon):.4f}\u00b0{ew}"
        date_label = str(capture_date) if capture_date is not None else ""
        pad = int(6 * SS)
        bar_h = int(30 * SS)
        draw.rectangle([0, H - bar_h, W, H], fill=(15, 18, 25, 165))
        draw.text((pad, H - bar_h + pad // 2), label, font=_FONT, fill=(255, 255, 255, 235))
        draw.text((pad, H - bar_h + pad // 2 + int(14 * SS)), date_label, font=_FONT, fill=(200, 210, 225, 220))

    img = img.resize(FINAL_SIZE, Image.LANCZOS)
    return img


def main():
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    photos_df = pd.read_csv(DATA_DIR / "photos.csv")

    filenames = []
    dup_source_by_category = {}

    for idx, row in photos_df.iterrows():
        category = row["category"]
        photo_id = row["photo_id"]
        fname = f"{photo_id}.jpg"
        fpath = PHOTO_DIR / fname

        if row["is_seeded_duplicate"]:
            if dup_source_by_category:
                src_fname = np.random.choice(list(dup_source_by_category.values()))
                img = Image.open(PHOTO_DIR / src_fname)
            else:
                img = render_image(category, seed=idx, lat=row["latitude"], lon=row["longitude"],
                                    capture_date=row["capture_date"])
        else:
            img = render_image(category, seed=idx, lat=row["latitude"], lon=row["longitude"],
                                capture_date=row["capture_date"])
            if category not in dup_source_by_category:
                dup_source_by_category[category] = fname

        img.convert("RGB").save(fpath, quality=87)
        filenames.append(fname)

    photos_df["filename"] = filenames
    photos_df.to_csv(DATA_DIR / "photos.csv", index=False)
    print(f"Rendered {len(filenames)} images into {PHOTO_DIR}")


if __name__ == "__main__":
    main()
