#!/usr/bin/env python3
"""Glitch toolkit on top of collage.py — pixel sorting, jpeg corruption, displacement, VHS animation, real datamosh."""
import sys, os, io, math, random, subprocess, struct
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageChops, ImageEnhance
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collage as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)

# ---------------- pixel sorting ----------------
def pixel_sort(img, mask=None, lo=60, hi=200, axis=1, seed=0, chance=.9):
    """Sort runs of pixels whose luminance is between lo..hi, along rows (axis=1) or columns (0). mask (L) limits where."""
    arr = np.asarray(img.convert("RGB")).copy(); lum = arr.mean(2)
    sel = (lum > lo) & (lum < hi)
    if mask is not None: sel &= np.asarray(mask.resize(img.size)) > 128
    rng = np.random.default_rng(seed)
    if axis == 0: arr = arr.transpose(1, 0, 2); sel = sel.T
    for y in range(arr.shape[0]):
        row = sel[y]
        if not row.any(): continue
        edges = np.diff(np.concatenate([[0], row.astype(np.int8), [0]])); starts = np.where(edges == 1)[0]; ends = np.where(edges == -1)[0]
        for s, e in zip(starts, ends):
            if e - s < 8 or rng.random() > chance: continue
            seg = arr[y, s:e]; order = np.argsort(seg.mean(1)); arr[y, s:e] = seg[order]
    if axis == 0: arr = arr.transpose(1, 0, 2)
    return Image.fromarray(arr)

# ---------------- jpeg corruption ----------------
def jpeg_glitch(img, n=12, q=70, seed=0, amount=1):
    rng = random.Random(seed); b = io.BytesIO(); img.convert("RGB").save(b, "JPEG", quality=q); data = bytearray(b.getvalue())
    sos = data.find(b"\xff\xda") + 600                                   # stay clear of the headers
    for _ in range(n):
        i = rng.randrange(sos, len(data) - 2); data[i] = rng.randrange(256)
    try: return Image.open(io.BytesIO(bytes(data))).convert("RGB")
    except Exception: return img.convert("RGB")

# ---------------- displacement by silhouette ----------------
def displace(img, shape, strength=60, blur=18, seed=0):
    """warp img by the gradient of a blurred silhouette (RGBA cut-out) placed over it"""
    w, h = img.size; m = Image.new("L", (w, h), 0); sh = C.alpha_of(C.fit(shape, w=int(w * .9))); m.paste(sh, ((w - sh.width) // 2, (h - sh.height) // 2))
    m = np.asarray(m.filter(ImageFilter.GaussianBlur(blur))).astype(np.float32) / 255
    gy, gx = np.gradient(m); y, x = np.mgrid[0:h, 0:w]
    sx = np.clip(x + gx * strength * 40, 0, w - 1).astype(np.int32); sy = np.clip(y + gy * strength * 40, 0, h - 1).astype(np.int32)
    arr = np.asarray(img.convert("RGB")); return Image.fromarray(arr[sy, sx])

# ---------------- VHS frame ----------------
def vhs_frame(img, t, seed=0):
    rng = np.random.default_rng(seed + int(t * 1000)); arr = np.asarray(img.convert("RGB")).copy(); h, w, _ = arr.shape
    # row jitter bands
    for _ in range(rng.integers(1, 4)):
        y0 = int(rng.integers(0, h - 40)); hh = int(rng.integers(6, 40)); dx = int(rng.integers(-40, 40)); arr[y0:y0 + hh] = np.roll(arr[y0:y0 + hh], dx, axis=1)
    # tracking bar
    yb = int((t * 0.35 % 1) * h); band = 26
    arr[max(0, yb - band):yb] = (arr[max(0, yb - band):yb].astype(np.int16) * .5 + 90).clip(0, 255).astype(np.uint8)
    arr[max(0, yb - band):yb] = np.roll(arr[max(0, yb - band):yb], int(rng.integers(10, 60)), axis=1)
    im = Image.fromarray(arr); im = C.chroma(im, 3 + int(rng.integers(0, 5))); im = C.scanlines(im, 3, .22)
    # colour bleed: blur chroma only
    ycc = im.convert("YCbCr"); y, cb, cr = ycc.split(); cb = cb.filter(ImageFilter.GaussianBlur(3)); cr = cr.filter(ImageFilter.GaussianBlur(3)); im = Image.merge("YCbCr", (y, cb, cr)).convert("RGB")
    return C.grain(im, 22, seed + int(t * 100))

# ---------------- real datamosh ----------------
def _vop_type(frame):
    i = frame.find(b"\x00\x00\x01\xb6")
    if i < 0 or i + 4 >= len(frame): return None
    return frame[i + 4] >> 6          # 0 = I, 1 = P, 2 = B
def datamosh_avi(src_avi, dst_avi, keep_first=True, drop_every=None, dup_p=0):
    """Delete I-frames (except the first) from an MPEG-4 ASP AVI, optionally duplicating P-frames to bloom."""
    d = bytearray(open(src_avi, "rb").read())
    movi = d.find(b"movi"); i = movi + 4; out = bytearray(d[:i]); frames = []; idx = 0
    while i + 8 <= len(d):
        fourcc = bytes(d[i:i + 4]); size = struct.unpack("<I", d[i + 4:i + 8])[0]
        if fourcc == b"idx1": break
        chunk = bytes(d[i:i + 8 + size + (size & 1)]); payload = chunk[8:8 + size]
        if fourcc[2:] == b"dc":
            t = _vop_type(payload); drop = (t == 0 and idx > 0) if keep_first else (t == 0)
            if drop_every and t == 0 and idx > 0: drop = (idx % drop_every) != 0
            if not drop:
                frames.append(chunk)
                if dup_p and t == 1 and idx % 7 == 3: frames += [chunk] * dup_p
            idx += 1
        i += 8 + size + (size & 1)
    out += b"".join(frames)
    # rewrite RIFF/LIST sizes loosely; ffmpeg copes with a broken index when we skip idx1
    open(dst_avi, "wb").write(bytes(out))
def encode_frames(frames_glob, avi, fps=24):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", frames_glob, "-c:v", "libxvid", "-q:v", "6", "-g", "12", "-bf", "0", avi], check=True)
def to_mp4(avi, mp4, fps=24):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-err_detect", "ignore_err", "-fflags", "+genpts+igndts", "-i", avi, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", mp4], check=True)

# ---------------- scenes ----------------
def scene_frames(style, seed, n=96, size=(1080, 1080)):
    """elements flying through a poster: the motion that will be moshed"""
    r = random.Random(seed); W, H = size
    bg = C.RECIPES.get(style, C.recipe_punk)(seed).convert("RGB").resize(size, Image.LANCZOS)
    els = [C.load(f, 700) for f in C.rng_pick(r, C.pool(style), 7)]
    sprites = []
    for e in els:
        e = C.fit(e, w=int(W * r.uniform(.25, .55)))
        sprites.append({"im": e, "x": r.uniform(-W * .2, W * 1.2), "y": r.uniform(-H * .2, H * 1.2), "vx": r.uniform(-14, 14), "vy": r.uniform(-10, 10), "rot": r.uniform(0, 360), "vr": r.uniform(-4, 4)})
    frames = []
    for k in range(n):
        f = bg.copy().convert("RGBA")
        if k == n // 2 and r.random() < .9:                                         # a hard cut mid-way = second mosh event
            bg = C.RECIPES.get(style, C.recipe_punk)(seed + 99).convert("RGB").resize(size, Image.LANCZOS); f = bg.copy().convert("RGBA")
        for s in sprites:
            s["x"] += s["vx"]; s["y"] += s["vy"]; s["rot"] += s["vr"]
            if s["x"] < -W * .4 or s["x"] > W * 1.4: s["vx"] *= -1
            if s["y"] < -H * .4 or s["y"] > H * 1.4: s["vy"] *= -1
            C.paste(f, s["im"], s["x"], s["y"], rot=s["rot"])
        frames.append(f.convert("RGB"))
    return frames

def make_datamosh(style, seed, tag):
    d = os.path.join(OUT, f"frames_{tag}"); os.makedirs(d, exist_ok=True)
    for k, fr in enumerate(scene_frames(style, seed)): fr.save(os.path.join(d, f"f{k:04d}.png"))
    avi = os.path.join(OUT, f"{tag}_clean.avi"); encode_frames(os.path.join(d, "f%04d.png"), avi)
    moshed = os.path.join(OUT, f"{tag}_mosh.avi"); datamosh_avi(avi, moshed, dup_p=2)
    mp4 = os.path.join(OUT, f"{tag}_datamosh.mp4"); to_mp4(moshed, mp4); return mp4

def make_vhs_gif(img, tag, n=36):
    frames = [vhs_frame(img, k / 12, seed=3) for k in range(n)]
    small = [f.resize((540, int(540 * f.height / f.width)), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    path = os.path.join(OUT, f"{tag}_vhs.gif"); small[0].save(path, save_all=True, append_images=small[1:], duration=70, loop=0); return path

if __name__ == "__main__":
    what = sys.argv[1]
    if what == "stills":
        for style, seed in [("punk", 2), ("free", 4), ("mound", 1), ("death", 2), ("sigil", 4)]:
            base = C.RECIPES[style](seed).convert("RGB"); shape = C.load(random.Random(seed).choice(C.pool(style, ["symbol", "logo"]) or C.pool(style)), 1200)
            ps = pixel_sort(base, lo=40, hi=210, axis=random.Random(seed).choice([0, 1]), seed=seed); ps.save(os.path.join(OUT, f"{style}_sorted.png"))
            jg = jpeg_glitch(base, n=18, q=60, seed=seed); jg.save(os.path.join(OUT, f"{style}_jpegglitch.png"))
            dp = displace(base, shape, strength=90, blur=22); dp = pixel_sort(dp, lo=30, hi=180, axis=0, seed=seed + 1, chance=.4); dp.save(os.path.join(OUT, f"{style}_displaced.png"))
            print("stills", style)
    elif what == "vhs":
        base = C.RECIPES["mound"](1).convert("RGB"); print(make_vhs_gif(base, "mound"))
    elif what == "mosh":
        style = sys.argv[2] if len(sys.argv) > 2 else "punk"; print(make_datamosh(style, 2, style))
