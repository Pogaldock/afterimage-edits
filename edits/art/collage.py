#!/usr/bin/env python3
"""Collage engine: composes pieces from the atlas element library using one recipe per scene.
Usage: python3 collage.py <recipe> <seed> <out.png>     recipes: free punk death mound sigil
"""
import sys, os, glob, random, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageChops, ImageEnhance

ELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "els")
W, H = 1080, 1350
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
COND = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"
SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"

def pool(style, kinds=None):
    fs = sorted(glob.glob(f"{ELS}/{style}/*.png"))
    if kinds: fs = [f for f in fs if os.path.basename(f).split("-")[0] in kinds]
    return fs
def load(f, maxside=None):
    im = Image.open(f).convert("RGBA")
    if maxside and max(im.size) > maxside: im.thumbnail((maxside, maxside), Image.LANCZOS)
    return im
def fit(im, w=None, h=None):
    r = im.width / im.height
    if w and not h: h = int(w / r)
    if h and not w: w = int(h * r)
    return im.resize((max(1, w), max(1, h)), Image.LANCZOS)
def paste(base, im, x, y, rot=0, alpha=1.0):
    if rot: im = im.rotate(rot, expand=True, resample=Image.BICUBIC)
    if alpha < 1:
        a = im.split()[3].point(lambda v: int(v * alpha)); im = im.copy(); im.putalpha(a)
    base.alpha_composite(im, (int(x - im.width / 2), int(y - im.height / 2)))
def alpha_of(im): return im.split()[3]
def crisp(im, thr=110, clean=3):
    """binarise a cut-out's alpha and remove speckle (for chrome / stamp fills)"""
    a = alpha_of(im).point(lambda v: 255 if v > thr else 0).filter(ImageFilter.MinFilter(clean)).filter(ImageFilter.MaxFilter(clean)); im = im.copy(); im.putalpha(a); return im
def ink_fraction(im):
    a = np.asarray(alpha_of(im)); return float((a > 128).mean())

# ---------------- textures / effects ----------------
def noise(w, h, amt=1.0, seed=0):
    rng = np.random.default_rng(seed); return Image.fromarray((rng.random((h, w)) * 255 * amt).astype(np.uint8), "L")
def paper(w, h, tone=(236, 232, 222), seed=0):
    base = Image.new("RGB", (w, h), tone)
    n = noise(w, h, 1, seed).filter(ImageFilter.GaussianBlur(1.2)); n = ImageOps.autocontrast(n)
    fib = noise(w, h, 1, seed + 1).resize((w // 6, h)).resize((w, h)).filter(ImageFilter.GaussianBlur(.6))
    g = ImageChops.multiply(ImageOps.invert(n).point(lambda v: 200 + v * 55 // 255), ImageOps.invert(fib).point(lambda v: 210 + v * 45 // 255))
    return ImageChops.multiply(base, Image.merge("RGB", (g, g, g)))
def xerox(im, thr=128, soft=38, dirt=.35, seed=0):
    """photocopy: greyscale → hard threshold with noise → ink bleed. keeps alpha."""
    a = alpha_of(im); g = ImageOps.grayscale(im.convert("RGB")); g = ImageOps.autocontrast(g, cutoff=2)
    n = noise(im.width, im.height, 1, seed).filter(ImageFilter.GaussianBlur(.8))
    arr = np.asarray(g).astype(np.float32) + (np.asarray(n).astype(np.float32) - 128) * dirt
    ink = np.clip((thr - arr) * 255 / soft, 0, 255).astype(np.uint8)
    ink = Image.fromarray(ink).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(.5))
    out = Image.new("RGBA", im.size, (0, 0, 0, 0)); black = Image.new("RGBA", im.size, (18, 16, 14, 255))
    out.paste(black, (0, 0), ImageChops.multiply(ink, a)); return out
def stamp(im, color, seed=0):
    """keep only the alpha shape, fill with a flat colour, roughen edges"""
    a = alpha_of(im); n = noise(im.width, im.height, 1, seed).filter(ImageFilter.GaussianBlur(1))
    a = ImageChops.multiply(a, n.point(lambda v: 160 + v * 95 // 255)); a = a.filter(ImageFilter.GaussianBlur(.4))
    out = Image.new("RGBA", im.size, color + (0,)); out.putalpha(a); return out
def halftone(img, cell=6, angle=22, seed=0):
    """classic dot screen over an RGB image"""
    g = ImageOps.grayscale(img); w, h = g.size; big = g.rotate(angle, expand=True, fillcolor=255)
    out = Image.new("L", big.size, 255); d = ImageDraw.Draw(out); px = big.load()
    for y in range(0, big.height, cell):
        for x in range(0, big.width, cell):
            v = px[min(x + cell // 2, big.width - 1), min(y + cell // 2, big.height - 1)]
            r = (255 - v) / 255 * cell * .75
            if r > .6: d.ellipse([x + cell / 2 - r, y + cell / 2 - r, x + cell / 2 + r, y + cell / 2 + r], fill=0)
    out = out.rotate(-angle, expand=True); cx, cy = out.width // 2, out.height // 2
    return out.crop((cx - w // 2, cy - h // 2, cx - w // 2 + w, cy - h // 2 + h))
def grain(img, amt=18, seed=0):
    n = np.asarray(noise(img.width, img.height, 1, seed)).astype(np.int16) - 128
    arr = np.asarray(img.convert("RGB")).astype(np.int16) + (n[..., None] * amt // 128)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
def scanlines(img, period=3, dark=.35):
    arr = np.asarray(img.convert("RGB")).astype(np.float32); arr[::period] *= (1 - dark); return Image.fromarray(arr.astype(np.uint8))
def chroma(img, dx=4):
    r, g, b = img.convert("RGB").split(); return Image.merge("RGB", (ImageChops.offset(r, -dx, 0), g, ImageChops.offset(b, dx, 1)))
def crunch(img, q=12, rounds=3):
    for _ in range(rounds):
        from io import BytesIO; b = BytesIO(); img.convert("RGB").save(b, "JPEG", quality=q); b.seek(0); img = Image.open(b).convert("RGB")
    return img
def vignette(img, k=.55):
    w, h = img.size; y, x = np.mgrid[0:h, 0:w]; r = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2)
    m = np.clip(1 - k * np.clip(r - .45, 0, 1.2), 0, 1); arr = np.asarray(img.convert("RGB")).astype(np.float32) * m[..., None]
    return Image.fromarray(arr.astype(np.uint8))
def chrome(im, hi=(255, 255, 255), mid=(150, 160, 175), lo=(28, 30, 40), seed=0):
    """fill a cut-out's shape with a vertical chrome ramp + specular band"""
    a = alpha_of(im); w, h = im.size; y = np.linspace(0, 1, h)[:, None].repeat(w, 1)
    band = np.exp(-((y - .42) ** 2) / .004) * .9 + np.exp(-((y - .78) ** 2) / .002) * .5
    t = np.clip(np.abs(np.sin(y * math.pi * 1.6 + .3)), 0, 1)
    col = np.array(lo) * (1 - t)[..., None] + np.array(mid) * t[..., None]; col = col + (np.array(hi) - col) * band[..., None]
    rgb = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8)); out = rgb.convert("RGBA"); out.putalpha(a)
    # bevel: inner shadow
    edge = ImageChops.subtract(a, a.filter(ImageFilter.MinFilter(7))).filter(ImageFilter.GaussianBlur(2))
    dark = Image.new("RGBA", im.size, (0, 0, 0, 0)); dark.putalpha(edge.point(lambda v: v * 150 // 255)); out.alpha_composite(dark); return out
def glow(base, im, x, y, color, radius=30, strength=.8):
    g = Image.new("RGBA", im.size, color + (0,)); g.putalpha(alpha_of(im)); g = g.filter(ImageFilter.GaussianBlur(radius))
    paste(base, g, x, y, alpha=strength)
def text(base, s, xy, size, font=MONO, fill=(0, 0, 0, 255), rot=0, anchor="mm", spacing=0):
    f = ImageFont.truetype(font, size); tmp = Image.new("RGBA", (int(size * len(s) * .75) + 40, size * 2), (0, 0, 0, 0)); d = ImageDraw.Draw(tmp)
    d.text((tmp.width / 2, tmp.height / 2), s, font=f, fill=fill, anchor="mm"); bb = tmp.getbbox()
    if not bb: return
    tmp = tmp.crop(bb); paste(base, tmp, xy[0], xy[1], rot=rot)
def rng_pick(r, fs, n): fs = list(fs); r.shuffle(fs); return fs[:n]

# ---------------- recipes ----------------
def recipe_free(seed):
    r = random.Random(seed); base = paper(W, H, (240, 236, 226), seed).convert("RGBA")
    logos = pool("free", ["logo", "symbol"]); objs = pool("free", ["object", "illustration"]); letters = pool("free", ["lettering"]); people = pool("free", ["person"])
    # big central sigil
    main = None
    for f in rng_pick(r, logos, 12):
        cand = xerox(load(f, 1200), thr=140, soft=40, seed=seed)
        if .06 < ink_fraction(cand) < .45: main = cand; break
    if main is None: main = stamp(crisp(load(r.choice(logos), 1200)), (18, 16, 14), seed)
    main = fit(main, w=int(W * r.uniform(.62, .82)))
    paste(base, main, W / 2 + r.uniform(-30, 30), H * .40 + r.uniform(-40, 40), rot=r.uniform(-4, 4))
    # a rig or truck at the bottom
    for k, f in enumerate(rng_pick(r, objs + people, 2)):
        o = xerox(load(f, 900), thr=135, soft=45, seed=seed + k); o = fit(o, w=int(W * r.uniform(.3, .46)))
        paste(base, o, W * (.26 + .5 * k) + r.uniform(-40, 40), H * .74 + r.uniform(-30, 30), rot=r.uniform(-6, 6))
    # spiral / 23 motifs scattered small
    for k, f in enumerate(rng_pick(r, logos, 5)):
        s = xerox(load(f, 400), thr=130, seed=seed + 10 + k); s = fit(s, w=int(W * r.uniform(.08, .16)))
        paste(base, s, r.uniform(60, W - 60), r.uniform(80, H - 80), rot=r.uniform(-30, 30), alpha=r.uniform(.6, 1))
    # type: event line, info line, 23
    text(base, r.choice(["23", "SP23", "NETWORK 23", "TEKNIVAL", "FREE PARTY", "FORWARD THE REVOLUTION", "MAKE SOME FUCKING NOISE"]), (W / 2, H * .085), r.randint(54, 76), COND, (20, 18, 16, 255), rot=r.uniform(-2, 2))
    text(base, "INFO LINE  0836 ·· ·· ··   FROM 10PM", (W / 2, H * .935), 30, MONO, (20, 18, 16, 255), rot=r.uniform(-1, 1))
    text(base, r.choice(["SOUND SYSTEM", "FREE PARTY PEOPLE", "ALL NIGHT ALL WEEKEND", "RIG · DIESEL · TARMAC"]), (W / 2, H * .89), 26, MONO, (20, 18, 16, 255))
    for f in letters[:2]:
        l = xerox(load(f, 700), thr=140, seed=seed + 7); l = fit(l, w=int(W * r.uniform(.3, .5))); paste(base, l, r.uniform(200, W - 200), H * r.choice([.17, .58]), rot=r.uniform(-5, 5))
    out = base.convert("RGB"); out = grain(out, 22, seed); out = ImageEnhance.Contrast(out).enhance(1.18)
    # second-generation photocopy: slight blur + threshold-ish contrast + skew
    out = out.rotate(r.uniform(-1.2, 1.2), resample=Image.BICUBIC, fillcolor=(236, 232, 222)); return out

def recipe_punk(seed):
    r = random.Random(seed); base = paper(W, H, (232, 226, 210), seed).convert("RGBA")
    accent = r.choice([(210, 30, 30), (240, 40, 120), (20, 80, 200), (250, 210, 0)])
    # torn colour block
    blk = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(blk)
    pts = [(r.uniform(-50, 80), r.uniform(H * .2, H * .35))]; pts += [(x, pts[0][1] + r.uniform(-60, 60)) for x in range(120, W + 120, 120)]
    pts += [(W + 60, H * .62 + r.uniform(-50, 50))] + [(x, H * .62 + r.uniform(-50, 50)) for x in range(W - 120, -120, -120)]
    d.polygon(pts, fill=accent + (255,)); base.alpha_composite(blk)
    figs = pool("punk", ["person", "face", "illustration"]); letters = pool("punk", ["lettering"]); syms = pool("punk", ["symbol", "logo"])
    # main figure, halftoned
    f = load(r.choice(figs), 1300); f = fit(f, h=int(H * r.uniform(.5, .66))); ht = halftone(f.convert("RGB"), cell=5, angle=r.uniform(10, 40))
    fi = Image.new("RGBA", f.size, (10, 10, 10, 255)); fi.putalpha(ImageChops.multiply(ImageOps.invert(ht), alpha_of(f)))
    paste(base, fi, W * r.uniform(.3, .7), H * .47, rot=r.uniform(-7, 7))
    # ransom lettering strips
    for k, lf in enumerate(rng_pick(r, letters, 4)):
        l = load(lf, 900); l = fit(l, w=int(W * r.uniform(.35, .7)))
        # paper backing
        back = Image.new("RGBA", (l.width + 24, l.height + 20), (245, 240, 228, 255)); back.alpha_composite(l, (12, 10))
        paste(base, back, W / 2 + r.uniform(-120, 120), H * (.12 + .26 * k) + r.uniform(-20, 20), rot=r.uniform(-8, 8))
    for k, sf in enumerate(rng_pick(r, syms, 2)):
        s = stamp(load(sf, 500), (15, 15, 15), seed + k); s = fit(s, w=int(W * r.uniform(.14, .26))); paste(base, s, r.choice([110, W - 110]), r.uniform(120, H - 120), rot=r.uniform(-20, 20))
    text(base, r.choice(["NO FUTURE", "BORED", "DESTROY", "WE DON'T CARE", "ANARCHY", "CASH FROM CHAOS"]), (W / 2, H * .93), r.randint(70, 100), COND, (12, 12, 12, 255), rot=r.uniform(-3, 3))
    out = base.convert("RGB"); out = grain(out, 16, seed); out = ImageEnhance.Contrast(out).enhance(1.12); return out

def recipe_death(seed):
    r = random.Random(seed); base = Image.new("RGBA", (W, H), (6, 4, 5, 255))
    # rays
    rays = Image.new("L", (W, H), 0); d = ImageDraw.Draw(rays); cx, cy = W / 2, H * .45
    for k in range(18): a = k * 20 + r.uniform(-4, 4); d.polygon([(cx, cy), (cx + 1600 * math.cos(math.radians(a)), cy + 1600 * math.sin(math.radians(a))), (cx + 1600 * math.cos(math.radians(a + 7)), cy + 1600 * math.sin(math.radians(a + 7)))], fill=70)
    rays = rays.filter(ImageFilter.GaussianBlur(6)); base.alpha_composite(Image.merge("RGBA", (rays.point(lambda v: v * 120 // 70), rays.point(lambda v: v * 20 // 70), rays.point(lambda v: v * 20 // 70), rays)))
    logos = pool("death", ["logo", "symbol"]); figs = pool("death", ["person", "face", "object", "illustration"])
    fig = load(r.choice(figs), 1300); fig = fit(fig, h=int(H * r.uniform(.5, .68))); fig = ImageEnhance.Contrast(fig).enhance(1.3)
    glow(base, fig, W / 2, H * .52, (170, 20, 20), 40, .7); paste(base, fig, W / 2 + r.uniform(-40, 40), H * .52)
    for k, lf in enumerate(rng_pick(r, logos, 3)):
        l = load(lf, 1100); l = fit(l, w=int(W * r.uniform(.5, .9)))
        y = [H * .16, H * .86, H * .5][k]; glow(base, l, W / 2, y, (255, 40, 30), 26, .9); paste(base, l, W / 2 + r.uniform(-30, 30), y, rot=r.uniform(-3, 3), alpha=1 if k < 2 else .35)
    text(base, r.choice(["LEFT HAND PATH", "ALTARS", "SCREAM BLOODY GORE", "CAUSE OF DEATH", "MENTAL FUNERAL"]), (W / 2, H * .955), 30, SERIF, (200, 190, 180, 255))
    out = base.convert("RGB"); out = grain(out, 26, seed); out = vignette(out, .7); out = crunch(out, 35, 1); return out

def recipe_mound(seed):
    r = random.Random(seed); base = Image.new("RGBA", (W, H), (r.randint(10, 40), r.randint(5, 20), r.randint(5, 25), 255))
    bg = pool("mound", ["illustration", "object"]); people = pool("mound", ["person", "face"]); letters = pool("mound", ["lettering", "logo"]); syms = pool("mound", ["symbol"])
    sc = load(r.choice(bg), 1400); sc = fit(sc, w=int(W * 1.25)); paste(base, sc, W / 2, H * .55, alpha=.85)
    for k, pf in enumerate(rng_pick(r, people, 2)):
        p = load(pf, 1300); p = fit(p, h=int(H * r.uniform(.55, .75))); paste(base, p, W * (.35 + .35 * k) + r.uniform(-60, 60), H * .6, rot=r.uniform(-4, 4))
    for k, sf in enumerate(rng_pick(r, syms, 3)):
        s = load(sf, 500); s = fit(s, w=int(W * r.uniform(.12, .24))); paste(base, s, r.uniform(80, W - 80), r.uniform(80, H * .45), rot=r.uniform(-25, 25), alpha=.9)
    l = load(r.choice(letters), 1100); l = fit(l, w=int(W * r.uniform(.6, .9))); glow(base, l, W / 2, H * .14, (255, 255, 255), 18, .5); paste(base, l, W / 2, H * .14, rot=r.uniform(-3, 3))
    out = base.convert("RGB"); out = ImageEnhance.Color(out).enhance(1.9); out = ImageEnhance.Contrast(out).enhance(1.5)
    out = chroma(out, 5); out = crunch(out, 9, 4); out = scanlines(out, 3, .25); out = grain(out, 30, seed); out = vignette(out, .5); return out

def recipe_sigil(seed):
    r = random.Random(seed); base = Image.new("RGBA", (W, H), (4, 4, 6, 255))
    inks = pool("sigil", ["symbol", "logo"]) or pool("sigil"); inks_all = pool("sigil", ["ink", "symbol", "logo"])
    # symmetric composition: one big chrome sigil mirrored, two satellites
    main = crisp(load(r.choice(inks), 1300), 120, 5); main = fit(main, w=int(W * r.uniform(.6, .85)))
    half = main.crop((0, 0, main.width // 2, main.height)); sym = Image.new("RGBA", (half.width * 2, main.height), (0, 0, 0, 0)); sym.alpha_composite(half, (0, 0)); sym.alpha_composite(ImageOps.mirror(half), (half.width, 0))
    ch = chrome(sym, seed=seed); glow(base, ch, W / 2, H * .42, (120, 200, 255), 50, .6); paste(base, ch, W / 2, H * .42)
    sats = [f for f in rng_pick(r, inks_all, 12) if ink_fraction(crisp(load(f, 400), 120, 3)) < .5][:2]
    for k, f in enumerate(sats):
        s = chrome(crisp(load(f, 700), 120, 3), seed=seed + k); s = fit(s, w=int(W * r.uniform(.22, .34))); paste(base, s, W * (.22 + .56 * k), H * .8 + r.uniform(-40, 40), rot=r.uniform(-15, 15))
    # thin cyber lines
    d = ImageDraw.Draw(base)
    for k in range(14): y = r.uniform(40, H - 40); d.line([(r.uniform(0, W * .3), y), (r.uniform(W * .7, W), y)], fill=(120, 130, 150, r.randint(30, 90)), width=1)
    text(base, r.choice(["SYMBIOZE", "ERROR 23", "NO SIGNAL", "ASCEND", "DEADLINK"]), (W / 2, H * .94), 34, MONO, (190, 200, 220, 255))
    out = base.convert("RGB"); out = chroma(out, 2); out = grain(out, 10, seed); out = vignette(out, .6); return out

RECIPES = {"free": recipe_free, "punk": recipe_punk, "death": recipe_death, "mound": recipe_mound, "sigil": recipe_sigil}
if __name__ == "__main__":
    name, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    RECIPES[name](seed).save(out, quality=92); print("saved", out)
