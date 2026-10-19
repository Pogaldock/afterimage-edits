#!/usr/bin/env python3
"""craft.py — composition-first pieces from the element library. One subject, one word, one mark, real print craft.
Usage: python3 craft.py <recipe> <seed> <out.png>   recipes: riso punk metal crt sigil
"""
import sys, os, math, random, glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageChops, ImageEnhance
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collage as C
from collage import W, H, pool, load, fit, paste, alpha_of, crisp, xerox, stamp, halftone, grain, noise, paper, vignette, chroma, crunch, scanlines

F = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "ttf")
FONT = {k: os.path.join(F, k + ".ttf") for k in ["anton", "bebas-neue", "six-caps", "special-elite", "unifrakturcook", "metal-mania", "vt323", "rubik-glitch", "unbounded", "pirata-one", "press-start-2p", "oswald", "nosifer"]}

def tx(s, size, font, fill=(0, 0, 0, 255), tracking=0):
    """render text to a tight RGBA sprite"""
    f = ImageFont.truetype(font, size); tmp = Image.new("RGBA", (int(size * len(s) * 1.1) + tracking * len(s) + 60, int(size * 1.6)), (0, 0, 0, 0)); d = ImageDraw.Draw(tmp)
    x = 20
    for ch in s:
        d.text((x, tmp.height / 2), ch, font=f, fill=fill, anchor="lm"); x += f.getlength(ch) + tracking
    bb = tmp.getbbox(); return tmp.crop(bb) if bb else tmp
def biggest(files, n=6):
    """the n largest elements by pixel area (usually the cleanest subjects)"""
    sized = []
    for f in files:
        try: im = Image.open(f); sized.append((im.width * im.height, f))
        except Exception: pass
    return [f for _, f in sorted(sized, reverse=True)[:n]]
def intricate(files, n=6, thr=120):
    """shapes with the most edge per area — sigils, logos with counters — not slabs"""
    sc = []
    for f in files:
        try:
            im = crisp(load(f, 500), thr, 3); a = np.asarray(alpha_of(im)) > 128
            if a.mean() < .04 or a.mean() > .6: continue
            edge = np.abs(np.diff(a.astype(np.int8), axis=0)).sum() + np.abs(np.diff(a.astype(np.int8), axis=1)).sum()
            sc.append((edge / max(1, a.sum()) * math.sqrt(a.sum()), f))
        except Exception: pass
    return [f for _, f in sorted(sc, reverse=True)[:n]]
def ink_layer(size, color):
    return Image.new("RGBA", size, color + (0,))
def riso_print(paper_img, layers, seed=0):
    """layers: [(mask L, ink rgb, offset px)] — each ink laid down with its own misregistration and texture"""
    out = paper_img.convert("RGB"); rng = random.Random(seed)
    for mask, ink, off in layers:
        m = mask.copy()
        tex = noise(m.width, m.height, 1, seed + len(ink)).filter(ImageFilter.GaussianBlur(.7)).point(lambda v: 150 + v * 105 // 255)
        m = ImageChops.multiply(m, tex).rotate(rng.uniform(-.35, .35), resample=Image.BICUBIC)
        m = ImageChops.offset(m, off[0], off[1])
        inkimg = Image.new("RGB", out.size, ink); out = Image.composite(ImageChops.multiply(out, inkimg), out, m)
    return out
def crease(img, seed=0):
    rng = random.Random(seed); w, h = img.size; L = Image.new("L", (w, h), 0); d = ImageDraw.Draw(L)
    x = rng.uniform(w * .3, w * .7); d.line([(x + rng.uniform(-40, 40), 0), (x + rng.uniform(-40, 40), h)], fill=255, width=3)
    y = rng.uniform(h * .4, h * .6); d.line([(0, y), (w, y + rng.uniform(-30, 30))], fill=180, width=2)
    L = L.filter(ImageFilter.GaussianBlur(2)); arr = np.asarray(img.convert("RGB")).astype(np.int16); m = np.asarray(L)[..., None].astype(np.int16)
    return Image.fromarray(np.clip(arr - m * 28 // 255 + (np.roll(m, 3, 1) * 18 // 255), 0, 255).astype(np.uint8))
def phosphor(img, cell=3, strength=.45):
    """CRT RGB subpixel mask"""
    arr = np.asarray(img.convert("RGB")).astype(np.float32); h, w, _ = arr.shape; m = np.ones((h, w, 3), np.float32) * (1 - strength)
    x = np.arange(w) % cell; m[:, x == 0, 0] = 1; m[:, x == 1, 1] = 1; m[:, x == 2, 2] = 1
    y = np.arange(h) % cell; m[y == cell - 1] *= .7
    return Image.fromarray(np.clip(arr * m * 1.25, 0, 255).astype(np.uint8))
def bloom(img, thr=170, radius=18, amt=.8):
    g = ImageOps.grayscale(img).point(lambda v: 0 if v < thr else (v - thr) * 255 // (255 - thr)); b = img.convert("RGB").copy()
    glowed = Image.composite(b, Image.new("RGB", b.size, 0), g).filter(ImageFilter.GaussianBlur(radius))
    return ImageChops.add(b, ImageEnhance.Brightness(glowed).enhance(amt))

# ============================ recipes ============================
def riso(seed):
    """Free-party flyer: two inks on grey stock, 23 as the headline, the rig as the subject."""
    r = random.Random(seed); pap = paper(W, H, r.choice([(226, 222, 212), (233, 230, 220), (214, 220, 214)]), seed)
    black = Image.new("L", (W, H), 0); red = Image.new("L", (W, H), 0)
    ink2 = r.choice([(226, 44, 36), (20, 60, 190), (255, 90, 20), (0, 120, 110)])
    # headline: 23, huge, in the second ink
    word = r.choice(["NEVER TOO LATE", "FUCK SHIT UP", "BE WEIRD", "23", "NOT THE PLAN"]); head = tx(word, int(H * .36) if len(word) <= 2 else int(H * .19), FONT["six-caps"] if len(word) > 2 else FONT["anton"], (255, 255, 255, 255))
    head = fit(head, w=min(head.width, W - 120)); hx, hy = 70 + head.width // 2, 70 + head.height // 2
    red.paste(alpha_of(head), (hx - head.width // 2, hy - head.height // 2), alpha_of(head))
    # subject: the biggest rig/truck/object, photocopied, bleeding off the right/bottom
    subj = None
    for f in C.rng_pick(r, biggest(pool("free", ["object", "illustration", "person"]), 14), 14):
        cand = xerox(load(f, 1600), thr=138, soft=44, seed=seed)
        if .05 < C.ink_fraction(cand) < .42: subj = cand; break
    if subj is None: subj = xerox(load(biggest(pool("free", ["illustration"]), 1)[0], 1600), thr=150, soft=50, seed=seed)
    subj = fit(subj, w=int(W * r.uniform(.85, 1.1)))
    sx, sy = int(W * .62), int(H * .66); tmp = Image.new("L", (W, H), 0); tmp.paste(alpha_of(subj), (sx - subj.width // 2, sy - subj.height // 2), alpha_of(subj)); black = ImageChops.lighter(black, tmp)
    # one mark in ink 2, small, top right
    mark = crisp(load(r.choice(pool("free", ["symbol", "logo"])), 600), 110, 3); mark = fit(mark, w=int(W * .17)); tmp = Image.new("L", (W, H), 0); tmp.paste(alpha_of(mark), (W - 70 - mark.width, 80), alpha_of(mark)); red = ImageChops.lighter(red, tmp)
    # info block, typewriter, bottom left, black
    lines = [r.choice(["NEVER TOO LATE TO FUCK SHIT UP", "THIS ISN'T WHAT I PLANNED", "ALL MY FRIENDS ARE WEIRDOS"]), "INFO LINE  0836 ·· ·· ··", r.choice(["BRING A TORCH", "FOLLOW THE CONVOY", "RIG · GENERATOR · DIESEL"]), "FREE PARTY PEOPLE"]
    y = H - 70 - 34 * len(lines)
    for ln in lines:
        t = tx(ln, 30, FONT["special-elite"], (255, 255, 255, 255), 2); tmp = Image.new("L", (W, H), 0); tmp.paste(alpha_of(t), (70, y), alpha_of(t)); black = ImageChops.lighter(black, tmp); y += 36
    # a thin rule and a corner number
    d = ImageDraw.Draw(black); d.line([(70, H - 70 - 34 * len(lines) - 16), (W * .55, H - 70 - 34 * len(lines) - 16)], fill=255, width=3)
    t = tx("N° " + str(r.randint(100, 999)), 26, FONT["special-elite"], (255, 255, 255, 255)); black.paste(alpha_of(t), (W - 70 - t.width, H - 96), alpha_of(t))
    out = riso_print(pap, [(red, ink2, (r.randint(-6, 6), r.randint(-5, 5))), (black, (24, 22, 20), (0, 0))], seed)
    out = crease(out, seed); out = grain(out, 14, seed); return out

def punk(seed):
    """One face, huge and halftoned, one word on a red block, a stamp. Photocopied twice."""
    r = random.Random(seed); pap = paper(W, H, (230, 224, 206), seed).convert("RGBA")
    faces = biggest(pool("punk", ["face", "person"]) + pool("goth", ["face", "person"]), 10)
    face = load(r.choice(faces), 1600); face = fit(face, h=int(H * r.uniform(.78, .95)))
    ht = halftone(ImageEnhance.Contrast(face.convert("RGB")).enhance(1.25), cell=7, angle=r.choice([15, 45, 75]))
    fi = Image.new("RGBA", face.size, (16, 14, 12, 255)); fi.putalpha(ImageChops.multiply(ImageOps.invert(ht), alpha_of(face)))
    paste(pap, fi, W * r.uniform(.45, .62), H * .46, rot=r.uniform(-3, 3))
    # word on a block, knocked out
    word = r.choice(["BE FUCKING WEIRD", "FUCK SHIT UP", "WEIRDOS", "NOT THE PLAN", "OUT OF BED FOR THIS"]); t = tx(word, 150, FONT["anton"], (236, 230, 214, 255), 2); t = fit(t, w=min(t.width, W - 100))
    block = Image.new("RGBA", (t.width + 60, t.height + 40), r.choice([(214, 28, 28, 255), (16, 16, 16, 255), (232, 42, 128, 255)])); block.alpha_composite(t, (30, 20))
    paste(pap, block, W / 2 + r.uniform(-30, 30), H * r.choice([.14, .84]), rot=r.uniform(-5, 5))
    # rubber stamp
    sym = stamp(crisp(load(r.choice(pool("punk", ["symbol", "logo"])), 600), 110, 3), (190, 20, 20), seed); sym = fit(sym, w=int(W * .2)); paste(pap, sym, r.choice([W * .15, W * .85]), r.choice([H * .3, H * .66]), rot=r.uniform(-25, 25), alpha=.85)
    # one small ransom strip
    strip = fit(load(r.choice(pool("punk", ["lettering"])), 700), w=int(W * .34)); back = Image.new("RGBA", (strip.width + 20, strip.height + 16), (245, 240, 228, 255)); back.alpha_composite(strip, (10, 8))
    paste(pap, back, r.choice([W * .22, W * .78]), H * r.choice([.26, .72]), rot=r.uniform(-10, 10))
    out = pap.convert("RGB"); out = grain(out, 18, seed); out = ImageEnhance.Contrast(out).enhance(1.15); out = crease(out, seed + 1); return out

def metal(seed):
    """Black field, the logo as an altar, a corpse-paint face below, one title in blackletter."""
    r = random.Random(seed); base = Image.new("RGBA", (W, H), (4, 3, 4, 255))
    logos = biggest(pool("death", ["logo"]) + pool("black", ["logo", "symbol"]), 8); faces = biggest(pool("black", ["face", "person"]) + pool("death", ["face", "person"]), 8)
    face = load(r.choice(faces), 1600); face = fit(face, h=int(H * .66))
    g = ImageOps.grayscale(face.convert("RGB")); g = ImageOps.autocontrast(g, cutoff=3); g = ImageEnhance.Contrast(g).enhance(1.9)
    fb = Image.merge("RGBA", (g, g, g, alpha_of(face))); paste(base, fb, W / 2, H * .62)
    # fade the face into black at the bottom
    fade = Image.new("L", (W, H), 0); d = ImageDraw.Draw(fade)
    for y in range(int(H * .62), H): d.line([(0, y), (W, y)], fill=int(255 * min(1, (y - H * .62) / (H * .36))))
    base.alpha_composite(Image.merge("RGBA", (Image.new("L", (W, H), 4), Image.new("L", (W, H), 3), Image.new("L", (W, H), 4), fade)))
    logo = load(r.choice(logos), 1400); logo = fit(logo, w=int(W * .86)); C.glow(base, logo, W / 2, H * .2, (255, 30, 20), 34, .9); paste(base, logo, W / 2, H * .2)
    title = tx(r.choice(["All my friends are weirdos", "Never too late", "Not how I thought it would go", "Be weird"]), 58, FONT["unifrakturcook"], (210, 200, 190, 255), 2); title = fit(title, w=min(title.width, W - 120)); paste(base, title, W / 2, H * .935)
    d = ImageDraw.Draw(base); d.rectangle([34, 34, W - 34, H - 34], outline=(120, 110, 100, 255), width=2)
    out = base.convert("RGB"); out = grain(out, 24, seed); out = vignette(out, .75); return out

def crt(seed, caption=None):
    """The edit: one figure, dark field, phosphor mask, bloom, chroma, a lowercase caption."""
    r = random.Random(seed); base = Image.new("RGB", (W, H), (0, 0, 0))
    figs = biggest(pool("mound", ["person", "face"]) + pool("scene", ["person", "face"]) + pool("drain", ["person", "face"]) + pool("digi", ["person", "face"]), 14)
    f = load(r.choice(figs), 1600); f = fit(f, h=int(H * r.uniform(.72, .9)))
    # cold grade + crushed blacks
    rgb = f.convert("RGB"); rgb = ImageEnhance.Color(rgb).enhance(.55); arr = np.asarray(rgb).astype(np.float32); arr[..., 2] *= 1.18; arr[..., 0] *= .92; arr = np.clip((arr - 20) * 1.15, 0, 255); rgb = Image.fromarray(arr.astype(np.uint8))
    fg = rgb.convert("RGBA"); fg.putalpha(alpha_of(f))
    # ambient gradient behind the figure
    gl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(gl); gd.ellipse([W * .1, H * .15, W * .9, H * .85], fill=(40, 30, 70, 255)); gl = gl.filter(ImageFilter.GaussianBlur(120)); base.paste(gl, (0, 0), gl)
    b = base.convert("RGBA"); paste(b, fg, W / 2 + r.uniform(-40, 40), H * .52); base = b.convert("RGB")
    base = chroma(base, 4); base = bloom(base, 205, 16, .45); base = ImageEnhance.Contrast(base).enhance(1.25); base = phosphor(base, 3, .42); base = scanlines(base, 4, .16)
    cap = caption or r.choice(["this isn't what i planned", "i got out of bed for this shit", "this is not how i thought my life would go", "never too late to fuck shit up", "all my friends are weirdos"])
    t = tx(cap, 44, FONT["vt323"], (235, 235, 245, 255), 1); t2 = tx(cap, 44, FONT["vt323"], (120, 220, 255, 140), 1)
    bb = base.convert("RGBA"); paste(bb, t2, W / 2 + 3, H * .88 + 1); paste(bb, t, W / 2, H * .88); base = bb.convert("RGB")
    base = grain(base, 20, seed); base = vignette(base, .6); return base

def sigil(seed):
    """One chrome sigil, dead centre, with environment reflections; a single small word."""
    r = random.Random(seed); base = Image.new("RGBA", (W, H), (3, 3, 5, 255))
    shapes = intricate(pool("sigil", ["symbol", "logo", "ink"]), 8); s = crisp(load(r.choice(shapes), 1600), 120, 3); s = fit(s, w=int(W * .84))
    half = s.crop((0, 0, s.width // 2, s.height)); sym = Image.new("RGBA", (half.width * 2, s.height), (0, 0, 0, 0)); sym.alpha_composite(half, (0, 0)); sym.alpha_composite(ImageOps.mirror(half), (half.width, 0))
    ch = C.chrome(sym, hi=(255, 255, 255), mid=(170, 182, 200), lo=(18, 20, 30), seed=seed)
    # environment: a horizon band reflection
    env = Image.new("RGBA", ch.size, (0, 0, 0, 0)); ed = ImageDraw.Draw(env); ed.rectangle([0, int(ch.height * .46), ch.width, int(ch.height * .54)], fill=(255, 255, 255, 110)); env = env.filter(ImageFilter.GaussianBlur(6)); env.putalpha(ImageChops.multiply(alpha_of(env), alpha_of(ch))); ch.alpha_composite(env)
    C.glow(base, ch, W / 2, H * .47, (90, 160, 255), 60, .55); paste(base, ch, W / 2, H * .47)
    t = tx(r.choice(["BE FUCKING WEIRD", "NOT THE PLAN", "NEVER TOO LATE", "SYMBIOZE"]), 30, FONT["unbounded"], (200, 210, 230, 255), 8); t = fit(t, w=min(t.width, W - 120)); paste(base, t, W / 2, H * .9)
    out = base.convert("RGB"); out = chroma(out, 2); out = bloom(out, 200, 14, .6); out = grain(out, 8, seed); out = vignette(out, .5); return out

RECIPES = {"riso": riso, "punk": punk, "metal": metal, "crt": crt, "sigil": sigil}
if __name__ == "__main__":
    name, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    RECIPES[name](seed).save(out, quality=94); print("saved", out)
