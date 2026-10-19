#!/usr/bin/env python3
"""Recognisable symbols drawn as clean vectors (supersampled), so the hero of every shot is crisp at 1080x1920.
symbol(name, size, fg, bg=None, line=None) -> RGBA image.  NAMES lists them.
"""
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SS = 4  # supersampling factor

def _canvas(size): s = size * SS; return Image.new("RGBA", (s, s), (0, 0, 0, 0)), s

def _finish(im, size): return im.resize((size, size), Image.LANCZOS)

def smiley(size, fg, bg):
    im, s = _canvas(size); d = ImageDraw.Draw(im); r = s * .47; c = s / 2; w = int(s * .045)
    d.ellipse([c - r, c - r, c + r, c + r], fill=fg, outline=bg, width=w)
    ex, ey, ew, eh = s * .14, s * .38, s * .07, s * .16
    for sx in (-1, 1): d.ellipse([c + sx * ex - ew / 2, ey - eh / 2, c + sx * ex + ew / 2, ey + eh / 2], fill=bg)
    d.arc([c - r * .62, c - r * .5, c + r * .62, c + r * .66], 20, 160, fill=bg, width=int(s * .055))
    return _finish(im, size)

def alien(size, fg, bg):
    im, s = _canvas(size); d = ImageDraw.Draw(im); c = s / 2
    pts = []                                                  # inverted-teardrop head
    for i in range(200):
        a = 2 * math.pi * i / 200; x = math.sin(a); y = -math.cos(a)
        wx = .40 * (1 - .45 * max(0, y)) * (1 + .05 * (1 - y)); pts.append((c + x * s * wx, c - s * .06 + y * s * (.46 if y < 0 else .40)))
    d.polygon(pts, fill=fg, outline=bg, width=int(s * .02))
    for sx in (-1, 1):                                        # almond eyes, tilted
        eye = Image.new("RGBA", (s, s), (0, 0, 0, 0)); ed = ImageDraw.Draw(eye)
        ed.ellipse([c - s * .11, c - s * .055, c + s * .11, c + s * .055], fill=bg)
        eye = eye.rotate(-28 * sx, center=(c, c), resample=Image.BICUBIC); im.alpha_composite(eye, (int(sx * s * .14), int(-s * .03)))
    d.line([c - s * .035, c + s * .21, c + s * .035, c + s * .21], fill=bg, width=int(s * .012))
    return _finish(im, size)

def trefoil(size, fg, bg):
    im, s = _canvas(size); d = ImageDraw.Draw(im); c = s / 2; R = s * .47
    d.ellipse([c - R, c - R, c + R, c + R], fill=bg)
    for k in range(3):
        a0 = -90 + 120 * k - 30; d.pieslice([c - R * .92, c - R * .92, c + R * .92, c + R * .92], a0, a0 + 60, fill=fg)
    d.ellipse([c - R * .3, c - R * .3, c + R * .3, c + R * .3], fill=bg); d.ellipse([c - R * .2, c - R * .2, c + R * .2, c + R * .2], fill=fg)
    return _finish(im, size)

def peace(size, fg, bg):
    im, s = _canvas(size); d = ImageDraw.Draw(im); c = s / 2; R = s * .44; w = int(s * .07)
    d.ellipse([c - R, c - R, c + R, c + R], outline=fg, width=w)
    d.line([c, c - R, c, c + R], fill=fg, width=w)
    for sx in (-1, 1): d.line([c, c, c + sx * R * math.sin(math.radians(45)), c + R * math.cos(math.radians(45))], fill=fg, width=w)
    return _finish(im, size)

def anarchy(size, fg, bg):
    im, s = _canvas(size); d = ImageDraw.Draw(im); c = s / 2; R = s * .42; w = int(s * .06)
    d.ellipse([c - R, c - R, c + R, c + R], outline=fg, width=w)
    top = (c, c - R * 1.12); bl = (c - R * .78, c + R * 1.08); br = (c + R * .78, c + R * 1.08)
    d.line([bl, top, br], fill=fg, width=int(w * 1.3), joint="curve"); d.line([c - R * .55, c + R * .28, c + R * .55, c + R * .28], fill=fg, width=w)
    return _finish(im, size)

def eye(size, fg, bg):
    im, s = _canvas(size); d = ImageDraw.Draw(im); c = s / 2; w = int(s * .035)
    tri = [(c, s * .06), (s * .95, s * .88), (s * .05, s * .88)]; d.polygon(tri, outline=fg, width=w)
    ey = s * .58; ew, eh = s * .26, s * .12
    d.chord([c - ew, ey - eh * 1.4, c + ew, ey + eh * 1.4], 200, 340, fill=fg); d.chord([c - ew, ey - eh * 1.4, c + ew, ey + eh * 1.4], 20, 160, fill=fg)
    d.ellipse([c - ew * .95, ey - eh, c + ew * .95, ey + eh], fill=fg); d.ellipse([c - eh * .85, ey - eh * .85, c + eh * .85, ey + eh * .85], fill=bg); d.ellipse([c - eh * .35, ey - eh * .35, c + eh * .35, ey + eh * .35], fill=fg)
    return _finish(im, size)

def yinyang(size, fg, bg):
    im, s = _canvas(size); d = ImageDraw.Draw(im); c = s / 2; R = s * .46
    d.ellipse([c - R, c - R, c + R, c + R], fill=bg, outline=fg, width=int(s * .02)); d.pieslice([c - R, c - R, c + R, c + R], 90, 270, fill=fg)
    d.ellipse([c - R / 2, c - R, c + R / 2, c], fill=fg); d.ellipse([c - R / 2, c, c + R / 2, c + R], fill=bg)
    d.ellipse([c - R * .13, c - R * .63, c + R * .13, c - R * .37], fill=bg); d.ellipse([c - R * .13, c + R * .37, c + R * .13, c + R * .63], fill=fg)
    return _finish(im, size)

def spiral(size, fg, bg):
    im, s = _canvas(size); d = ImageDraw.Draw(im); c = s / 2; pts = []
    for i in range(1400):
        t = i / 1400 * 5.5 * 2 * math.pi; r = s * .46 * (i / 1400); pts.append((c + math.cos(t) * r, c + math.sin(t) * r))
    d.line(pts, fill=fg, width=int(s * .04), joint="curve"); return _finish(im, size)

def twentythree(size, fg, bg, font=None):
    im, s = _canvas(size); d = ImageDraw.Draw(im); c = s / 2
    try: f = ImageFont.truetype(font, int(s * .78)) if font else ImageFont.load_default()
    except Exception: f = ImageFont.load_default()
    d.ellipse([c - s * .48, c - s * .48, c + s * .48, c + s * .48], outline=fg, width=int(s * .035))
    d.text((c, c * 1.04), "23", font=f, fill=fg, anchor="mm"); return _finish(im, size)

def star(size, fg, bg):
    im, s = _canvas(size); d = ImageDraw.Draw(im); c = s / 2; pts = []
    for k in range(10):
        r = s * (.48 if k % 2 == 0 else .2); a = math.radians(-90 + 36 * k); pts.append((c + math.cos(a) * r, c + math.sin(a) * r))
    d.polygon(pts, fill=fg); return _finish(im, size)

def sun(size, fg, bg):
    im, s = _canvas(size); d = ImageDraw.Draw(im); c = s / 2; R = s * .2
    for k in range(16):
        a = 2 * math.pi * k / 16; a2 = a + math.pi / 16
        d.polygon([(c + math.cos(a - .12) * R * 1.3, c + math.sin(a - .12) * R * 1.3), (c + math.cos(a2) * s * .48, c + math.sin(a2) * s * .48), (c + math.cos(a + .32) * R * 1.3, c + math.sin(a + .32) * R * 1.3)], fill=fg)
    d.ellipse([c - R, c - R, c + R, c + R], fill=fg); return _finish(im, size)

def pyramid(size, fg, bg):
    im, s = _canvas(size); d = ImageDraw.Draw(im); c = s / 2; w = int(s * .03)
    d.polygon([(c, s * .08), (s * .94, s * .86), (s * .06, s * .86)], outline=fg, width=w)
    for k in range(1, 6):
        y = s * (.08 + .78 * k / 6); half = (y - s * .08) / (.78 * s) * s * .44; d.line([c - half, y, c + half, y], fill=fg, width=int(w * .6))
    return _finish(im, size)

def ufo(size, fg, bg):
    im, s = _canvas(size); d = ImageDraw.Draw(im); c = s / 2
    d.ellipse([c - s * .17, s * .26, c + s * .17, s * .56], fill=fg); d.ellipse([c - s * .47, s * .42, c + s * .47, s * .62], fill=fg)
    for k in range(-2, 3): d.ellipse([c + k * s * .15 - s * .03, s * .5 - s * .03, c + k * s * .15 + s * .03, s * .5 + s * .03], fill=bg)
    d.polygon([(c - s * .14, s * .62), (c + s * .14, s * .62), (c + s * .34, s * .96), (c - s * .34, s * .96)], fill=fg[:3] + (90,))
    return _finish(im, size)

FUNCS = {"smiley": smiley, "alien": alien, "trefoil": trefoil, "peace": peace, "anarchy": anarchy, "eye": eye, "yinyang": yinyang,
         "spiral": spiral, "23": twentythree, "star": star, "sun": sun, "pyramid": pyramid, "ufo": ufo}
NAMES = list(FUNCS)

def symbol(name, size, fg=(255, 255, 255, 255), bg=(0, 0, 0, 255), font=None, glow=None):
    fn = FUNCS[name]; im = fn(size, fg, bg, font) if name == "23" else fn(size, fg, bg)
    if glow:                                                  # neon halo behind the shape
        a = im.split()[3].filter(ImageFilter.GaussianBlur(size * .035)); halo = Image.new("RGBA", im.size, glow + (0,)); halo.putalpha(a.point(lambda v: min(255, int(v * 1.6))))
        base = Image.new("RGBA", im.size, (0, 0, 0, 0)); base.alpha_composite(halo); base.alpha_composite(im); return base
    return im

if __name__ == "__main__":
    import sys
    out = Image.new("RGBA", (5 * 300, 3 * 300), (20, 0, 40, 255))
    for i, n in enumerate(NAMES): out.alpha_composite(symbol(n, 280, (255, 240, 0, 255), (20, 0, 40, 255), font=sys.argv[2] if len(sys.argv) > 2 else None, glow=(255, 0, 180)), ((i % 5) * 300 + 10, (i // 5) * 300 + 10))
    out.convert("RGB").save(sys.argv[1] if len(sys.argv) > 1 else "symbols.jpg")
