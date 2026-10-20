#!/usr/bin/env python3
"""A little collage on top of the chaos: one or two real, recognisable things per section (a cut-out, a meme, a photo
scrap), in full colour, crisp, with a hard zine-style shadow. They give the eye something it knows. Never more than two.

    P = Paste(1080, 1920, rng)
    P.plan(cutouts, scraps)          # cutouts = RGBA images with alpha; scraps = photos/memes shown as torn rectangles
    img = P.apply(img, t, beat, u)
"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SPOTS = [(.24, .3), (.76, .34), (.26, .6), (.74, .58), (.5, .62), (.3, .22), (.72, .2)]

def torn(im, w, rng):
    """a photo scrap: cropped, white border, ragged edge"""
    im = im.convert("RGB"); r = w / im.width; im = im.resize((int(w), max(1, int(im.height * r))), Image.LANCZOS)
    cw, ch = im.width, min(im.height, int(w * rng.uniform(.7, 1.25))); y0 = rng.randrange(0, max(1, im.height - ch + 1)); im = im.crop((0, y0, cw, y0 + ch))
    b = 14; card = Image.new("RGBA", (cw + 2 * b, ch + 2 * b), (245, 242, 232, 255)); card.paste(im, (b, b))
    m = Image.new("L", card.size, 255); d = ImageDraw.Draw(m)
    for x in range(0, card.width, 6):                                     # ragged bottom/top edges
        d.rectangle([x, 0, x + 6, rng.randrange(0, 7)], fill=0); d.rectangle([x, card.height - rng.randrange(0, 9), x + 6, card.height], fill=0)
    card.putalpha(m); return card

class Paste:
    def __init__(self, w, h, rng):
        self.w, self.h, self.r = w, h, rng; self.items = []

    def plan(self, cutouts, scraps, count, center=False):
        r = self.r; self.items = []; spots = r.sample(SPOTS, len(SPOTS))
        if center: spots.append((.5, .44))                                 # the first item takes the centre (shrine)
        pool = [("cut", c) for c in cutouts if c is not None] + [("scrap", s) for s in scraps if s is not None]
        r.shuffle(pool)
        for kind, im in pool[:count]:
            w = self.w * (r.uniform(.5, .62) if center and not self.items else r.uniform(.28, .44) * getattr(self, 'size', 1.0))
            if kind == "cut":
                im = im.convert("RGBA"); bb = im.split()[3].getbbox()
                if bb: im = im.crop(bb)
                sc = w / max(im.width, im.height * .8); spr = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))), Image.LANCZOS)
            else: spr = torn(im, w, r)
            spr = spr.rotate(r.uniform(-9, 9), resample=Image.BICUBIC, expand=True)
            sh = Image.new("RGBA", spr.size, (0, 0, 0, 0)); sh.putalpha(spr.split()[3].point(lambda v: 255 if v > 100 else 0))
            x, y = spots.pop(); self.items.append({"spr": spr, "sh": sh, "x": x, "y": y, "side": r.choice([-1, 1]), "dx": r.uniform(-.02, .02), "dy": r.uniform(-.015, .015)})

    def apply(self, img, t, beat, u):
        if not self.items: return img
        img = img.convert("RGBA"); W, H = self.w, self.h
        for it in self.items:
            s = it["spr"]; ent = min(1.0, u * 8)                            # slides in fast at the start of the section, then drifts
            x = (it["x"] + it["dx"] * u) * W - s.width / 2 + it["side"] * (1 - ent) * W * .6; y = (it["y"] + it["dy"] * u) * H - s.height / 2 - 10 * beat
            img.alpha_composite(it["sh"], (int(x + 14), int(y + 14))); img.alpha_composite(s, (int(x), int(y)))
        return img.convert("RGB")
