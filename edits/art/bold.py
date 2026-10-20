#!/usr/bin/env python3
"""Bold layer, pixel-art edition: colour and shape that cut through the picture, made from the edit's own material.
Two strong colours per edit, over a mostly black-and-white image. At most one element per section:

    sprite    a cut-out / meme / hero crushed into a big pixel sprite, stepping on the kick
    invader   a procedural mirrored creature: one giant, or a marching formation of small ones
    pglyph    a planet / zodiac / I Ching glyph as a huge chunky pixel glyph
    dither    a shape where the picture turns into big dithered pixels in the edit's colours
    band      horizontal bands of dithered pixels sliding through the frame
    window    a shape where the picture turns negative, with a stepped pixel edge
"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageChops
import pixel as PX

PALETTE = [(255, 40, 40), (255, 225, 0), (0, 90, 255), (0, 255, 140), (255, 0, 170), (255, 110, 0), (0, 225, 255)]
KINDS = ["sprite", "sprite", "invader", "pglyph", "dither", "band", "window"]
GLYPHS = "☉☽☿♀♂♃♄♈♉♊♋♌♍♎♏♐♑♒♓☰☷䷀䷁䷊䷋"

def pixel_mask(W, H, shape, cx, cy, R, step=12):
    """a shape drawn at low resolution and blown up, so its edge is stair-stepped pixels"""
    w, h = W // step, H // step; m = Image.new("L", (w, h), 0); d = ImageDraw.Draw(m); x, y, r = cx / step, cy / step, R / step
    if shape == "circle": d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    elif shape == "diamond": d.polygon([(x, y - r * 1.2), (x + r, y), (x, y + r * 1.2), (x - r, y)], fill=255)
    else: d.rectangle([x - r, y - r * .7, x + r, y + r * .7], fill=255)
    return m.resize((W, H), Image.NEAREST)

class Bold:
    def __init__(self, w, h, rng, colours=None):
        self.w, self.h, self.r = w, h, rng; self.cols = colours or rng.sample(PALETTE, 2); self.kind = None; self.images = []
        self.nprng = np.random.default_rng(int(rng.random() * 1e9))

    def plan(self, kind, images=None):
        r = self.r; self.kind = kind; self.images = [im for im in (images or []) if im is not None]
        self.col = r.choice(self.cols); self.col2 = [c for c in self.cols if c != self.col][0] if len(self.cols) > 1 else (255, 255, 255)
        self.x, self.y = r.choice([(.5, .45), (.5, .4), (.35, .38), (.65, .5)]); self.shape = r.choice(["circle", "diamond", "rect"])
        pal = [(0, 0, 0), self.col, (255, 255, 255)] + ([self.col2] if r.random() < .5 else [])
        self.pal = pal; self.spr = None; self.spr2 = None; self.inv = None; self.gl = None
        if kind == "sprite":
            src = r.choice(self.images) if self.images else None
            if src is None: self.kind = "invader"
            else:
                hgt = r.choice([56, 72, 88]); sp_pal = [(0, 0, 0), (70, 70, 70), self.col, (200, 200, 200), (255, 255, 255)]; self.spr = PX.sprite_from(src, hgt, sp_pal); self.spr2 = PX.sprite_from(src.transpose(Image.FLIP_LEFT_RIGHT), hgt, sp_pal)
        if self.kind == "invader":
            self.inv = [PX.invader(self.nprng, int(r.choice([9, 11, 13, 15])), [self.col, self.col2, (255, 255, 255)][: r.choice([2, 3])]) for _ in range(2)]
            self.swarm = r.random() < .5
        if kind == "pglyph":
            self.gl = PX.pixel_glyph(r.choice(GLYPHS), r.choice([12, 14, 16]), self.col) or PX.invader(self.nprng, 11, [self.col, self.col2])
        self.dir = r.choice([-1, 1])

    def apply(self, img, t, beat, u):
        if not self.kind: return img
        W, H = self.w, self.h; k = self.kind; kick = 1 if beat > .6 else 0
        if k in ("dither", "window"):
            R = W * (.3 + .06 * kick) * min(1.0, u * 5); m = pixel_mask(W, H, self.shape, self.x * W, self.y * H, R)
            if k == "window": return Image.composite(ImageChops.invert(img), img, m)
            return PX.dither_region(img, m, self.pal, cell=10)
        if k == "band":
            m = Image.new("L", (W // 10, H // 10), 0); d = ImageDraw.Draw(m)
            for q in range(2):
                y = ((u * .9 + q * .45) % 1.1 - .05) * H / 10; d.rectangle([0, y, W, y + (18 + 8 * q) * (1 + .3 * kick)], fill=255)
            return PX.dither_region(img, m.resize((W, H), Image.NEAREST), self.pal, cell=10)
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        if k == "sprite" and self.spr is not None:
            s = self.spr if int(t * 4) % 2 == 0 else self.spr2                        # two-frame wiggle
            scale = max(4, int(W * .62 / s.width)); big = PX.up(s, scale)
            jump = -scale * 2 * kick; x = int(self.x * W - big.width / 2); y = int(self.y * H - big.height / 2 + jump)
            ov.alpha_composite(big, (x, y))
        elif k == "invader" and self.inv:
            fr = self.inv[int(t * 3) % 2]
            if self.swarm:                                                          # a marching formation, stepping on the beat
                scale = 9; sp = PX.up(fr, scale); cols, rows = 4, 3; step = int(t * 2.5) % 8
                for j in range(rows):
                    for i in range(cols):
                        x = int(W * .1 + i * (sp.width + 30) + self.dir * step * 12); y = int(H * .2 + j * (sp.height + 26) + u * 120)
                        if -sp.width < x < W: ov.alpha_composite(sp, (x, y))
            else:
                scale = max(6, int(W * .6 / fr.width)); big = PX.up(fr, scale); ov.alpha_composite(big, (int(self.x * W - big.width / 2), int(self.y * H - big.height / 2 - scale * kick)))
        elif k == "pglyph" and self.gl is not None:
            scale = max(8, int(W * .66 / self.gl.width)) + 2 * kick; big = PX.up(self.gl, scale)
            ov.alpha_composite(big, (int(self.x * W - big.width / 2), int(self.y * H - big.height / 2)))
        img = img.convert("RGBA"); img.alpha_composite(ov); return img.convert("RGB")
