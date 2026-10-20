#!/usr/bin/env python3
"""Bold layer: flat, saturated, thick-lined graphics that cut through the picture. One or two strong colours per edit,
over a mostly black-and-white image, so the colour reads instantly. Each section gets at most one element:

    symbol   a big recognisable symbol (smiley, alien, eye, star...) in flat colour, popping on the kick
    glyph    a huge esoteric glyph (planet, zodiac, hexagram) in flat colour
    rings    thick concentric rings thrown out by the kick
    bars     two or three thick diagonal bands sweeping across
    chevrons stacked arrows climbing the frame
    frame    a heavy border with a crosshair
    window   a solid circle or diamond where the picture turns negative (difference blend)

    B = Bold(1080, 1920, rng, colours)
    B.plan(kind)                                  # or None for nothing this section
    img = B.apply(img, t, beat, u)                # u = 0..1 progress through the section
"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageChops, ImageFont

PALETTE = [(255, 40, 40), (255, 225, 0), (0, 90, 255), (0, 255, 140), (255, 0, 170), (255, 110, 0), (0, 225, 255)]
KINDS = ["symbol", "symbol", "glyph", "rings", "bars", "chevrons", "frame", "window", "window"]
GLYPHS = "☉☽☿♀♂♃♄♈♉♊♋♌♍♎♏♐♑♒♓☰☷䷀䷁䷊䷋"

class Bold:
    def __init__(self, w, h, rng, colours=None):
        self.w, self.h, self.r = w, h, rng
        self.cols = colours or rng.sample(PALETTE, 2); self.kind = None; self._sym = {}
        try: self.gfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 900)
        except Exception: self.gfont = None

    def plan(self, kind, symbols=None):
        r = self.r; self.kind = kind; self.col = r.choice(self.cols); self.col2 = [c for c in self.cols if c != self.col][0] if len(self.cols) > 1 else (255, 255, 255)
        self.x, self.y = r.choice([(.5, .45), (.5, .42), (.32, .38), (.68, .5)]); self.rot0 = r.uniform(-15, 15); self.spin = r.choice([0, 0, 25, -25])
        self.outline = r.random() < .3; self.shape = r.choice(["circle", "diamond", "square"])
        self.symname = r.choice(symbols or ["smiley", "alien", "eye", "star", "trefoil", "peace", "anarchy", "yinyang", "sun", "ufo", "23", "pyramid"])
        self.glyph = r.choice(GLYPHS); self.angle = r.choice([-30, -20, 20, 30])

    def _symbol(self, size):
        k = (self.symname, size, self.col, self.outline)
        if k not in self._sym:
            import symbols as SYM
            fg = self.col + (255,); bg = (0, 0, 0, 255)
            im = SYM.symbol(self.symname, size, fg, bg)
            if self.outline:                                               # outline only: keep the edge, drop the fill
                a = np.asarray(im.split()[3]); from PIL import ImageFilter
                e = np.asarray(Image.fromarray(a).filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(9)))
                im = Image.new("RGBA", im.size, self.col + (0,)); im.putalpha(Image.fromarray(e))
            self._sym = {k: im}
        return self._sym[k]

    def apply(self, img, t, beat, u):
        if not self.kind: return img
        W, H = self.w, self.h; k = self.kind
        pop = 1 + .12 * beat; enter = min(1.0, u * 6)                    # quick entrance, then it holds and breathes on the kick
        if k == "window":                                                  # the picture turns negative inside a bold shape
            m = Image.new("L", (W, H), 0); d = ImageDraw.Draw(m); R = W * .36 * pop * enter; cx, cy = self.x * W, self.y * H
            if self.shape == "circle": d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=255)
            elif self.shape == "diamond": d.polygon([(cx, cy - R * 1.2), (cx + R, cy), (cx, cy + R * 1.2), (cx - R, cy)], fill=255)
            else:
                a = math.radians(self.rot0 + t * self.spin); pts = [(cx + R * math.cos(a + q * math.pi / 2), cy + R * math.sin(a + q * math.pi / 2)) for q in range(4)]; d.polygon(pts, fill=255)
            inv = ImageChops.invert(img); out = Image.composite(inv, img, m)
            d2 = ImageDraw.Draw(out)                                       # a bold coloured edge around the negative window
            if self.shape == "circle": d2.ellipse([cx - R, cy - R, cx + R, cy + R], outline=self.col, width=16)
            elif self.shape == "diamond": d2.polygon([(cx, cy - R * 1.2), (cx + R, cy), (cx, cy + R * 1.2), (cx - R, cy)], outline=self.col, width=16)
            return out
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov); C = self.col + (255,)
        if k == "symbol":
            s = int(W * .62 * pop * (.4 + .6 * enter)); sym = self._symbol(int(W * .62)).resize((s, s), Image.BICUBIC).rotate(self.rot0 + t * self.spin, resample=Image.BICUBIC, expand=True)
            ov.alpha_composite(sym, (int(self.x * W - sym.width / 2), int(self.y * H - sym.height / 2)))
        elif k == "glyph" and self.gfont is not None:
            g = Image.new("RGBA", (1100, 1100), (0, 0, 0, 0)); ImageDraw.Draw(g).text((550, 560), self.glyph, font=self.gfont, fill=C, anchor="mm")
            s = int(W * .8 * pop * (.4 + .6 * enter)); g = g.resize((s, s), Image.BICUBIC).rotate(self.rot0, resample=Image.BICUBIC)
            ov.alpha_composite(g, (int(self.x * W - s / 2), int(self.y * H - s / 2)))
        elif k == "rings":
            cx, cy = self.x * W, self.y * H
            for q in range(5):
                R = ((t * .9 + q * .2) % 1.0) * W * 1.1 + 40 * beat; w_ = int(26 + 30 * beat)
                d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(self.col if q % 2 == 0 else self.col2) + (255,), width=w_)
        elif k == "bars":
            a = math.radians(self.angle); dx, dy = math.cos(a), math.sin(a); L = W * 2.2
            for q in range(3):
                off = ((u * 1.4 + q * .33) % 1.4 - .7) * H; th = 70 + 50 * (q == 1) + 30 * beat
                cx, cy = W / 2 - dy * off, H / 2 + dx * off; nx, ny = -dy * th / 2, dx * th / 2
                d.polygon([(cx - dx * L + nx, cy - dy * L + ny), (cx + dx * L + nx, cy + dy * L + ny), (cx + dx * L - nx, cy + dy * L - ny), (cx - dx * L - nx, cy - dy * L - ny)], fill=(self.col if q != 1 else self.col2) + (255,))
        elif k == "chevrons":
            cx = self.x * W; step = 150; off = (t * 260) % step
            for q in range(14):
                y = H - q * step + off - 100; w_ = W * .32
                if y < H * .12 or y > H * .9: continue
                d.line([(cx - w_, y + w_ * .45), (cx, y), (cx + w_, y + w_ * .45)], fill=(self.col if q % 2 else self.col2) + (255,), width=int(34 + 20 * beat), joint="curve")
        elif k == "frame":
            m = int(40 + 20 * beat); w_ = 34
            d.rectangle([m, m + H * .12, W - m, H * .7], outline=C, width=w_)
            cx, cy = W / 2, H * .41; L = W * .12 * pop
            d.line([(cx - L, cy), (cx + L, cy)], fill=self.col2 + (255,), width=18); d.line([(cx, cy - L), (cx, cy + L)], fill=self.col2 + (255,), width=18)
        img = img.convert("RGBA"); img.alpha_composite(ov); return img.convert("RGB")
