#!/usr/bin/env python3
"""Readable meaning layer drawn crisp over the edit: data streams, readouts tied to the real track, a spectrum of the actual
audio, a sigil built from the song title, esoteric glyphs (planets, zodiac, I Ching, runes, alchemy) and sacred geometry,
and a target box that locks onto the brightest part of the picture. One accent colour plus white, thin lines, kept to the
margins and to a few elements at a time so the picture underneath stays easy to follow.

    H = HUD(1080, 1920, rng, title="Gonzi - FULL POWER", bpm=140, accent=(0,255,170))
    H.plan(kinds)                                       # which elements this section shows (see KINDS)
    rgba = H.frame(t, beat, spectrum_row, picture)      # transparent overlay for one frame
"""
import math, os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

KINDS = ["stream", "readout", "spectrum", "sigil", "glyphs", "geometry", "target"]

PLANETS = "☉☽☿♀♂♃♄♅♆"
ZODIAC = "♈♉♊♋♌♍♎♏♐♑♒♓"
TRIGRAMS = "☰☱☲☳☴☵☶☷"
HEXAGRAMS = "".join(chr(0x4DC0 + i) for i in range(64))
RUNES = "".join(chr(c) for c in range(0x16A0, 0x16F1))
ALCHEMY = "".join(chr(c) for c in range(0x1F700, 0x1F774))
NAMES = {"☉": "SOL", "☽": "LUNA", "☿": "MERCURIUS", "♀": "VENUS", "♂": "MARS", "♃": "JUPITER", "♄": "SATURNUS", "♅": "URANUS", "♆": "NEPTUNUS"}

FONT_DIRS = ["/usr/share/fonts/truetype/noto", "/usr/share/fonts/opentype/noto", "/usr/share/fonts/truetype/freefont", "/usr/share/fonts/truetype/dejavu"]
FONT_FILES = ["NotoSansSymbols-Regular.ttf", "NotoSansSymbols2-Regular.ttf", "NotoSansRunic-Regular.ttf", "FreeMono.ttf", "FreeSerif.ttf", "DejaVuSans.ttf"]
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

_cmaps = None
def _fonts():
    """every symbol font found on the machine, with the set of characters each can draw"""
    global _cmaps
    if _cmaps is None:
        _cmaps = []
        from fontTools.ttLib import TTFont
        for d in FONT_DIRS:
            for f in FONT_FILES:
                p = os.path.join(d, f)
                if os.path.exists(p):
                    try: _cmaps.append((p, set(TTFont(p).getBestCmap())))
                    except Exception: pass
    return _cmaps

_fcache = {}
def glyph_font(ch, size):
    for p, cm in _fonts():
        if ord(ch) in cm:
            k = (p, size)
            if k not in _fcache: _fcache[k] = ImageFont.truetype(p, size)
            return _fcache[k]
    return None

def drawable(chars): return [c for c in chars if any(ord(c) in cm for _, cm in _fonts())]

def mono(size):
    k = ("mono", size)
    if k not in _fcache:
        try: _fcache[k] = ImageFont.truetype(MONO, size)
        except Exception: _fcache[k] = ImageFont.load_default()
    return _fcache[k]

def sigil_points(title):
    """chaos-magic style: the title's consonants without repeats, each a point on a circle of 26 → one path"""
    s = []
    for ch in title.upper():
        if "A" <= ch <= "Z" and ch not in "AEIOU" and ch not in s: s.append(ch)
    s = s[:9] or ["S", "Y", "M"]
    return [(math.cos(-math.pi / 2 + 2 * math.pi * (ord(c) - 65) / 26), math.sin(-math.pi / 2 + 2 * math.pi * (ord(c) - 65) / 26)) for c in s], "".join(s)

class HUD:
    def __init__(self, w, h, rng, title="", bpm=130.0, accent=(0, 255, 170)):
        self.w, self.h, self.r = w, h, rng; self.title = title or "untitled"; self.bpm = bpm or 130.0; self.acc = tuple(accent)
        self.kinds = []; self.sig, self.sigtxt = sigil_points(self.title)
        pool = drawable(PLANETS + ZODIAC + TRIGRAMS + HEXAGRAMS[:16] + RUNES[:24] + ALCHEMY[:40])
        self.glyphpool = pool or list("+x*#")
        self.streamchars = drawable("0123456789ABCDEF" + RUNES[:12] + TRIGRAMS) or list("0123456789ABCDEF")
        self.cols = []; self.glyphs = []; self.geo = 0; self.section = 0; self.seed = rng.random()

    def plan(self, kinds, section=0):
        self.kinds = list(kinds); self.section = section; r = self.r
        self.glyphs = [(r.choice(self.glyphpool), r.uniform(.12, .88), r.uniform(.2, .66), r.choice([90, 120, 160])) for _ in range(r.choice([1, 2, 3]))]
        self.geo = r.randrange(4); self.side = r.choice(["L", "R", "LR"])
        self.cols = [[r.choice(self.streamchars) + r.choice(self.streamchars) for _ in range(60)] for _ in range(2)]

    def frame(self, t, beat, spec=None, pic=None):
        W, H = self.w, self.h; im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        A = self.acc; WHITE = (255, 255, 255); a_line = int(150 + 90 * beat)
        if "stream" in self.kinds:                          # columns of hex and runes scrolling at the tempo, the current row lit
            f = mono(26); lh = 34; speed = self.bpm / 60 * lh * .5
            for ci, side in enumerate(["L", "R"]):
                if side not in self.side: continue
                x = 34 if side == "L" else W - 34 - 44; off = (t * speed) % lh; n0 = int(t * speed // lh)
                for k in range(int(H * .62 / lh)):
                    y = H * .14 + k * lh - off; s = self.cols[ci][(n0 + k) % len(self.cols[ci])]
                    lit = k == int(H * .3 / lh)
                    for j, ch in enumerate(s):
                        gf = glyph_font(ch, 26) if ord(ch) > 127 else f
                        if gf: d.text((x + j * 20, y), ch, font=gf, fill=(A if lit else WHITE) + ((255 if lit else 95),))
        if "readout" in self.kinds:                         # real numbers: tempo, time, section, the sigil's letters
            f = mono(24); y0 = H * .155
            lines = [f"BPM {self.bpm:6.1f}", f"T+{int(t // 60):02d}:{t % 60:05.2f}", f"SEC {self.section:02d}", f"SIG {self.sigtxt}", f"0x{int((t * 997 + self.seed * 1e6)) & 0xFFFFFF:06X}"]
            for k, s in enumerate(lines): d.text((W - 300, y0 + k * 32), s, font=f, fill=(A if k == 0 and beat > .5 else WHITE) + (210,))
            d.line([(W - 310, y0 - 8), (W - 310, y0 + 5 * 32)], fill=A + (180,), width=2)
        if "spectrum" in self.kinds and spec is not None:   # the actual audio, as thin bars above the captions
            n = len(spec); bw = (W * .72) / n; x0 = W * .14; yb = H * .69
            for k, v in enumerate(spec):
                hh = 6 + 120 * float(v); d.rectangle([x0 + k * bw + 2, yb - hh, x0 + (k + 1) * bw - 2, yb], fill=(A if v > .6 else WHITE) + (200,))
            d.line([(x0, yb + 4), (x0 + W * .72, yb + 4)], fill=WHITE + (120,), width=1)
        if "sigil" in self.kinds:                           # the song title as a sigil, slowly turning, drawn on with the beat
            cx, cy, R = W / 2, H * .44, W * .3; rot = t * .25 + self.seed * 6
            pts = [(cx + R * (x * math.cos(rot) - y * math.sin(rot)), cy + R * (x * math.sin(rot) + y * math.cos(rot))) for x, y in self.sig]
            d.ellipse([cx - R * 1.08, cy - R * 1.08, cx + R * 1.08, cy + R * 1.08], outline=WHITE + (110,), width=2)
            prog = min(1.0, (t % 4) / 2.0 + .25); m = max(2, int(len(pts) * prog))
            if len(pts) > 1: d.line(pts[:m], fill=A + (a_line,), width=5, joint="curve")
            x, y = pts[0]; d.ellipse([x - 12, y - 12, x + 12, y + 12], outline=A + (230,), width=4)
            if m >= len(pts):
                x, y = pts[-1]; dx, dy = x - cx, y - cy; L = math.hypot(dx, dy) or 1; nx, ny = -dy / L * 18, dx / L * 18; d.line([(x - nx, y - ny), (x + nx, y + ny)], fill=A + (230,), width=5)
        if "glyphs" in self.kinds:                          # one to three esoteric glyphs, large and thin, pulsing on the kick
            for ch, gx, gy, sz in self.glyphs:
                gf = glyph_font(ch, int(sz * (1 + .15 * beat)))
                if gf is None: continue
                X, Y = gx * W, gy * H
                d.text((X, Y), ch, font=gf, fill=WHITE + (int(170 + 85 * beat),), anchor="mm")
                lab = NAMES.get(ch, f"U+{ord(ch):04X}"); d.text((X, Y + sz * .75), lab, font=mono(20), fill=A + (200,), anchor="mm")
        if "geometry" in self.kinds:                        # sacred geometry over the picture: flower of life, metatron, star tetrahedron, nested polygons
            cx, cy = W / 2, H * .44; R = W * .13 * (1 + .04 * beat); col = WHITE + (120,)
            if self.geo == 0:
                for k in range(7):
                    ox, oy = (0, 0) if k == 0 else (R * math.cos(k * math.pi / 3 + t * .1), R * math.sin(k * math.pi / 3 + t * .1))
                    d.ellipse([cx + ox - R, cy + oy - R, cx + ox + R, cy + oy + R], outline=col, width=2)
            elif self.geo == 1:
                P = [(cx, cy)] + [(cx + R * 1.7 * math.cos(k * math.pi / 3 + t * .1), cy + R * 1.7 * math.sin(k * math.pi / 3 + t * .1)) for k in range(6)]
                P += [(cx + R * .85 * math.cos(k * math.pi / 3 + t * .1), cy + R * .85 * math.sin(k * math.pi / 3 + t * .1)) for k in range(6)]
                for i in range(len(P)):
                    for j in range(i + 1, len(P)): d.line([P[i], P[j]], fill=WHITE + (55,), width=1)
                for x, y in P: d.ellipse([x - R * .3, y - R * .3, x + R * .3, y + R * .3], outline=A + (150,), width=2)
            elif self.geo == 2:
                for s_ in (1, -1):
                    pts = [(cx + R * 2 * math.cos(-math.pi / 2 * s_ + k * 2 * math.pi / 3 + t * .15 * s_), cy + R * 2 * math.sin(-math.pi / 2 * s_ + k * 2 * math.pi / 3 + t * .15 * s_)) for k in range(3)]
                    d.polygon(pts, outline=(A if s_ > 0 else WHITE) + (170,), width=3)
                d.ellipse([cx - R * 2, cy - R * 2, cx + R * 2, cy + R * 2], outline=col, width=2)
            else:
                for n in range(3, 9):
                    rr = R * (.5 + .28 * (n - 3)); pts = [(cx + rr * math.cos(k * 2 * math.pi / n + t * .2 * (1 if n % 2 else -1)), cy + rr * math.sin(k * 2 * math.pi / n + t * .2 * (1 if n % 2 else -1))) for k in range(n)]
                    d.polygon(pts, outline=(A if n == 3 + int(t * 2) % 6 else WHITE) + (140,), width=2)
        if "target" in self.kinds and pic is not None:      # a lock on the brightest part of the picture, with a label
            g = np.asarray(pic.convert("L").resize((27, 48)), np.float32); k = int(np.argmax(g)); gy, gx = divmod(k, 27)
            X, Y = (gx + .5) * W / 27, (gy + .5) * H / 48; X = min(max(X, 160), W - 160); Y = min(max(Y, H * .18), H * .66)
            s = 120 + 30 * beat; c = 34
            for sx in (-1, 1):
                for sy in (-1, 1):
                    d.line([(X + sx * s, Y + sy * s), (X + sx * (s - c), Y + sy * s)], fill=A + (230,), width=4)
                    d.line([(X + sx * s, Y + sy * s), (X + sx * s, Y + sy * (s - c))], fill=A + (230,), width=4)
            ch = self.glyphs[0][0] if self.glyphs else "+"; gf = glyph_font(ch, 30)
            lab = f"{NAMES.get(ch, 'OBJ')} {int(X):04d}.{int(Y):04d}"
            d.text((X - s, Y + s + 12), lab, font=mono(22), fill=WHITE + (220,))
            if gf: d.text((X + s - 30, Y - s - 40), ch, font=gf, fill=A + (230,))
        return pixelate(im, 4)

def pixelate(im, k):
    """turn the overlay into hard k×k pixels: each block takes its strongest pixel, then nearest-neighbour back up"""
    a = np.asarray(im); h, w = a.shape[0] // k * k, a.shape[1] // k * k; b = a[:h, :w].reshape(h // k, k, w // k, k, 4)
    al = b[..., 3]; idx = al.reshape(h // k, k, w // k, k).transpose(0, 2, 1, 3).reshape(h // k, w // k, k * k).argmax(-1)
    blk = b.transpose(0, 2, 1, 3, 4).reshape(h // k, w // k, k * k, 4); pick = np.take_along_axis(blk, idx[..., None, None], 2)[:, :, 0]
    pick[..., 3] = np.where(pick[..., 3] > 60, 255, 0)
    out = Image.fromarray(pick.astype(np.uint8)).resize((w, h), Image.NEAREST)
    if (w, h) != im.size: o2 = Image.new("RGBA", im.size, (0, 0, 0, 0)); o2.paste(out, (0, 0)); out = o2
    return out
