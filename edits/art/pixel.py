#!/usr/bin/env python3
"""Pixel-art elements made from the edit's own material, never pasted vectors:

    sprite_from(im)       a cut-out, meme or photo crushed into a pixel sprite: few colours, ordered dither, black outline
    invader(rng)          a procedural creature/glyph (mirror-symmetric cellular sprite, after Jared Tarbell's invaders)
    pixel_glyph(ch)       an esoteric glyph rendered tiny with no anti-aliasing, so it comes out as chunky pixels
    dither_region(img..)  part of the frame turned into 2-4 colour ordered-dither pixels (Bayer 4x4)

All are drawn small and scaled up with nearest-neighbour, so every pixel is a hard square.
"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) + .5) / 16

def dither(a, levels):
    """ordered dither of a float image 0..1 (H,W) or (H,W,3) into `levels` steps"""
    h, w = a.shape[:2]; t = np.tile(BAYER, (h // 4 + 1, w // 4 + 1))[:h, :w]
    if a.ndim == 3: t = t[..., None]
    return np.floor(a * (levels - 1) + t) / (levels - 1)

def quantize_to(im, palette):
    """map an RGB image onto a fixed palette with ordered dithering (palette sorted dark → light by luminance)"""
    pal = np.array(sorted(palette, key=lambda c: .3 * c[0] + .59 * c[1] + .11 * c[2]), np.float32)
    L = np.asarray(im.convert("L"), np.float32) / 255; idx = np.clip(dither(L, len(pal)) * (len(pal) - 1), 0, len(pal) - 1).astype(int)
    return Image.fromarray(pal[idx].astype(np.uint8))

def sprite_from(im, height=40, palette=None, outline=(0, 0, 0)):
    """crush a cut-out (with alpha) or any image into a pixel sprite of `height` pixels"""
    im = im.convert("RGBA"); r_ = height / im.height; small = im.resize((max(4, int(im.width * r_)), height), Image.BOX)
    a = np.asarray(small.split()[3]) > 110
    if not a.any() or a.mean() > .97:                                       # no real alpha: keep the brighter half as the shape
        L = np.asarray(small.convert("L"), np.float32); a = L > np.percentile(L, 45)
    rgb = small.convert("RGB")
    if palette: rgb = quantize_to(rgb, palette)
    else: rgb = Image.fromarray((dither(np.asarray(rgb, np.float32) / 255, 4) * 255).astype(np.uint8))
    out = np.zeros((a.shape[0] + 2, a.shape[1] + 2, 4), np.uint8); out[1:-1, 1:-1, :3] = np.asarray(rgb); out[1:-1, 1:-1, 3] = a * 255
    m = np.pad(a, 1); ring = (np.roll(m, 1, 0) | np.roll(m, -1, 0) | np.roll(m, 1, 1) | np.roll(m, -1, 1)) & ~m
    out[ring] = list(outline) + [255]
    return Image.fromarray(out)

def invader(rng, size=13, cols=None):
    """mirror-symmetric creature: random half mask, two passes of cellular smoothing, mirrored, outlined"""
    half = (size + 1) // 2; m = rng.random((size, half)) < .48
    for _ in range(2):
        n = sum(np.roll(np.roll(m, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) - m
        m = (m & (n >= 2)) | (~m & (n >= 5))
    m[:, 0] = False if rng.random() < .3 else m[:, 0]
    full = np.concatenate([m[:, ::-1], m[:, 1:] if size % 2 else m], 1)
    cols = cols or [(255, 255, 255), (255, 40, 40)]
    out = np.zeros((size + 2, full.shape[1] + 2, 4), np.uint8); body = np.pad(full, 1)
    shade = np.pad(rng.random(full.shape) < .25, 1) & body
    out[body] = list(cols[0]) + [255]; out[shade] = list(cols[1 % len(cols)]) + [255]
    ring = (np.roll(body, 1, 0) | np.roll(body, -1, 0) | np.roll(body, 1, 1) | np.roll(body, -1, 1)) & ~body; out[ring] = [0, 0, 0, 255]
    eyes = rng.integers(2, max(3, size // 2)); ey = rng.integers(2, max(3, size // 2))     # two dark eye pixels, mirrored
    for x in (eyes, out.shape[1] - 1 - eyes):
        if body[ey, x]: out[ey, x] = [0, 0, 0, 255]
    return Image.fromarray(out)

_gf = {}
def pixel_glyph(ch, px=14, col=(255, 255, 255)):
    """an esoteric glyph drawn at ~14 px with no anti-aliasing → chunky pixel glyph"""
    import hud
    f = hud.glyph_font(ch, px)
    if f is None: return None
    im = Image.new("L", (px * 2, px * 2), 0); d = ImageDraw.Draw(im); d.fontmode = "1"; d.text((px, px), ch, font=f, fill=255, anchor="mm")
    bb = im.getbbox()
    if not bb: return None
    im = im.crop(bb); a = np.asarray(im) > 0; out = np.zeros((a.shape[0] + 2, a.shape[1] + 2, 4), np.uint8); m = np.pad(a, 1)
    out[m] = list(col) + [255]; ring = (np.roll(m, 1, 0) | np.roll(m, -1, 0) | np.roll(m, 1, 1) | np.roll(m, -1, 1)) & ~m; out[ring] = [0, 0, 0, 255]
    return Image.fromarray(out)

def up(im, scale):
    return im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.NEAREST)

def dither_region(img, mask, palette, cell=8):
    """inside `mask` (L image), the frame becomes big dithered pixels in the given palette"""
    W, H = img.size; small = img.resize((W // cell, H // cell), Image.BOX)
    px = up(quantize_to(small, palette), cell).resize((W, H), Image.NEAREST)
    return Image.composite(px, img, mask)
