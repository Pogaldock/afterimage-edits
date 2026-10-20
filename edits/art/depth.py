#!/usr/bin/env python3
"""Monocular depth for atlas photos: Depth Anything V2 Small (Apache-2.0), ONNX on the CPU, ~1 s per image at 518 px.
Returns an 8-bit greyscale PIL image, white = near. Falls back to a blurred-luminance guess when the model can't be loaded
(local tests without network), so the rest of the engine never breaks.

    d = depth_of(pil_image)                      # PIL "L", same aspect as the input (long side 518)
"""
import os
import numpy as np
from PIL import Image, ImageFilter

REPO, FILE = "onnx-community/depth-anything-v2-small", "onnx/model.onnx"
_sess = None; _failed = False; _cache = {}

def _session():
    global _sess, _failed
    if _sess is not None or _failed: return _sess
    try:
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort
        path = hf_hub_download(REPO, FILE)
        so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count() or 2
        _sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    except Exception as e:
        print("depth model unavailable, using luminance fallback:", type(e).__name__, str(e)[:120]); _failed = True
    return _sess

def _fallback(im):
    g = im.convert("L").filter(ImageFilter.GaussianBlur(6)); a = np.asarray(g, np.float32)
    yy = np.linspace(0, 1, a.shape[0], dtype=np.float32)[:, None]                 # lower in frame tends to be nearer
    d = .65 * a / 255 + .35 * yy; d = (d - d.min()) / (np.ptp(d) + 1e-6)
    return Image.fromarray((d * 255).astype(np.uint8))

def depth_of(im, key=None):
    if key is not None and key in _cache: return _cache[key]
    im = im.convert("RGB"); s = _session()
    if s is None: out = _fallback(im)
    else:
        x = np.asarray(im.resize((518, 518), Image.BICUBIC), np.float32) / 255
        x = (x - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        name = s.get_inputs()[0].name; d = s.run(None, {name: x})[0][0]
        d = (d - d.min()) / (np.ptp(d) + 1e-6)
        out = Image.fromarray((d * 255).astype(np.uint8)).resize(im.size if max(im.size) <= 518 else (int(518 * im.width / max(im.size)), int(518 * im.height / max(im.size))), Image.BICUBIC)
        out = out.filter(ImageFilter.GaussianBlur(1.2))
    if key is not None: _cache[key] = out
    return out
