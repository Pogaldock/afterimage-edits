#!/usr/bin/env python3
"""Gaussian splats from a single photo, on a CPU (moderngl / llvmpipe).

Every pixel of the photo becomes a 3D Gaussian: its position comes from the monocular depth map (Depth Anything V2 Small,
already used elsewhere), its colour from the pixel, its size from the pixel's footprint at that depth. The camera then
moves through the photo as if it were a real space. Splats are sorted back to front every frame and alpha-blended with
the Gaussian falloff alpha = o * exp(-d^2 / 2 sigma^2), the way 3D Gaussian splatting renders (Kerbl et al. 2023;
shader approach after antimatter15/splat, MIT).

"4D" without training: each splat also carries a seed, and a dissolve amount D pushes the splats along a smooth flow field,
grows them and fades them, so the photo can come apart into drifting light and pull itself back together on the beat.
"""
import math
import numpy as np
from PIL import Image

VS = """#version 330
uniform vec2 R; uniform float T, B, D, F, CAMZ, YAW, PITCH, ROLL;
in vec3 pos; in vec3 colr; in float sc; in float sd;
out vec3 col; out float alpha;
mat3 ry(float a){ float c=cos(a), s=sin(a); return mat3(c,0.,-s, 0.,1.,0., s,0.,c); }
mat3 rx(float a){ float c=cos(a), s=sin(a); return mat3(1.,0.,0., 0.,c,s, 0.,-s,c); }
mat3 rz(float a){ float c=cos(a), s=sin(a); return mat3(c,s,0., -s,c,0., 0.,0.,1.); }
void main(){
  vec3 p = pos;
  float d = clamp(D + .35*B*D, 0., 1.2);
  vec3 fl = vec3(sin(p.y*3.1 + T*.9 + sd*6.3), sin(p.z*2.7 + T*1.1 + sd*4.1), sin(p.x*3.3 + T*.7 + sd*5.2))
          + .5*vec3(sin(p.z*7.+T*1.7), sin(p.x*6.+T*1.3), sin(p.y*8.+T*1.9));              // a smooth, curling flow
  p += fl * d * d * (.35 + .7*sd);                                                       // eases in: intact at first, then it comes apart
  p = rz(ROLL) * rx(PITCH) * ry(YAW) * p;
  p.z += CAMZ;
  if (p.z < .05) { gl_Position = vec4(2.,2.,2.,1.); gl_PointSize = 0.; col = vec3(0.); alpha = 0.; return; }
  vec2 s = p.xy / p.z * F; s.x *= R.y / R.x;
  gl_Position = vec4(s, 0., 1.);
  gl_PointSize = clamp(sc * (1. + 1.4*d*d) / p.z * F * R.y * .5, 1., 96.);
  col = colr * (1. + .25*B);
  alpha = clamp(.97 - .35*d, .2, 1.);
}"""
FS = """#version 330
in vec3 col; in float alpha; out vec4 o;
void main(){ vec2 c = (gl_PointCoord - .5) * 2.; float r2 = dot(c, c); if (r2 > 1.) discard;
  float a = alpha * exp(-r2 * 2.4); o = vec4(col * a, a); }"""


def _rot(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)       # same matrices as the shader (GLSL is column-major)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float32)
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], np.float32)
    return Rz @ Rx @ Ry


class Splats:
    def __init__(self, ctx, w, h, side=300):
        self.ctx, self.w, self.h, self.side = ctx, w, h, side
        self.prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
        self.fb = ctx.framebuffer(color_attachments=[ctx.texture((w, h), 4)])
        self.key = None; self.vao = None; self.n = 0; self.F = 1.6
        self.dissolve = 0.0; self.move = 0

    def load(self, im, depth, key):
        """lift a photo (PIL) and its depth map (PIL L, bright = near) to splats"""
        if key == self.key: return
        self.key = key
        aspect = self.w / self.h; gh = int(self.side / aspect) if aspect < 1 else self.side; gw = int(gh * aspect)
        cut = im.mode == "RGBA" and im.split()[3].getextrema()[0] < 200          # a cut-out: centred and contained, empty pixels dropped
        fit = _contain if cut else _cover
        rgba = np.asarray(fit(im.convert("RGBA"), gw, gh), np.float32) / 255; rgb = rgba[..., :3]
        dep = np.asarray(fit(depth.convert("L"), gw, gh), np.float32) / 255
        z = 1.0 + (1.0 - dep) * 2.2                                            # inverse depth → distance (near 1, far 3.2)
        u, v = np.meshgrid((np.arange(gw) + .5) / gw - .5, .5 - (np.arange(gh) + .5) / gh)
        x = u * aspect * 2 * z / self.F; y = v * 2 * z / self.F                    # unproject so the first frame matches the photo
        pos = np.stack([x, y, z - 2.0], -1).reshape(-1, 3)                       # centred on the scene, camera sits at CAMZ
        foot = 2.0 / gh / self.F * z * 2.6                                        # pixel footprint at that depth, a bit of overlap
        rng = np.random.default_rng(int(abs(hash(key)) % 2 ** 31))
        pos += rng.normal(0, .003, pos.shape).astype(np.float32)
        data = np.concatenate([pos, rgb.reshape(-1, 3), foot.reshape(-1, 1), rng.random((pos.shape[0], 1))], 1).astype("f4")
        if cut:
            keep = rgba[..., 3].reshape(-1) > .4; data = data[keep]; pos = pos[keep]
            if len(pos) == 0: self.vao = None; return
        self.pos = pos.astype(np.float32); self.n = len(pos)
        if self.vao is not None: self.vao.release(); self.vbo.release(); self.ibo.release()
        self.vbo = self.ctx.buffer(data.tobytes()); self.ibo = self.ctx.buffer(reserve=self.n * 4)
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "3f 3f 1f 1f", "pos", "colr", "sc", "sd")], index_buffer=self.ibo, index_element_size=4)

    def camera(self, t, beat):
        """slow dolly into the photo plus a gentle orbit; move 1 = a faster fly-through, 2 = a roll"""
        k = [.35, .9, .5][self.move % 3]
        camz = 2.0 - min(1.1, .12 * t * k) - .05 * beat
        yaw = .22 * math.sin(t * .31 * (1 + k)); pitch = .08 * math.sin(t * .23); roll = .25 * math.sin(t * .2) if self.move == 2 else 0.0
        return camz, yaw, pitch, roll

    def render(self, t, beat):
        if self.vao is None: return self.fb.color_attachments[0]
        camz, yaw, pitch, roll = self.camera(t, beat)
        zv = (self.pos @ _rot(yaw, pitch, roll).T)[:, 2] + camz
        self.ibo.write(np.argsort(-zv).astype("u4").tobytes())                 # far to near
        p = self.prog
        for k_, v_ in (("R", (float(self.w), float(self.h))), ("T", float(t)), ("B", float(beat)), ("D", float(self.dissolve)), ("F", self.F), ("CAMZ", camz), ("YAW", yaw), ("PITCH", pitch), ("ROLL", roll)):
            if k_ in p: p[k_].value = v_
        self.fb.use(); self.ctx.clear(0, 0, 0, 0)
        self.ctx.enable(self.ctx.BLEND | self.ctx.PROGRAM_POINT_SIZE); self.ctx.blend_func = (self.ctx.ONE, self.ctx.ONE_MINUS_SRC_ALPHA)
        self.vao.render(mode=self.ctx.POINTS); self.ctx.disable(self.ctx.BLEND)
        return self.fb.color_attachments[0]


def _contain(im, w, h, k=.86):
    """centre the image inside w x h at k of the frame, transparent (or black, for L) around it"""
    r = min(w / im.width, h / im.height) * k; sm = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.BILINEAR)
    out = Image.new(im.mode, (w, h), 0); out.paste(sm, ((w - sm.width) // 2, (h - sm.height) // 2)); return out

def _cover(im, w, h):
    r = max(w / im.width, h / im.height); im = im.resize((max(w, int(im.width * r + .5)), max(h, int(im.height * r + .5))), Image.BILINEAR)
    x, y = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((x, y, x + w, y + h))
