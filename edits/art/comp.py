#!/usr/bin/env python3
"""Compositor: the VJ engine behind the edits.

Each section of an edit is a small graph, in the spirit of live VJ tools (Hydra, TouchDesigner): sources (photos, cut-outs,
video clips, static, noise, a point cloud, the previous frame) go through coordinate ops (modulate one source by another,
non-linear stretches, kaleidoscopes, polar warps, repeats), colour ops (luma keys, colorama, posterize, threshold) and blends
(mask one source by another, difference, layer, feedback). The graph is compiled to a single GLSL fragment shader and run on
the GPU (headless; Mesa llvmpipe on a CPU runner). Parameters ride the beat.

    C = Comp(1080, 1920)
    C.set_source(0, pil_photo); C.set_source(1, pil_photo2); C.set_source(2, pil_cutout)
    g = recipe(rng, kinds)                         # a graph (nested tuples), see RECIPES / random_graph
    C.compile(g)
    img = C.frame(t, beat, energy, seed)           # PIL RGB, feeds back into itself next frame

Also here: a point-cloud renderer (a photo broken into points pushed into depth by brightness, morphing into another image
or a shape) and the analog pass (VHS / composite / film / CRT), both on the same GPU context.
"""
import math
import numpy as np
from PIL import Image

HEAD = """#version 330
uniform vec2 R; uniform float T, B, E, SEED, MIX;
uniform sampler2D S0, S1, S2, S3, PREV, PC, SIM, RE, CARDS, D0, D1, D3, FLUID, SLIME, ATTR;
uniform vec2 A0, A1, A2, A3;        // source sizes (for cover-fit)
in vec2 vuv; out vec4 o;
float hash(vec2 p){ p = fract(p*vec2(123.34, 456.21)); p += dot(p, p+45.32); return fract(p.x*p.y); }
float vnoise(vec2 p){ vec2 i=floor(p), f=fract(p); f=f*f*(3.-2.*f);
  return mix(mix(hash(i),hash(i+vec2(1,0)),f.x), mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x), f.y); }
float fbm(vec2 p){ float a=.5, s=0.; for(int i=0;i<4;i++){ s+=a*vnoise(p); p*=2.03; a*=.5; } return s; }
mat2 rot(float a){ float c=cos(a), s=sin(a); return mat2(c,-s,s,c); }
float luma(vec4 c){ return dot(c.rgb, vec3(.299,.587,.114)); }
vec3 hsv2rgb(vec3 c){ vec4 K=vec4(1.,2./3.,1./3.,3.); vec3 p=abs(fract(c.xxx+K.xyz)*6.-K.www); return c.z*mix(K.xxx, clamp(p-K.xxx,0.,1.), c.y); }
vec3 rgb2hsv(vec3 c){ vec4 K=vec4(0.,-1./3.,2./3.,-1.); vec4 p=mix(vec4(c.bg,K.wz),vec4(c.gb,K.xy),step(c.b,c.g));
  vec4 q=mix(vec4(p.xyw,c.r),vec4(c.r,p.yzx),step(p.x,c.r)); float d=q.x-min(q.w,q.y); float e=1e-10;
  return vec3(abs(q.z+(q.w-q.y)/(6.*d+e)), d/(q.x+e), q.x); }
vec2 cover(vec2 uv, vec2 sz){ float ra = (R.x/R.y) / (sz.x/sz.y); vec2 c = uv-.5; if(ra<1.) c.x*=ra; else c.y/=ra; return c+.5; }
vec4 tx(sampler2D s, vec2 uv, vec2 sz){ vec2 q = cover(uv, sz); return texture(s, q); }
vec4 txw(sampler2D s, vec2 uv, vec2 sz){ vec2 q = cover(uv, sz); return texture(s, fract(q)); }
"""

VERT = "#version 330\nin vec2 v; out vec2 vuv; void main(){ vuv = v*.5+.5; gl_Position = vec4(v,0.,1.); }"

# ------------------------------------------------------------------ graph → GLSL
# A graph is a nested tuple: (op, args..., children...). emit() returns GLSL lines and the name of the vec4 it produced.
class _G:
    def __init__(self): self.lines = []; self.n = 0
    def var(self, kind="vec4"): self.n += 1; return f"v{self.n}"

def _f(x): return f"{float(x):.4f}"

def emit(g, node, uv):
    op = node[0]; a = node[1:]
    L = g.lines
    def new(expr, kind="vec4"):
        v = g.var(); L.append(f"{kind} {v} = {expr};"); return v
    # ---------- sources
    if op == "tex":                       # ("tex", slot, wrap)
        s = a[0]; return new(f"{'txw' if (len(a) > 1 and a[1]) else 'tx'}(S{s}, {uv}, A{s})")
    if op == "cut":                       # a cut-out, contained (not cover-fit), centred: ("cut", slot, size, beatpump, spin)
        s, sz, bp, sp = a; q = g.var()
        L.append(f"vec2 {q}; {{ vec2 c=({uv}-.5)*vec2(R.x/R.y,1.); c=rot(T*{_f(sp)}+.15*sin(T*.7))*c; float k={_f(sz)}*(1.+{_f(bp)}*B); vec2 asp=A{s}/max(A{s}.x,A{s}.y); c/=k*asp*vec2(R.x/R.y,1.)*vec2(1.,R.x/R.y)/vec2(1.,R.x/R.y); {q}=c*vec2(1.,1.)/vec2(R.x/R.y,1.)*vec2(R.x/R.y,1.)+.5; }}")
        return new(f"(all(greaterThanEqual({q},vec2(0.)))&&all(lessThanEqual({q},vec2(1.)))) ? texture(S{s}, {q}) : vec4(0.)")
    if op == "parallax":                  # a still photo turned into a 3D space: ("parallax", slot, amount)
        sl, amt = a; q = g.var(); dn = {0: "D0", 1: "D1", 3: "D3"}[sl]
        L.append(f"vec2 {q}; {{ vec2 cam = vec2(sin(T*.37+SEED*6.), cos(T*.29+SEED*4.))*{_f(amt)}*(1.+.6*B); vec2 c = cover({uv}, A{sl}); "
                 f"vec2 st = cam/10.; vec2 pq = c; float h = 1.; for(int k=0;k<10;k++){{ float dd = texture({dn}, pq).r; if (h < 1.-dd) break; h -= .1; pq -= st; }} {q} = pq; }}")
        return new(f"texture(S{sl}, {q})")
    if op == "dzoom":                     # dolly-zoom: near things grow while far things shrink, on the beat
        sl, k = a; c = new(f"cover({uv}, A{sl})", "vec2"); dn = {0: "D0", 1: "D1", 3: "D3"}[sl]
        dd = new(f"texture({dn}, {c}).r", "float")
        q = new(f"({c}-.5)/(1.+{_f(k)}*(.5+.5*sin(T*.9))*(1.+B)*({dd}-.5))+.5", "vec2"); return new(f"texture(S{sl}, {q})")
    if op == "depth":                     # the depth map itself as a source (contour lines on it read as a topographic scan)
        sl, = a; dn = {0: "D0", 1: "D1", 3: "D3"}[sl]; dd = new(f"texture({dn}, cover({uv}, A{sl})).r", "float")
        return new(f"vec4(vec3(smoothstep(.0,.08,abs(fract({dd}*14.-T*.4)-.5)-.38)), 1.)")
    if op == "rutt":   return new(f"texture(RE, {uv})")
    if op == "fluid":  return new(f"texture(FLUID, {uv})")
    if op == "slime":  return new(f"texture(SLIME, {uv})")
    if op == "attractor": return new(f"texture(ATTR, {uv})")
    if op == "hyper":                     # Poincaré disk {p,q} tiling of a source, flowing under a Möbius transformation
        P_, Q_, sl = a; v = g.var(); cnt = g.var()
        L.append(f"vec2 {v}; float {cnt} = 0.; {{ vec2 z = ({uv}-.5)*vec2(R.x/R.y,1.)*2.05; float rr = dot(z,z);"
                 f" vec2 m = .55*vec2(cos(T*.21+SEED*6.), sin(T*.17)); vec2 num = z - m; vec2 den = vec2(1.,0.) - vec2(m.x*z.x+m.y*z.y, m.x*z.y-m.y*z.x);"
                 f" z = vec2(num.x*den.x+num.y*den.y, num.y*den.x-num.x*den.y)/dot(den,den); z = rot(T*.12)*z;"
                 f" float pp = 3.14159265/{_f(P_)}, qq = 3.14159265/{_f(Q_)}; float cr = 1./sqrt(cos(qq)*cos(qq)/(sin(pp)*sin(pp))-1.); vec2 C = vec2(sqrt(1.+cr*cr), 0.); vec2 n = vec2(-sin(pp), cos(pp));"
                 f" for(int i=0;i<40;i++){{ bool f = false; if (z.y < 0.) {{ z.y = -z.y; {cnt} += 1.; f = true; }} float dn = dot(z,n); if (dn > 0.) {{ z -= 2.*dn*n; {cnt} += 1.; f = true; }}"
                 f" vec2 w = z - C; float ww = dot(w,w); if (ww < cr*cr) {{ z = C + cr*cr*w/ww; {cnt} += 1.; f = true; }} if (!f) break; }}"
                 f" {v} = rr < 1. ? z*1.6 + .5 : vec2(-9.); }}")
        c = new(f"{v}.x < -1. ? vec4(0.,0.,0.,1.) : tx(S{sl}, {v}, A{sl})")
        return new(f"vec4({c}.rgb * (mod({cnt}, 2.) < .5 ? 1. : .55 + .45*B), 1.)")
    if op == "cards":  return new(f"texture(CARDS, {uv})")
    if op == "prev":   return new(f"texture(PREV, {uv})")
    if op == "pc":     return new(f"texture(PC, {uv})")
    if op == "sim":    return new(f"texture(SIM, {uv})")
    if op == "static":                    # ("static", px, colour)
        px, col = a; q = new(f"floor({uv}*R/{_f(px)}) + floor(T*24.)*vec2(17.3,9.1)", "vec2")
        if col: return new(f"vec4(hash({q}), hash({q}+3.1), hash({q}+7.7), 1.)")
        return new(f"vec4(vec3(hash({q})), 1.)")
    if op == "noise":                     # ("noise", scale, speed)
        sc, sp = a; return new(f"vec4(vec3(fbm({uv}*{_f(sc)} + vec2(T*{_f(sp)}, -T*{_f(sp*.7)}) + SEED*10.)), 1.)")
    if op == "osc":                       # ("osc", freq, speed, colour offset)
        fr, sp, off = a
        return new(f"vec4(.5+.5*sin(({uv}.x*{_f(fr)} + T*{_f(sp)}) + vec3(0., {_f(off)}, {_f(off*2)})), 1.)")
    if op == "voronoi":                   # ("voronoi", scale, speed)
        sc, sp = a; p = new(f"{uv}*{_f(sc)}", "vec2"); d = g.var();
        L.append(f"float {d} = 9.; for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++){{ vec2 c=floor({p})+vec2(i,j); vec2 pt=c+.5+.4*sin(T*{_f(sp)}+6.28*vec2(hash(c),hash(c+1.3))); {d}=min({d}, length({p}-pt)); }}")
        return new(f"vec4(vec3(smoothstep(0.,.9,{d})), 1.)")
    if op == "grid":                      # HUD-style scan grid: ("grid", cells, thickness)
        n, th = a; q = new(f"abs(fract({uv}*vec2({_f(n)}, {_f(n*16/9)}))-.5)", "vec2")
        return new(f"vec4(vec3(smoothstep(.5-{_f(th)}, .5, max({q}.x, {q}.y))), 1.)")
    # ---------- coordinate ops (transform uv, then evaluate the child)
    if op == "rotate":                    # ("rotate", angle, speed, child)
        ang, sp, ch = a; u = new(f"(rot({_f(ang)} + T*{_f(sp)})*({uv}-.5))+.5", "vec2"); return emit(g, ch, u)
    if op == "scale":                     # ("scale", s, beatpump, child)
        s, bp, ch = a; u = new(f"({uv}-.5)/({_f(s)}*(1.+{_f(bp)}*B))+.5", "vec2"); return emit(g, ch, u)
    if op == "scroll":                    # ("scroll", x, y, child)
        x, y, ch = a; u = new(f"{uv}+vec2({_f(x)},{_f(y)})*T", "vec2"); return emit(g, ch, u)
    if op == "kaleid":                    # ("kaleid", n, child)
        n, ch = a; u = g.var()
        L.append(f"vec2 {u}; {{ vec2 c={uv}-.5; float an=atan(c.y,c.x), r=length(c), seg=6.2831/{_f(n)}; an=mod(an,seg); an=abs(an-seg*.5); {u}=vec2(cos(an),sin(an))*r+.5; }}")
        return emit(g, ch, u)
    if op == "repeat":                    # ("repeat", nx, ny, child)
        nx, ny, ch = a; u = new(f"fract({uv}*vec2({_f(nx)},{_f(ny)}))", "vec2"); return emit(g, ch, u)
    if op == "pixelate":                  # ("pixelate", n, child)
        n, ch = a; u = new(f"(floor({uv}*vec2({_f(n)},{_f(n*16/9)}))+.5)/vec2({_f(n)},{_f(n*16/9)})", "vec2"); return emit(g, ch, u)
    if op == "swirl":                     # ("swirl", k, child)
        k, ch = a; u = g.var(); L.append(f"vec2 {u}; {{ vec2 c={uv}-.5; float r=length(c); c*=rot({_f(k)}*(1.+.6*B)*(.6-r)*smoothstep(.6,0.,r)); {u}=c+.5; }}")
        return emit(g, ch, u)
    if op == "fisheye":                   # ("fisheye", k, child)
        k, ch = a; u = g.var(); L.append(f"vec2 {u}; {{ vec2 c={uv}-.5; float r=length(c); c*=1.+{_f(k)}*(1.+.5*B)*r*r*4.; {u}=c+.5; }}")
        return emit(g, ch, u)
    if op == "stretch":                   # non-linear stretch: ("stretch", kind, k, child)  kind 0 pow-y, 1 sine bands, 2 log-polar tunnel, 3 slit columns
        kind, k, ch = a; u = g.var(); kk = _f(k)
        if kind == 0: L.append(f"vec2 {u} = vec2({uv}.x, pow(clamp({uv}.y,0.,1.), 1.+{kk}*(.5+.5*sin(T*.7))*(1.+B)));")
        elif kind == 1: L.append(f"vec2 {u} = {uv} + vec2(sin({uv}.y*{_f(6+k*14)}+T*2.)*{_f(.02+.05*k)}*(1.+2.*B), 0.);")
        elif kind == 2: L.append(f"vec2 {u}; {{ vec2 c={uv}-.5; c.x*=R.x/R.y; float r=length(c), an=atan(c.y,c.x); {u}=vec2(an/6.2831+.5, fract(log(r+1e-3)*{_f(.4+k)} - T*.4)); }}")
        else: L.append(f"vec2 {u} = vec2({uv}.x, {uv}.y + (hash(vec2(floor({uv}.x*{_f(20+k*60)}), floor(T*8.)))-.5)*{_f(.05+.2*k)}*(.3+B));")
        return emit(g, ch, u)
    if op in ("modulate", "modscale", "modrot"):   # one source bends another: (op, amount, child, by)
        amt, ch, by = a; m = emit(g, by, uv)
        if op == "modulate": u = new(f"{uv} + ({m}.rg-.5)*{_f(amt)}*(1.+1.5*B)", "vec2")
        elif op == "modscale": u = new(f"({uv}-.5)*(1.+({m}.r-.5)*{_f(amt)}*(1.+B))+.5", "vec2")
        else: u = new(f"(rot(({m}.r-.5)*{_f(amt)}*(1.+B))*({uv}-.5))+.5", "vec2")
        return emit(g, ch, u)
    # ---------- colour ops
    if op == "lumakey":                   # ("lumakey", thr, soft, invert, child) → alpha from brightness
        thr, soft, inv, ch = a; c = emit(g, ch, uv)
        k = f"smoothstep({_f(thr)}, {_f(thr+soft)}, luma({c}))"; k = f"(1.-{k})" if inv else k
        return new(f"vec4({c}.rgb, {c}.a*{k})")
    if op == "colorama":                  # ("colorama", amt, child) hue shift by brightness
        amt, ch = a; c = emit(g, ch, uv); h = new(f"rgb2hsv({c}.rgb)", "vec3")
        return new(f"vec4(hsv2rgb(vec3(fract({h}.x+{_f(amt)}*{h}.z+T*.05), clamp({h}.y*1.4+.2,0.,1.), {h}.z)), {c}.a)")
    if op == "posterize":
        n, ch = a; c = emit(g, ch, uv); return new(f"vec4(floor({c}.rgb*{_f(n)})/{_f(n)}, {c}.a)")
    if op == "thresh":
        thr, ch = a; c = emit(g, ch, uv); return new(f"vec4(vec3(step({_f(thr)}+.1*sin(T), luma({c}))), {c}.a)")
    if op == "invert":
        ch, = a; c = emit(g, ch, uv); return new(f"vec4(1.-{c}.rgb, {c}.a)")
    if op == "tint":                      # ("tint", (r,g,b) shadows, (r,g,b) highs, child): duotone onto two colours
        c0, c1, ch = a; c = emit(g, ch, uv)
        return new(f"vec4(mix(vec3({_f(c0[0])},{_f(c0[1])},{_f(c0[2])}), vec3({_f(c1[0])},{_f(c1[1])},{_f(c1[2])}), smoothstep(.1,.9,luma({c}))), {c}.a)")
    if op == "contrast":
        k, ch = a; c = emit(g, ch, uv); return new(f"vec4(clamp(({c}.rgb-.5)*{_f(k)}+.5,0.,1.), {c}.a)")
    if op == "edges":                     # ("edges", slot): line drawing of a source (scan look)
        s, = a; e = g.var()
        L.append(f"float {e}; {{ vec2 d=1.5/R; vec2 q={uv}; float l=luma(tx(S{s},q-vec2(d.x,0.),A{s}))-luma(tx(S{s},q+vec2(d.x,0.),A{s})); float m=luma(tx(S{s},q-vec2(0.,d.y),A{s}))-luma(tx(S{s},q+vec2(0.,d.y),A{s})); {e}=clamp(length(vec2(l,m))*5.,0.,1.); }}")
        return new(f"vec4(vec3({e}), 1.)")
    # ---------- blends
    if op == "mask":                      # ("mask", a, b, m): a where m is bright, b elsewhere
        ca = emit(g, a[0], uv); cb = emit(g, a[1], uv); cm = emit(g, a[2], uv)
        return new(f"mix({cb}, {ca}, smoothstep(.35,.65,luma({cm}))*{cm}.a)")
    if op == "layer":                     # ("layer", top, bottom): top over bottom by top's alpha
        ct = emit(g, a[0], uv); cb = emit(g, a[1], uv); return new(f"vec4(mix({cb}.rgb, {ct}.rgb, {ct}.a), 1.)")
    if op == "diff":
        ca = emit(g, a[0], uv); cb = emit(g, a[1], uv); return new(f"vec4(abs({ca}.rgb-{cb}.rgb), 1.)")
    if op == "add":
        k = a[0]; ca = emit(g, a[1], uv); cb = emit(g, a[2], uv); return new(f"vec4(clamp({ca}.rgb+{cb}.rgb*{_f(k)},0.,1.), 1.)")
    if op == "mult":
        ca = emit(g, a[0], uv); cb = emit(g, a[1], uv); return new(f"vec4({ca}.rgb*{cb}.rgb*1.6, 1.)")
    if op == "feedback":                  # ("feedback", amount, zoom, turn, hue, child): child over a transformed copy of the last frame
        amt, zm, tn, hu, ch = a; c = emit(g, ch, uv)
        pu = new(f"(rot({_f(tn)})*({uv}-.5))/{_f(zm)}+.5", "vec2"); p = new(f"texture(PREV, {pu})")
        ph = new(f"rgb2hsv({p}.rgb)", "vec3"); pp = new(f"vec4(hsv2rgb(vec3(fract({ph}.x+{_f(hu)}), {ph}.y, {ph}.z*.985)), 1.)")
        return new(f"vec4(mix({pp}.rgb, {c}.rgb, clamp({c}.a*(1.-{_f(amt)})+{_f(1-amt)}*.5 + .25*B, 0., 1.)), 1.)")
    raise ValueError("unknown op " + op)

def to_glsl(graph):
    g = _G(); out = emit(g, graph, "vuv")
    body = "\n  ".join(g.lines)
    # MIX lets the caller melt the previous section into this one (liquid: the old frame flows by the new frame's colours)
    return HEAD + f"""
void main(){{
  {body}
  vec4 c = {out};
  if (MIX < 1.) {{ vec2 d = (c.rg-.5)*.18*(1.-MIX); vec4 p = texture(PREV, vuv + d); c = mix(p, c, smoothstep(0.,1.,MIX)); }}
  o = vec4(c.rgb, 1.);
}}"""

# ------------------------------------------------------------------ recipes: generalised from what reads well
# slots: 0 = photo A, 1 = photo B, 2 = cut-out / meme (has alpha), 3 = moving clip (or a third photo)
def recipes(r):
    P = lambda s: ("tex", s, 0)
    st = lambda: ("static", r.choice([1.5, 2, 3, 5]), r.random() < .35)
    nz = lambda: ("noise", r.uniform(2, 7), r.uniform(.1, .5))
    w = lambda: r.choice([nz(), ("osc", r.uniform(4, 30), r.uniform(.2, 2), r.uniform(.2, 1.5)), ("voronoi", r.uniform(4, 12), r.uniform(.5, 2)), P(1), P(3)])
    fb = lambda ch: ("feedback", r.uniform(.35, .65), r.uniform(1.005, 1.04), r.uniform(-.03, .03), r.uniform(-.02, .02), ch)
    stretch = lambda ch: ("stretch", r.randrange(4), r.uniform(.3, 1.2), ch)
    R = {
        # a photo peeking through static, the window shaped by another photo's light, the cut-out on top
        "peek": lambda: ("layer", ("cut", 2, r.uniform(.55, .85), .08, r.uniform(-.1, .1)), ("mask", ("modulate", .08, P(0), nz()), (st() if r.random() < .3 else ("contrast", .45, ("scale", 1.3, .04, P(1)))), ("lumakey", r.uniform(.35, .6), .15, r.random() < .5, P(1)))),
        # one photo bent by another, stretched non-linearly, fed back into itself
        "bend": lambda: fb(stretch(("modulate", r.uniform(.15, .5), P(0), P(1)))),
        # the clip kaleidoscoped, warped by a photo, keyed over the other photo's edges
        "kaleid": lambda: ("layer", ("lumakey", .3, .2, 0, ("kaleid", r.choice([3, 4, 5, 6, 8]), ("modrot", r.uniform(1, 4), P(3), P(0)))), ("tint", (0, 0, .05), (.2, 1, .8), ("edges", 1))),
        # the point cloud of photo A over a dark, slowly rotating photo B
        "cloud": lambda: ("add", 1.0, ("pc",), ("contrast", .6, ("rotate", 0, r.uniform(-.05, .05), ("scale", 1.2, .05, P(1))))),
        # two photos scrolling against each other, difference + colour by brightness
        "diff": lambda: ("colorama", r.uniform(.3, 1.2), ("diff", ("scroll", r.uniform(-.05, .05), r.uniform(-.08, .08), P(0)), ("scroll", r.uniform(-.05, .05), r.uniform(-.08, .08), P(1)))),
        # tunnel made of a photo, its shadows full of static
        "tunnel": lambda: ("mask", ("stretch", 2, r.uniform(.3, 1.), P(r.choice([0, 1, 3]))), st(), ("lumakey", .25, .15, 0, ("stretch", 2, .6, P(0)))),
        # the photo as a repeated tile wallpaper, modulated by noise, with the cut-out large in front
        "tiles": lambda: ("layer", ("cut", 2, r.uniform(.6, .9), .1, 0.), ("modulate", r.uniform(.05, .2), ("repeat", r.choice([2, 3, 4]), r.choice([3, 4, 6]), P(r.choice([0, 1]))), w())),
        # thresholded photo as a stencil: clip inside, static outside, fed back
        "stencil": lambda: fb(("mask", P(3), st(), ("thresh", r.uniform(.35, .6), P(0)))),
        # swirl/fisheye of photo A, modulated by photo B, posterized
        "warp": lambda: ("posterize", r.choice([3, 4, 6, 8]), (r.choice(["swirl", "fisheye"]), r.uniform(1.5, 5.), ("modscale", r.uniform(.5, 2.), P(0), P(1)))),
        # HUD scan: edges of a photo on a grid, the photo dim behind
        "scan": lambda: ("add", 1.0, ("tint", (0, 0, 0), (.3, 1, .9), ("edges", 0)), ("add", .35, ("contrast", .5, P(0)), ("grid", r.choice([6, 9, 12]), .04))),
        # slit-scan: columns of the clip offset in time-ish, through the other photo's light
        # the fight: black/white chemistry vs colour spirals, the photo dissolving into it
        "fight": lambda: ("mask", ("contrast", 1.5, P(r.choice([0, 1]))), ("sim",), ("sim",)),          # the photo lives inside the chemistry
        # real photo, barely touched: a slow push, the cut-out in front. Honest moments between the madness
        "raw": lambda: ("layer", ("cut", 2, r.uniform(.5, .8), .1, r.uniform(-.06, .06)), ("scale", r.uniform(1.0, 1.15), .06, ("scroll", r.uniform(-.01, .01), r.uniform(-.015, .015), P(r.choice([0, 1, 3]))))),
        # hieroglyphs: a cultural reference reduced to a glyph, repeated into a wall of signs, the wall bent by noise, a photo behind
        "glyphs": lambda: ("mask", ("tint", (0, 0, 0), (1, 1, 1), ("modulate", r.uniform(.02, .08), ("repeat", r.choice([3, 4, 5]), r.choice([5, 6, 8]), ("thresh", r.uniform(.4, .6), ("tex", 2, 1))), nz())), P(0), ("lumakey", .5, .1, 0, ("modulate", .03, ("repeat", 4, 6, ("thresh", .5, ("tex", 2, 1))), nz()))),
        # two references colliding: one inside the other's silhouette, swapping on the beat
        "collide": lambda: ("mask", ("tex", 2, 0), P(1), ("thresh", r.uniform(.4, .6), P(r.choice([0, 3])))),
        # the reference smeared into a sign: kaleidoscoped and stretched until it is almost unreadable
        "smear": lambda: ("colorama", r.uniform(.1, .4), ("stretch", r.choice([1, 3]), r.uniform(.15, .45), ("kaleid", 2, ("tex", 2, 0)))),
        # TouchDesigner-style: the photo as 3D space, scan lines, instanced cards, light-trail edges
        "space": lambda: ("parallax", r.choice([0, 1]), r.uniform(.05, .11)),
        "vertigo": lambda: ("dzoom", r.choice([0, 1]), r.uniform(.25, .6)),
        "rutt": lambda: ("add", r.uniform(.15, .35), ("rutt",), ("contrast", .35, ("parallax", 0, .05))),
        "rutt2": lambda: ("rutt",),
        "cloudsph": lambda: ("add", 1.0, ("pc",), ("contrast", .15, ("tex", 0, 0))),
        "dark": lambda: ("contrast", .3, ("scale", 1.15, .03, ("tex", 0, 0))),
        "topo": lambda: ("mask", ("parallax", 0, .06), ("tint", (0, 0, 0), (1, 1, 1), ("depth", 0)), ("lumakey", .5, .3, 0, ("depth", 0))),
        "cards": lambda: ("add", 1.0, ("cards",), ("contrast", .3, ("scale", 1.25, .03, P(r.choice([0, 1]))))),
        "trails": lambda: ("feedback", r.uniform(.6, .8), r.uniform(1.01, 1.03), r.uniform(-.01, .01), r.uniform(-.01, .01), ("add", 1.0, ("tint", (0, 0, 0), (.4, 1, .9), ("edges", 0)), ("contrast", .35, P(0)))),
        # simulations from the literature, all fed by the photos
        "fluid": lambda: ("contrast", 1.7, ("fluid",)),
        "fluidmix": lambda: ("layer", ("lumakey", .55, .2, 0, ("fluid",)), ("contrast", .4, P(1))),
        "slime": lambda: ("slime",),
        "attractor": lambda: ("add", 1.0, ("attractor",), ("contrast", .22, P(r.choice([0, 1])))),
        "attractor2": lambda: ("attractor",),
        "hyper": lambda: ("hyper", r.choice([7, 5, 8, 6]), r.choice([3, 4]), r.choice([0, 2])),
        "hyperref": lambda: ("colorama", r.uniform(.1, .4), ("hyper", r.choice([7, 8]), 3, 2)),
        "rawpair": lambda: ("mask", P(0), P(1), ("osc", r.uniform(2, 6), r.uniform(.5, 2), 0.)),
        # the photo eaten by the labyrinth: shown only where the chemistry has grown
        "erode": lambda: ("mask", ("modulate", r.uniform(.02, .08), P(r.choice([0, 3])), ("sim",)), ("invert", ("sim",)), ("sim",)),
        # the photo pushed around by the systems like a fluid
        "slit": lambda: ("mask", ("stretch", 3, r.uniform(.3, 1.), P(3)), ("stretch", 3, r.uniform(.3, 1.), P(0)), ("lumakey", .5, .2, r.random() < .5, P(1))),
    }
    return R

def random_graph(r, depth=3):
    """a fully random chain from the same parts, for the sections that shouldn't look like any recipe"""
    srcs = [("tex", 0, 0), ("tex", 1, 0), ("tex", 3, 0), ("static", r.choice([1.5, 2, 4]), r.random() < .3), ("noise", r.uniform(2, 8), r.uniform(.1, .6))]
    def node(d):
        if d <= 0: return r.choice(srcs[:3] * 2 + srcs[3:])
        k = r.choice(["modulate", "modscale", "modrot", "kaleid", "stretch", "mask", "diff", "colorama", "repeat", "swirl", "lumalayer"])
        if k in ("modulate", "modscale", "modrot"): return (k, r.uniform(.1, .6) if k == "modulate" else r.uniform(.5, 3), node(d - 1), node(d - 2))
        if k == "kaleid": return ("kaleid", r.choice([3, 4, 6, 8]), node(d - 1))
        if k == "stretch": return ("stretch", r.randrange(4), r.uniform(.3, 1.2), node(d - 1))
        if k == "mask": return ("mask", node(d - 1), node(d - 2), node(d - 2))
        if k == "diff": return ("diff", node(d - 1), node(d - 2))
        if k == "colorama": return ("colorama", r.uniform(.2, 1.), node(d - 1))
        if k == "repeat": return ("repeat", r.choice([2, 3]), r.choice([2, 3, 4]), node(d - 1))
        if k == "swirl": return ("swirl", r.uniform(1, 5), node(d - 1))
        return ("layer", ("lumakey", r.uniform(.3, .6), .2, r.random() < .5, node(d - 1)), node(d - 2))
    g = ("feedback", r.uniform(.5, .85), r.uniform(1.0, 1.03), r.uniform(-.02, .02), r.uniform(-.015, .015), node(depth))
    # a real photo always shows through: keyed on top or as the window
    return ("layer", ("lumakey", r.uniform(.35, .6), .15, r.random() < .5, ("tex", r.choice([0, 1]), 0)), g) if r.random() < .7 else ("mask", ("tex", 0, 0), g, ("tex", 1, 0))

# ------------------------------------------------------------------ point cloud
PC_VS = """#version 330
uniform sampler2D SA, SB, DA; uniform vec2 AA, AB, R; uniform float T, B, M, SEED, HASD; uniform int SHAPE;
in vec2 g; out vec3 col;
vec2 cover(vec2 uv, vec2 sz){ float ra=(R.x/R.y)/(sz.x/sz.y); vec2 c=uv-.5; if(ra<1.) c.x*=ra; else c.y/=ra; return c+.5; }
mat3 ry(float a){ float c=cos(a), s=sin(a); return mat3(c,0.,-s, 0.,1.,0., s,0.,c); }
mat3 rx(float a){ float c=cos(a), s=sin(a); return mat3(1.,0.,0., 0.,c,s, 0.,-s,c); }
void main(){
  vec3 ca = texture(SA, cover(g, AA)).rgb, cb = texture(SB, cover(g, AB)).rgb;
  float la = dot(ca, vec3(.3,.59,.11)), lb = dot(cb, vec3(.3,.59,.11));
  float za = mix(la, texture(DA, cover(g, AA)).r, HASD);                     // real depth when we have it
  vec3 pa = vec3((g.x-.5)*R.x/R.y*2., (g.y-.5)*2., -(za*1.3-.65));
  vec3 pb;
  if (SHAPE==0) pb = vec3((g.x-.5)*R.x/R.y*2., (g.y-.5)*2., lb*1.2-.6);                         // the other image
  else if (SHAPE==1) { float th=g.x*6.2831, ph=g.y*3.1416; pb = vec3(sin(ph)*cos(th), cos(ph), sin(ph)*sin(th))*(.9+.15*lb); }   // sphere
  else { float th=g.x*6.2831, ph=g.y*6.2831; pb = vec3((1.+.38*cos(ph))*cos(th), .38*sin(ph), (1.+.38*cos(ph))*sin(th))*.8; }    // torus
  float m = smoothstep(0., 1., clamp(M*1.3 - fract(sin(dot(g, vec2(12.9,78.2)))*43758.5)*.3, 0., 1.));   // points leave at slightly different times
  vec3 p = mix(pa, pb, m);
  p += .04*B*vec3(sin(g.y*80.+T*9.), cos(g.x*70.+T*7.), sin((g.x+g.y)*50.));                    // the kick shakes the cloud
  p = rx(.35*sin(T*.3)) * ry(T*.35 + SEED*6.) * p;
  p.z += 2.6; vec2 s = p.xy / p.z * 1.7; s.x *= R.y/R.x;
  gl_Position = vec4(s, 0., 1.); gl_PointSize = 2.2 + 2.5*B;
  col = mix(ca, cb, m) * (1.15 + .8*B);
}"""
PC_FS = """#version 330
in vec3 col; out vec4 o; void main(){ vec2 c = gl_PointCoord-.5; if (dot(c,c) > .25) discard; o = vec4(col*.28, 1.); }"""

# ------------------------------------------------------------------ analog pass
ANALOG_FS = """#version 330
uniform sampler2D IN; uniform vec2 R; uniform float T, B, K; uniform int MODE;   // 0 none 1 vhs 2 composite 3 film 4 crt
in vec2 vuv; out vec4 o;
float hash(vec2 p){ p=fract(p*vec2(123.34,456.21)); p+=dot(p,p+45.32); return fract(p.x*p.y); }
vec3 rgb2yiq(vec3 c){ return mat3(.299,.596,.211, .587,-.274,-.523, .114,-.322,.312)*c; }
vec3 yiq2rgb(vec3 c){ return mat3(1.,1.,1., .956,-.272,-1.106, .621,-.647,1.703)*c; }
void main(){
  vec2 uv = vuv; float t = T;
  if (MODE == 0) { o = texture(IN, uv); return; }
  if (MODE == 1 || MODE == 2) {
    float line = floor(uv.y*R.y/2.);
    uv.x += (hash(vec2(line, floor(t*30.)))-.5)*.004*K + sin(uv.y*9.+t*2.)*.0015*K;                 // jitter + slow wobble
    if (MODE == 1 && uv.y < .045) uv.x += (hash(vec2(floor(uv.y*200.), floor(t*30.)))-.5)*.06*K;   // head-switching at the bottom
    vec3 y = rgb2yiq(texture(IN, uv).rgb);
    vec3 ch = vec3(0.); float w = 0.;                                                                 // chroma smeared sideways
    for (int i=-6;i<=6;i++){ float k = exp(-float(i*i)/14.); ch += rgb2yiq(texture(IN, uv+vec2(float(i)*.0028*K + .006*K, 0.)).rgb)*k; w += k; }
    ch /= w;
    float sharp = y.x*1.6 - .6*rgb2yiq(texture(IN, uv-vec2(.002,0.)).rgb).x;                        // luma ringing
    vec3 c = yiq2rgb(vec3(mix(y.x, sharp, .5), ch.y*1.15, ch.z*1.15));
    float n = hash(uv*R + floor(t*60.)); c += (n-.5)*.09*K;
    float band = smoothstep(.0, .015, abs(fract(uv.y - t*.07) - .5)); c = mix(c + (n-.5)*.5*K, c, band);   // rolling tracking band
    if (hash(vec2(floor(uv.y*240.), floor(t*12.))) > 1. - .004*K) c = vec3(.9);                     // dropouts
    c *= .93 + .07*sin(uv.y*R.y*1.57);
    o = vec4(c, 1.); return;
  }
  if (MODE == 3) {                                                                                     // film: weave, grain, dust, scratches, leak
    uv += vec2(hash(vec2(floor(t*24.),1.))-.5, hash(vec2(floor(t*24.),2.))-.5)*.004*K;
    vec3 c = texture(IN, uv).rgb; c = pow(c, vec3(1.08)); c = mix(c, c*vec3(1.06,1.,.88), .5);
    float gr = hash(uv*R*.5 + floor(t*24.)*13.1); c += (gr-.5)*.12*K;
    vec2 dp = uv*vec2(40.,70.) + floor(t*24.)*7.; if (hash(floor(dp)) > .997) c *= .2;
    float sx = hash(vec2(floor(t*6.), 3.)); if (abs(uv.x - sx) < .0012 && hash(vec2(floor(t*24.),5.)) > .5) c += .25;
    vec2 lp = vec2(.15+.7*hash(vec2(floor(t*.7),9.)), .2+.6*hash(vec2(floor(t*.7),4.)));
    c += vec3(1.,.45,.15) * .35*K * smoothstep(.55, 0., length((uv-lp)*vec2(1.,.6))) * (.6+.4*sin(t*1.3));
    c *= 1. - .35*length(uv-.5);
    o = vec4(c, 1.); return;
  }
  vec2 c2 = uv-.5; c2 *= 1. + .12*K*dot(c2,c2); uv = c2+.5;                                            // CRT: curvature, mask, glow
  vec3 c = vec3(texture(IN, uv+vec2(.0015,0.)).r, texture(IN, uv).g, texture(IN, uv-vec2(.0015,0.)).b);
  float mx = mod(gl_FragCoord.x, 3.); c *= mix(vec3(1.), vec3(mx<1.?1.2:.8, mx<2.&&mx>=1.?1.2:.8, mx>=2.?1.2:.8), .5*K);
  c *= .8 + .2*sin(uv.y*R.y*3.1416*.5); if (any(lessThan(uv, vec2(0.))) || any(greaterThan(uv, vec2(1.)))) c = vec3(0.);
  o = vec4(c*1.15, 1.);
}"""


# ------------------------------------------------------------------ Rutt-Etra: the photo as rows of lines pushed out by its depth
RE_VS = """#version 330
uniform sampler2D SA, DA; uniform vec2 AA, R; uniform float T, B, AMP, SEED; in vec2 g; out vec3 col;
vec2 cover(vec2 uv, vec2 sz){ float ra=(R.x/R.y)/(sz.x/sz.y); vec2 c=uv-.5; if(ra<1.) c.x*=ra; else c.y/=ra; return c+.5; }
void main(){
  vec2 q = cover(g, AA); vec3 c = texture(SA, q).rgb; float d = texture(DA, q).r, l = dot(c, vec3(.3,.59,.11));
  vec3 p = vec3((g.x-.5)*2.*R.x/R.y, (g.y-.5)*2., -(d*AMP + l*.15)*(1.+.7*B));
  float ay = .55*sin(T*.31+SEED*6.), ax = .35*sin(T*.23+SEED*3.)-.15;
  p.xz = mat2(cos(ay),-sin(ay),sin(ay),cos(ay))*p.xz; p.yz = mat2(cos(ax),-sin(ax),sin(ax),cos(ax))*p.yz;
  p.z += 2.7; vec2 s = p.xy/p.z*1.9; s.x *= R.y/R.x;
  gl_Position = vec4(s, 0., 1.); col = mix(vec3(l), c, .55) * (1.5 + B);
}"""
RE_FS = """#version 330
in vec3 col; out vec4 o; void main(){ o = vec4(col*1.25, 1.); }"""

# ------------------------------------------------------------------ instancing: photo cards in 3D formations, reshaping on the drop
CARD_VS = """#version 330
uniform vec2 R; uniform float T, B, M, SEED, N; uniform int FA, FB; in vec2 v; out vec2 uv; flat out int sl; out float edge;
vec3 form(int f, float i){
  float n = N, t = i/n;
  if (f==0) { float c = floor(sqrt(n)); return vec3((mod(i,c)/(c-1.)-.5)*3.2, (floor(i/c)/(c-1.)-.5)*5.2, 0.); }
  if (f==1) { float y = 1.-2.*(i+.5)/n, r = sqrt(1.-y*y), ph = i*2.39996; return vec3(cos(ph)*r, y*1.6, sin(ph)*r)*1.7; }
  if (f==2) { float a = i*.55; return vec3(cos(a)*1.3, (t-.5)*5., sin(a)*1.3); }
  float ring = floor(i/8.), a = mod(i,8.)/8.*6.2831; return vec3(cos(a)*1.4, sin(a)*1.4, ring*1.6 - 4. + mod(T*2.,1.6));
}
void main(){
  float i = float(gl_InstanceID); sl = gl_InstanceID % 4;
  vec3 c = mix(form(FA, i), form(FB, i), smoothstep(0.,1.,M));
  float a = T*.4 + i*.3 + B*.8; vec2 q = v*.36*(1.+.25*B*step(.5,fract(i*.37)));
  vec3 p = c + vec3(q.x*cos(a*.3), q.y, q.x*sin(a*.3));
  float ay = T*.25+SEED*6.; p.xz = mat2(cos(ay),-sin(ay),sin(ay),cos(ay))*p.xz;
  p.z += 5.2; vec2 s = p.xy/p.z*2.2; s.x *= R.y/R.x; gl_Position = vec4(s, p.z*.01, 1.);
  uv = v*.5+.5; edge = 0.;
}"""
CARD_FS = """#version 330
uniform sampler2D S0, S1, S2, S3; uniform float B; in vec2 uv; flat in int sl; out vec4 o;
void main(){
  vec4 c = sl==0 ? texture(S0,uv) : sl==1 ? texture(S1,uv) : sl==2 ? texture(S2,uv) : texture(S3,uv);
  float e = 1.-smoothstep(.0,.03,min(min(uv.x,1.-uv.x),min(uv.y,1.-uv.y)));
  o = vec4(mix(c.rgb, vec3(1.), e*(.5+.5*B)), 1.);
}"""

# ------------------------------------------------------------------ time displacement (TouchDesigner's Time Machine): each pixel from a different moment
TIME_FS = """#version 330
uniform sampler2DArray HIST; uniform sampler2D CUR, MAP; uniform vec2 AM; uniform float T, HEAD, NL; uniform int MODE; in vec2 vuv; out vec4 o;
vec2 cover(vec2 uv, vec2 sz){ float ra=(1080./1920.)/(sz.x/sz.y); vec2 c=uv-.5; if(ra<1.) c.x*=ra; else c.y/=ra; return c+.5; }
void main(){
  float m;
  if (MODE==1) m = dot(texture(MAP, cover(vuv, AM)).rgb, vec3(.3,.59,.11));       // bright parts lag
  else if (MODE==2) m = vuv.y;                                                      // slit-scan, top to bottom
  else if (MODE==3) m = abs(vuv.x-.5)*2.;                                           // from the centre outwards
  else if (MODE==4) m = 1.-texture(MAP, cover(vuv, AM)).r;                          // far things lag behind near ones
  else m = .5+.5*sin(length(vuv-.5)*18.-T*3.);                                      // ripples of time
  float d = floor(m*(NL-1.));
  if (d < 1.) { o = texture(CUR, vuv); return; }
  float layer = mod(HEAD - d + NL, NL);
  o = vec4(texture(HIST, vec3(vuv, layer)).rgb, 1.);
}"""

class Comp:
    def __init__(self, w=1080, h=1920):
        import moderngl
        try: self.ctx = moderngl.create_standalone_context(backend="egl")
        except Exception: self.ctx = moderngl.create_standalone_context()
        self.w, self.h = w, h; ctx = self.ctx
        self.quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes())
        self.fb = [ctx.framebuffer(color_attachments=[ctx.texture((w, h), 4)]) for _ in range(2)]; self.cur = 0
        for f in self.fb: f.color_attachments[0].repeat_x = f.color_attachments[0].repeat_y = False
        self.pcfb = ctx.framebuffer(color_attachments=[ctx.texture((w, h), 4)])
        self.anfb = ctx.framebuffer(color_attachments=[ctx.texture((w, h), 4)]); self.intex = ctx.texture((w, h), 3)
        self.src = [None] * 4; self.size = [(1, 1)] * 4; self.prog = None; self.vao = None; self._cache = {}
        blank = ctx.texture((2, 2), 4, bytes(16))
        for i in range(4): self.src[i] = blank
        self.pcprog = ctx.program(vertex_shader=PC_VS, fragment_shader=PC_FS)
        n = 360                                                             # grid side → ~230k points at 360 x 640
        gx, gy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, int(n * 16 / 9)))
        self.pcbuf = ctx.buffer(np.stack([gx.ravel(), gy.ravel()], 1).astype("f4").tobytes()); self.pcvao = ctx.vertex_array(self.pcprog, [(self.pcbuf, "2f", "g")])
        self.anprog = ctx.program(vertex_shader=VERT, fragment_shader=ANALOG_FS); self.anvao = ctx.simple_vertex_array(self.anprog, self.quad, "v")
        self.sim = Sim(ctx, self.quad); self.sim_seed = 0.0; self.sim_bal = 0.0
        # depth per slot (computed lazily, only when a graph needs it)
        self.pil = [None] * 4; self.dep = [ctx.texture((2, 2), 1, bytes(4)) for _ in range(4)]; self.dep_for = [None] * 4
        # Rutt-Etra lines
        rows, cols = 210, 300; gx = np.linspace(0, 1, cols, dtype="f4"); seg = []
        for y in np.linspace(0, 1, rows, dtype="f4"): seg.append(np.stack([np.repeat(gx, 2)[1:-1], np.full((cols - 1) * 2, y, "f4")], 1))
        self.rebuf = ctx.buffer(np.concatenate(seg).astype("f4").tobytes()); self.reprog = ctx.program(vertex_shader=RE_VS, fragment_shader=RE_FS)
        self.revao = ctx.vertex_array(self.reprog, [(self.rebuf, "2f", "g")]); self.refb = ctx.framebuffer(color_attachments=[ctx.texture((w, h), 4)])
        # instanced cards
        self.cprog = ctx.program(vertex_shader=CARD_VS, fragment_shader=CARD_FS); self.cbuf = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes())
        self.cvao = ctx.vertex_array(self.cprog, [(self.cbuf, "2f", "v")]); self.cfb = ctx.framebuffer(color_attachments=[ctx.texture((w, h), 4)], depth_attachment=ctx.depth_renderbuffer((w, h)))
        self.cards_form = (0, 1); self.cards_n = 64
        self.fluid = self.slime = self.attr = None; self.sim_cols = ((0, 255, 170), (255, 40, 40)); self.attr_kind = (0, 0)
        # time displacement history (half resolution, 36 frames ≈ 1.5 s)
        self.tw, self.th, self.tn = w // 2, h // 2, 36; self.hist = ctx.texture_array((self.tw, self.th, self.tn), 3); self.head = 0; self.tmode = 0
        self.tprog = ctx.program(vertex_shader=VERT, fragment_shader=TIME_FS); self.tvao = ctx.simple_vertex_array(self.tprog, self.quad, "v")
        self.tfb = ctx.framebuffer(color_attachments=[ctx.texture((w, h), 4)]); self.curtex = ctx.texture((w, h), 3)

    def set_source(self, i, im):
        """upload a PIL image into slot i (RGBA kept, so cut-outs carry their alpha)"""
        if im is None: return
        im = im.convert("RGBA"); im.thumbnail((1400, 1400))
        old = self.src[i]
        t = self.ctx.texture(im.size, 4, im.transpose(Image.FLIP_TOP_BOTTOM).tobytes()); t.build_mipmaps(); t.repeat_x = t.repeat_y = True
        self.src[i] = t; self.size[i] = im.size; self.pil[i] = im
        if old is not None and old.size != (2, 2): old.release()

    def compile(self, graph):
        code = to_glsl(graph)
        if code not in self._cache:
            p = self.ctx.program(vertex_shader=VERT, fragment_shader=code); self._cache[code] = (p, self.ctx.simple_vertex_array(p, self.quad, "v"))
        self.prog, self.vao = self._cache[code]; self._body = code.split("void main")[1]; self.uses_pc = "PC" in self._body
        if "SIM" in code.split("void main")[1]: self.sim_seed = 1.0             # a new section throws its photo into the chemistry
        return code

    def _u(self, p, k, v):
        if k in p: p[k].value = v

    def cloud(self, t, beat, morph, seed, shape=0, a=0, b=1):
        """render the point cloud of source a morphing into source b (shape 0) or a sphere (1) / torus (2)"""
        p = self.pcprog; self.pcfb.use(); self.ctx.clear(0, 0, 0, 1)
        self.ctx.enable(self.ctx.BLEND | self.ctx.PROGRAM_POINT_SIZE); self.ctx.blend_func = (self.ctx.ONE, self.ctx.ONE)
        self.src[a].use(5); self.src[b].use(6); p["SA"].value = 5; p["SB"].value = 6
        if "DA" in p: self.dep[a].use(7); p["DA"].value = 7
        self._u(p, "HASD", 1.0 if self.dep_for[a] is not None else 0.0)
        for k, v in (("AA", tuple(map(float, self.size[a]))), ("AB", tuple(map(float, self.size[b]))), ("R", (float(self.w), float(self.h))), ("T", float(t)), ("B", float(beat)), ("M", float(morph)), ("SEED", float(seed))): self._u(p, k, v)
        if "SHAPE" in p: p["SHAPE"].value = int(shape)
        self.pcvao.render(mode=self.ctx.POINTS); self.ctx.disable(self.ctx.BLEND)

    def frame(self, t, beat=0.0, energy=.5, seed=0.0, mix=1.0, pc=None):
        """one frame of the compiled graph; returns PIL RGB. pc = (morph, shape, a, b) renders the point cloud first."""
        body = self._body
        for i, nm in ((0, "D0"), (1, "D1"), (3, "D3")):
            if (nm in body or ("RE" in body and i == 0)) and self.pil[i] is not None and self.dep_for[i] is not id(self.pil[i]):
                from depth import depth_of
                dm = depth_of(self.pil[i], key=id(self.pil[i])); old = self.dep[i]
                self.dep[i] = self.ctx.texture(dm.size, 1, dm.transpose(Image.FLIP_TOP_BOTTOM).tobytes()); self.dep[i].repeat_x = self.dep[i].repeat_y = False
                self.dep_for[i] = id(self.pil[i]); old.release()
        if pc is not None and self.uses_pc: self.cloud(t, beat, pc[0], seed, pc[1], pc[2], pc[3])
        if "RE" in body: self.rutt(t, beat, seed)
        if "CARDS" in body: self.cards(t, beat, pc[0] if pc else .5, seed)
        import sims as SM
        if "FLUID" in body:
            if self.fluid is None: self.fluid = SM.Fluid(self.ctx, self.quad); self.fluid.reset(self.src[0], self.size[0]); self._fluid_src = id(self.src[0])
            if self._fluid_src != id(self.src[0]): self.fluid.load(self.src[0], self.size[0], .6); self._fluid_src = id(self.src[0])
            self.fluid.force = getattr(self, "fluid_force", 1.0); self._fluid_tex = self.fluid.step(t, beat, self.src[0], self.size[0], seed)
        if "SLIME" in body:
            if self.slime is None: self.slime = SM.Physarum(self.ctx, self.quad)
            self._slime_tex = self.slime.step(t, beat, self.src[0], self.size[0], *self.sim_cols)
        if "ATTR" in body:
            if self.attr is None: self.attr = SM.Attractor(self.ctx, self.w, self.h)
            k, i = self.attr_kind; k = k if k in (0, 1) else 0; params = (SM.CLIFFORD if k == 0 else SM.DEJONG)[i % 5]
            self._attr_tex = self.attr.render(t, beat, k, params, *self.sim_cols)
        prev = self.fb[self.cur].color_attachments[0]; dst = self.fb[1 - self.cur]
        dst.use(); p = self.prog
        for i in range(4):
            self.src[i].use(i)
            if f"S{i}" in p: p[f"S{i}"].value = i
            self._u(p, f"A{i}", tuple(map(float, self.size[i])))
        prev.use(4)
        if "PREV" in p: p["PREV"].value = 4
        if "PC" in p: self.pcfb.color_attachments[0].use(7); p["PC"].value = 7
        for nm, unit, tex in (("FLUID", 14, getattr(self, "_fluid_tex", None)), ("SLIME", 15, getattr(self, "_slime_tex", None)), ("ATTR", 16, getattr(self, "_attr_tex", None)), ("D0", 9, self.dep[0]), ("D1", 10, self.dep[1]), ("D3", 11, self.dep[3]), ("RE", 12, self.refb.color_attachments[0]), ("CARDS", 13, self.cfb.color_attachments[0])):
            if nm in p and tex is not None: tex.use(unit); p[nm].value = unit
        if "SIM" in p and self.sim is not None:
            st = self.sim.step(t, beat, self.src[0], self.size[0], self.sim_seed, self.sim_bal, 10); self.sim_seed *= .6
            dst.use(); st.use(8); p["SIM"].value = 8
        for k, v in (("R", (float(self.w), float(self.h))), ("T", float(t)), ("B", float(beat)), ("E", float(energy)), ("SEED", float(seed % 1.0)), ("MIX", float(mix))): self._u(p, k, v)
        self.vao.render(mode=5); self.cur = 1 - self.cur
        raw = dst.read(components=3)
        img = Image.frombytes("RGB", (self.w, self.h), raw).transpose(Image.FLIP_TOP_BOTTOM)
        return self.timepass(img, t) if self.tmode else img

    def timepass(self, img, t):
        """store this frame in the history, then show every pixel from the moment its map says"""
        self.head = (self.head + 1) % self.tn
        small = img.resize((self.tw, self.th), Image.BILINEAR).transpose(Image.FLIP_TOP_BOTTOM)
        self.hist.write(small.tobytes(), viewport=(0, 0, self.head, self.tw, self.th, 1))
        self.curtex.write(img.transpose(Image.FLIP_TOP_BOTTOM).tobytes())
        p = self.tprog; self.tfb.use(); self.hist.use(0); p["HIST"].value = 0; self.curtex.use(1); p["CUR"].value = 1
        mp = self.dep[0] if self.tmode == 4 else self.src[0]; mp.use(2); p["MAP"].value = 2
        for k, v in (("AM", tuple(map(float, self.size[0]))), ("T", float(t)), ("HEAD", float(self.head)), ("NL", float(self.tn))): self._u(p, k, v)
        p["MODE"].value = int(self.tmode); self.tvao.render(mode=5)
        return Image.frombytes("RGB", (self.w, self.h), self.tfb.read(components=3)).transpose(Image.FLIP_TOP_BOTTOM)

    def rutt(self, t, beat, seed):
        p = self.reprog; self.refb.use(); self.ctx.clear(0, 0, 0, 1); self.ctx.enable(self.ctx.BLEND); self.ctx.blend_func = (self.ctx.ONE, self.ctx.ONE)
        self.src[0].use(5); self.dep[0].use(6); p["SA"].value = 5; p["DA"].value = 6
        for k, v in (("AA", tuple(map(float, self.size[0]))), ("R", (float(self.w), float(self.h))), ("T", float(t)), ("B", float(beat)), ("AMP", .9), ("SEED", float(seed))): self._u(p, k, v)
        self.revao.render(mode=self.ctx.LINES); self.ctx.disable(self.ctx.BLEND)

    def cards(self, t, beat, morph, seed):
        p = self.cprog; self.cfb.use(); self.ctx.clear(0, 0, 0, 0, depth=1.0); self.ctx.enable(self.ctx.DEPTH_TEST)
        for i in range(4):
            self.src[i].use(i); p[f"S{i}"].value = i
        for k, v in (("R", (float(self.w), float(self.h))), ("T", float(t)), ("B", float(beat)), ("M", float(morph)), ("SEED", float(seed)), ("N", float(self.cards_n))): self._u(p, k, v)
        p["FA"].value, p["FB"].value = self.cards_form
        self.cvao.render(mode=5, instances=self.cards_n); self.ctx.disable(self.ctx.DEPTH_TEST)

    def analog(self, im, mode, t, beat=0.0, k=1.0):
        """VHS (1) / composite (2) / film (3) / CRT (4) on a finished frame (PIL RGB in, PIL RGB out)"""
        if not mode: return im
        self.intex.write(im.transpose(Image.FLIP_TOP_BOTTOM).tobytes()); self.anfb.use(); p = self.anprog
        self.intex.use(0); p["IN"].value = 0
        for kk, v in (("R", (float(self.w), float(self.h))), ("T", float(t)), ("B", float(beat)), ("K", float(k))): self._u(p, kk, v)
        p["MODE"].value = int(mode); self.anvao.render(mode=5)
        return Image.frombytes("RGB", (self.w, self.h), self.anfb.read(components=3)).transpose(Image.FLIP_TOP_BOTTOM)


# ------------------------------------------------------------------ dynamical systems: a black/white reaction-diffusion
# and a colour cyclic cellular automaton fighting over the frame. Photos are thrown into it (their edges seed the chemistry),
# kicks inject rings, drops shift the balance of power. Rendered at half resolution, many steps per frame.
SIM_STEP = """#version 330
uniform sampler2D ST, SEEDT; uniform vec2 RS; uniform float F, KK, T, B, SEEDAMT, BAL, NS, TH; uniform vec2 SA;
in vec2 vuv; out vec4 o;
float hash(vec2 p){ p=fract(p*vec2(123.34,456.21)); p+=dot(p,p+45.32); return fract(p.x*p.y); }
float vn(vec2 p){ vec2 i=floor(p), f=fract(p); f=f*f*(3.-2.*f); return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y); }
vec2 cover(vec2 uv, vec2 sz){ float ra=(RS.x/RS.y)/(sz.x/sz.y); vec2 c=uv-.5; if(ra<1.) c.x*=ra; else c.y/=ra; return c+.5; }
void main(){
  vec2 px = 1./RS, uv = vuv; vec4 s = texture(ST, uv);
  vec2 L = -s.xy;
  L += .2*(texture(ST,uv+vec2(px.x,0.)).xy + texture(ST,uv-vec2(px.x,0.)).xy + texture(ST,uv+vec2(0.,px.y)).xy + texture(ST,uv-vec2(0.,px.y)).xy);
  L += .05*(texture(ST,uv+px).xy + texture(ST,uv-px).xy + texture(ST,uv+vec2(px.x,-px.y)).xy + texture(ST,uv+vec2(-px.x,px.y)).xy);
  float a = s.x, b = s.y, r = a*b*b;
  float na = a + (1.0*L.x - r + F*(1.-a)), nb = b + (.5*L.y + r - (KK+F)*b);
  // the photo's edges seed the chemistry
  vec2 q = cover(uv, SA); float l0 = dot(texture(SEEDT, q).rgb, vec3(.3,.59,.11)), l1 = dot(texture(SEEDT, q+vec2(.002,0.)).rgb, vec3(.3,.59,.11));
  nb = max(nb, SEEDAMT*smoothstep(.05,.15,abs(l0-l1)));
  // a kick drops a ring of catalyst somewhere
  vec2 cpos = vec2(hash(vec2(floor(T*2.),1.)), hash(vec2(floor(T*2.),2.)));
  float d = length((uv-cpos)*vec2(RS.x/RS.y,1.)); nb = max(nb, B*.9*smoothstep(.012,0.,abs(d-.06-.1*(1.-B))));
  // cyclic CA: each cell advances when enough neighbours are one state ahead → colour spirals
  float st = floor(s.z*NS+.5), nx = mod(st+1., NS); float c = 0.;
  for (int j=-1;j<=1;j++) for (int i=-1;i<=1;i++) { if (i==0&&j==0) continue; float n = floor(texture(ST, uv+vec2(i,j)*px).z*NS+.5); c += step(abs(n-nx), .5); }
  if (c >= TH) st = nx;
  // territory: slow noise front, pushed by the balance of power
  float w = smoothstep(.45, .55, vn(uv*vec2(3.,5.) + vec2(T*.15, -T*.1)) + BAL);
  o = vec4(clamp(na,0.,1.), clamp(nb,0.,1.), st/NS, mix(s.w, w, .08));
}"""
SIM_SHOW = """#version 330
uniform sampler2D ST; uniform vec2 RS; uniform float T, B, NS, HUE; in vec2 vuv; out vec4 o;
vec3 hsv(float h, float s, float v){ vec3 k=clamp(abs(fract(h+vec3(0.,2./3.,1./3.))*6.-3.)-1.,0.,1.); return v*mix(vec3(1.),k,s); }
void main(){
  vec4 s = texture(ST, vuv); vec2 px = 1./RS;
  float rd = smoothstep(.18, .28, s.y);                                      // black/white labyrinth
  float st = s.z; float e = abs(texture(ST, vuv+vec2(px.x,0.)).z - st) + abs(texture(ST, vuv+vec2(0.,px.y)).z - st);
  vec3 col = hsv(st + HUE + T*.03, .9, 1.) * (1. - smoothstep(.0, .2, e)*.8);   // colour spirals with dark seams
  float w = s.w; float edge = smoothstep(.06, 0., abs(w-.5));
  vec3 c = mix(vec3(rd), col, smoothstep(.4, .6, w)) + edge*(.6+B)*vec3(1.);   // the front between them burns white
  o = vec4(c, 1.);
}"""

class Sim:
    def __init__(self, ctx, quad, w=540, h=960):
        self.ctx, self.w, self.h = ctx, w, h
        self.tex = [ctx.texture((w, h), 4, dtype="f4") for _ in range(2)]
        for t in self.tex: t.repeat_x = t.repeat_y = True
        self.fb = [ctx.framebuffer(color_attachments=[t]) for t in self.tex]; self.cur = 0
        self.show_tex = ctx.texture((w, h), 4); self.show_fb = ctx.framebuffer(color_attachments=[self.show_tex])
        self.step_p = ctx.program(vertex_shader=VERT, fragment_shader=SIM_STEP); self.step_v = ctx.simple_vertex_array(self.step_p, quad, "v")
        self.show_p = ctx.program(vertex_shader=VERT, fragment_shader=SIM_SHOW); self.show_v = ctx.simple_vertex_array(self.show_p, quad, "v")
        self.reset(0)
    def reset(self, seed):
        rng = np.random.default_rng(int(seed * 1e6) % 2**31); st = np.zeros((self.h, self.w, 4), "f4"); st[..., 0] = 1
        for _ in range(40):
            y, x = rng.integers(0, self.h - 12), rng.integers(0, self.w - 12); st[y:y + 10, x:x + 10, 1] = 1
        self.ns, self.th = [(4., 3.), (5., 3.), (4., 3.)][int(seed * 31) % 3]
        st[..., 2] = rng.integers(0, int(self.ns), (self.h, self.w)) / self.ns; st[..., 3] = .5
        self.tex[self.cur].write(st.tobytes()); self.params = [(.037, .06), (.03, .062), (.025, .06), (.039, .058), (.022, .051)][int(seed * 97) % 5]
        self.step(0., 0., None, (1, 1), 0., 0., 360, show=False)      # pre-grow: spirals and labyrinths already formed when it first appears
    def step(self, t, beat, seed_tex, seed_size, seed_amt, bal, n=10, show=True):
        p = self.step_p
        for i in range(n):
            src = self.tex[self.cur]; dst = self.fb[1 - self.cur]; dst.use(); src.use(0); p["ST"].value = 0
            if seed_tex is not None and "SEEDT" in p: seed_tex.use(1); p["SEEDT"].value = 1
            for k, v in (("RS", (float(self.w), float(self.h))), ("F", self.params[0]), ("KK", self.params[1]), ("T", float(t)), ("B", float(beat) if i == 0 else 0.),
                         ("SEEDAMT", float(seed_amt) if i == 0 else 0.), ("BAL", float(bal)), ("NS", self.ns), ("TH", self.th), ("SA", tuple(map(float, seed_size)))):
                if k in p: p[k].value = v
            self.step_v.render(mode=5); self.cur = 1 - self.cur
        if not show: return None
        self.show_fb.use(); self.tex[self.cur].use(0); q = self.show_p; q["ST"].value = 0
        for k, v in (("RS", (float(self.w), float(self.h))), ("T", float(t)), ("B", float(beat)), ("NS", self.ns), ("HUE", float((t * .01) % 1))):
            if k in q: q[k].value = v
        self.show_v.render(mode=5)
        return self.show_tex
