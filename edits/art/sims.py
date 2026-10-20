#!/usr/bin/env python3
"""Simulations from the literature, on the GPU (moderngl; Mesa llvmpipe on a CPU runner), all fed by photos.

    Fluid      Stam, "Stable Fluids" (SIGGRAPH 1999): semi-Lagrangian advection, Jacobi pressure projection, plus
               vorticity confinement (Fedkiw, Stam & Jensen 2001). The photo is the dye; kicks are force splats.
    Physarum   Jones, "Characteristics of pattern formation and evolution in approximations of Physarum transport
               networks" (Artificial Life 2010): agents sense ahead/left/right, turn toward trail, move, deposit;
               the trail diffuses and decays. Here the photo's brightness is also food, so the network draws the image.
    Attractor  Clifford / Peter de Jong maps: points iterated through x' = sin(a y) + c cos(a x), y' = sin(b x) + d cos(b y)
               (de Jong: x' = sin(a y) - cos(b x), y' = sin(c x) - cos(d y)), density accumulated and log tone-mapped.

All return a moderngl texture the compositor samples as a source.
"""
import math
import numpy as np

VERT = "#version 330\nin vec2 v; out vec2 vuv; void main(){ vuv = v*.5+.5; gl_Position = vec4(v,0.,1.); }"

def _prog(ctx, quad, fs):
    p = ctx.program(vertex_shader=VERT, fragment_shader="#version 330\nin vec2 vuv; out vec4 o;\n" + fs); return p, ctx.simple_vertex_array(p, quad, "v")

def _set(p, **kw):
    for k, v in kw.items():
        if k in p: p[k].value = v

COVER = "vec2 cover(vec2 uv, vec2 sz, vec2 R){ float ra=(R.x/R.y)/(sz.x/sz.y); vec2 c=uv-.5; if(ra<1.) c.x*=ra; else c.y/=ra; return c+.5; }\n"

# ============================================================ stable fluids
class Fluid:
    def __init__(self, ctx, quad, w=360, h=640):
        self.ctx, self.w, self.h = ctx, w, h
        mk = lambda c: ctx.texture((w, h), c, dtype="f4")
        self.vel = [mk(2), mk(2)]; self.prs = [mk(1), mk(1)]; self.div = mk(1); self.curl = mk(1); self.dye = [mk(4), mk(4)]
        for t in self.vel + self.prs + self.dye + [self.div, self.curl]: t.repeat_x = t.repeat_y = False
        fb = lambda t: ctx.framebuffer(color_attachments=[t])
        self.fvel = [fb(t) for t in self.vel]; self.fprs = [fb(t) for t in self.prs]; self.fdiv = fb(self.div); self.fcurl = fb(self.curl); self.fdye = [fb(t) for t in self.dye]
        self.cv = self.cp = self.cd = 0
        self.advect = _prog(ctx, quad, "uniform sampler2D V, Q; uniform vec2 TX; uniform float DT, DIS; void main(){ vec2 p = vuv - DT*texture(V, vuv).xy*TX; o = DIS*texture(Q, p); }")
        self.splat = _prog(ctx, quad, "uniform sampler2D Q; uniform vec2 PT, AS; uniform vec3 C; uniform float RAD; void main(){ vec2 d = (vuv-PT)*AS; o = texture(Q, vuv) + vec4(C*exp(-dot(d,d)/RAD), 0.); }")
        self.curlp = _prog(ctx, quad, "uniform sampler2D V; uniform vec2 TX; void main(){ float l=texture(V,vuv-vec2(TX.x,0.)).y, r=texture(V,vuv+vec2(TX.x,0.)).y, b=texture(V,vuv-vec2(0.,TX.y)).x, t=texture(V,vuv+vec2(0.,TX.y)).x; o = vec4(.5*(r-l-t+b)); }")
        self.vort = _prog(ctx, quad, "uniform sampler2D V, CU; uniform vec2 TX; uniform float EPS, DT; void main(){ float l=texture(CU,vuv-vec2(TX.x,0.)).x, r=texture(CU,vuv+vec2(TX.x,0.)).x, b=texture(CU,vuv-vec2(0.,TX.y)).x, t=texture(CU,vuv+vec2(0.,TX.y)).x, c=texture(CU,vuv).x; vec2 f = .5*vec2(abs(t)-abs(b), abs(r)-abs(l)); f /= length(f)+1e-5; f *= EPS*c*vec2(1.,-1.); o = vec4(texture(V,vuv).xy + f*DT, 0., 0.); }")
        self.divp = _prog(ctx, quad, "uniform sampler2D V; uniform vec2 TX; void main(){ float l=texture(V,vuv-vec2(TX.x,0.)).x, r=texture(V,vuv+vec2(TX.x,0.)).x, b=texture(V,vuv-vec2(0.,TX.y)).y, t=texture(V,vuv+vec2(0.,TX.y)).y; o = vec4(.5*(r-l+t-b)); }")
        self.jac = _prog(ctx, quad, "uniform sampler2D P, D; uniform vec2 TX; void main(){ float l=texture(P,vuv-vec2(TX.x,0.)).x, r=texture(P,vuv+vec2(TX.x,0.)).x, b=texture(P,vuv-vec2(0.,TX.y)).x, t=texture(P,vuv+vec2(0.,TX.y)).x; o = vec4((l+r+b+t-texture(D,vuv).x)*.25); }")
        self.grad = _prog(ctx, quad, "uniform sampler2D P, V; uniform vec2 TX; void main(){ float l=texture(P,vuv-vec2(TX.x,0.)).x, r=texture(P,vuv+vec2(TX.x,0.)).x, b=texture(P,vuv-vec2(0.,TX.y)).x, t=texture(P,vuv+vec2(0.,TX.y)).x; o = vec4(texture(V,vuv).xy - .5*vec2(r-l, t-b), 0., 0.); }")
        self.inject = _prog(ctx, quad, COVER + "uniform sampler2D Q, S; uniform vec2 SZ, R; uniform float K; void main(){ vec4 s = texture(S, cover(vuv, SZ, R)); o = mix(texture(Q, vuv), vec4(s.rgb, 1.), K); }")
        self.quad = quad; self.t_last = None; self.rng = np.random.default_rng(1)

    def _run(self, pv, fb, **u):
        p, vao = pv; fb.use(); _set(p, **u); vao.render(mode=5)

    def reset(self, src, size):
        for f in self.fvel + self.fprs + self.fdye: f.use(); self.ctx.clear(0, 0, 0, 0)
        self.load(src, size, 1.0)

    def load(self, src, size, k):
        """blend the photo into the dye (k=1: replace; small k: the photo keeps re-forming through the swirl)"""
        p, vao = self.inject; d0, d1 = self.dye[self.cd], self.fdye[1 - self.cd]
        d0.use(0); src.use(1); _set(p, Q=0, S=1, SZ=tuple(map(float, size)), R=(float(self.w), float(self.h)), K=float(k)); d1.use(); vao.render(mode=5); self.cd = 1 - self.cd

    def step(self, t, beat, src, size, seed):
        TX = (1 / self.w, 1 / self.h); DT = 1 / 24; ctx = self.ctx
        if beat > .9 or self.rng.random() < .08:                                # a kick throws a few force splats
            for _ in range(int(2 + 3 * beat)):
                pt = (float(self.rng.uniform(.15, .85)), float(self.rng.uniform(.2, .8))); ang = self.rng.uniform(0, 6.283); f = 900 * (.4 + beat)
                p, vao = self.splat; self.vel[self.cv].use(0); _set(p, Q=0, PT=pt, AS=(self.w / self.h, 1.0), C=(math.cos(ang) * f, math.sin(ang) * f, 0.0), RAD=.0016); self.fvel[1 - self.cv].use(); vao.render(mode=5); self.cv = 1 - self.cv
        # vorticity confinement keeps the small swirls alive
        p, vao = self.curlp; self.vel[self.cv].use(0); _set(p, V=0, TX=TX); self.fcurl.use(); vao.render(mode=5)
        p, vao = self.vort; self.vel[self.cv].use(0); self.curl.use(1); _set(p, V=0, CU=1, TX=TX, EPS=28.0, DT=DT); self.fvel[1 - self.cv].use(); vao.render(mode=5); self.cv = 1 - self.cv
        # projection: divergence → pressure (Jacobi) → subtract gradient
        p, vao = self.divp; self.vel[self.cv].use(0); _set(p, V=0, TX=TX); self.fdiv.use(); vao.render(mode=5)
        p, vao = self.jac
        for _ in range(22):
            self.prs[self.cp].use(0); self.div.use(1); _set(p, P=0, D=1, TX=TX); self.fprs[1 - self.cp].use(); vao.render(mode=5); self.cp = 1 - self.cp
        p, vao = self.grad; self.prs[self.cp].use(0); self.vel[self.cv].use(1); _set(p, P=0, V=1, TX=TX); self.fvel[1 - self.cv].use(); vao.render(mode=5); self.cv = 1 - self.cv
        # advect velocity and dye
        p, vao = self.advect
        self.vel[self.cv].use(0); _set(p, V=0, Q=0, TX=TX, DT=DT * 60, DIS=.995); self.fvel[1 - self.cv].use(); vao.render(mode=5); self.cv = 1 - self.cv
        self.vel[self.cv].use(0); self.dye[self.cd].use(1); _set(p, V=0, Q=1, TX=TX, DT=DT * 60, DIS=.999); self.fdye[1 - self.cd].use(); vao.render(mode=5); self.cd = 1 - self.cd
        self.load(src, size, .06)                                            # the image keeps re-forming through the swirl
        return self.dye[self.cd]

# ============================================================ physarum
class Physarum:
    def __init__(self, ctx, quad, w=540, h=960, side=512):
        self.ctx, self.w, self.h, self.n = ctx, w, h, side * side; self.side = side
        self.ag = [ctx.texture((side, side), 4, dtype="f4") for _ in range(2)]; self.fag = [ctx.framebuffer(color_attachments=[t]) for t in self.ag]; self.ca = 0
        for t in self.ag: t.filter = (ctx.NEAREST, ctx.NEAREST)
        self.tr = [ctx.texture((w, h), 1, dtype="f4") for _ in range(2)]; self.ftr = [ctx.framebuffer(color_attachments=[t]) for t in self.tr]; self.ct = 0
        self.show = ctx.texture((w, h), 4); self.fshow = ctx.framebuffer(color_attachments=[self.show])
        self.move = _prog(ctx, quad, COVER + """uniform sampler2D A, TR, F; uniform vec2 RS, FS; uniform float T, B, SA, SD, RA, SPD, SEED;
float h(vec2 p){ p=fract(p*vec2(123.34,456.21)); p+=dot(p,p+45.32); return fract(p.x*p.y); }
float sense(vec2 p, float a, float off){ vec2 q = p + vec2(cos(a+off), sin(a+off))*SD/RS; q = fract(q);
  return texture(TR, q).r + 5.5*pow(dot(texture(F, cover(q, FS, RS)).rgb, vec3(.3,.59,.11)), 1.5); }
void main(){ vec4 s = texture(A, vuv); vec2 p = s.xy; float a = s.z;
  float f = sense(p,a,0.), l = sense(p,a,SA), r = sense(p,a,-SA), rnd = h(vuv*91.+T);
  if (f > l && f > r) {} else if (f < l && f < r) a += (rnd-.5)*2.*RA; else if (l > r) a += RA; else a -= RA;
  p += vec2(cos(a), sin(a))*SPD*(1.+1.5*B)/RS; p = fract(p); o = vec4(p, a, 1.); }""")
        self.depo_vs = ctx.program(vertex_shader="""#version 330
uniform sampler2D A; uniform int SIDE; void main(){ ivec2 ij = ivec2(gl_VertexID % SIDE, gl_VertexID / SIDE); vec2 p = texelFetch(A, ij, 0).xy; gl_Position = vec4(p*2.-1., 0., 1.); gl_PointSize = 1.; }""",
            fragment_shader="#version 330\nout vec4 o; void main(){ o = vec4(.05); }")
        self.depo_vao = ctx.vertex_array(self.depo_vs, [])
        self.diff = _prog(ctx, quad, "uniform sampler2D TR; uniform vec2 TX; uniform float DEC; void main(){ float s=0.; for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++) s+=texture(TR, vuv+vec2(i,j)*TX).r; o = vec4(s/9.*DEC); }")
        self.showp = _prog(ctx, quad, COVER + """uniform sampler2D TR, F; uniform vec2 FS, RS; uniform vec3 C1, C2; uniform float B;
void main(){ float t = texture(TR, vuv).r; float k = 1.-exp(-t*3.); vec3 ph = texture(F, cover(vuv, FS, RS)).rgb;
  vec3 c = mix(vec3(0.), mix(C1, ph*1.4+.15, .55), k); c += C2*pow(k, 6.)*(1.+B); o = vec4(c, 1.); }""")
        self.reset(0.3)

    def reset(self, seed):
        rng = np.random.default_rng(int(seed * 1e6) % 2**31); s = self.side
        a = np.zeros((s, s, 4), "f4"); rr = np.sqrt(rng.random((s, s))) * .5; th = rng.random((s, s)) * 6.283
        a[..., 0] = .5 + rr * np.cos(th) * self.h / self.w; a[..., 1] = .5 + rr * np.sin(th); a[..., 2] = th + math.pi; a[..., 3] = 1   # a disc, all heading inward
        a[..., :2] %= 1.0; self.ag[self.ca].write(a.tobytes())
        for f in self.ftr: f.use(); self.ctx.clear(0, 0, 0, 0)

    def step(self, t, beat, food, fsize, c1, c2, n=2):
        ctx = self.ctx
        for _ in range(n):
            p, vao = self.move; self.ag[self.ca].use(0); self.tr[self.ct].use(1); food.use(2)
            _set(p, A=0, TR=1, F=2, RS=(float(self.w), float(self.h)), FS=tuple(map(float, fsize)), T=float(t), B=float(beat), SA=.4, SD=9.0, RA=.78, SPD=1.2)
            self.fag[1 - self.ca].use(); vao.render(mode=5); self.ca = 1 - self.ca
            self.ftr[self.ct].use(); ctx.enable(ctx.BLEND | ctx.PROGRAM_POINT_SIZE); ctx.blend_func = (ctx.ONE, ctx.ONE)
            self.ag[self.ca].use(0); self.depo_vs["A"].value = 0; self.depo_vs["SIDE"].value = self.side; self.depo_vao.render(mode=ctx.POINTS, vertices=self.n); ctx.disable(ctx.BLEND)
            p, vao = self.diff; self.tr[self.ct].use(0); _set(p, TR=0, TX=(1 / self.w, 1 / self.h), DEC=.93); self.ftr[1 - self.ct].use(); vao.render(mode=5); self.ct = 1 - self.ct
        p, vao = self.showp; self.tr[self.ct].use(0); food.use(1)
        _set(p, TR=0, F=1, FS=tuple(map(float, fsize)), RS=(float(self.w), float(self.h)), C1=tuple(c / 255 for c in c1), C2=tuple(c / 255 for c in c2), B=float(beat)); self.fshow.use(); vao.render(mode=5)
        return self.show

# ============================================================ strange attractors
class Attractor:
    def __init__(self, ctx, w=1080, h=1920, n=520000):
        self.ctx, self.w, self.h, self.n = ctx, w, h, n
        self.acc = ctx.texture((w, h), 1, dtype="f4"); self.facc = ctx.framebuffer(color_attachments=[self.acc])
        self.show = ctx.texture((w, h), 4); self.fshow = ctx.framebuffer(color_attachments=[self.show])
        self.pts = ctx.program(vertex_shader="""#version 330
uniform vec4 P; uniform int KIND; uniform vec2 R; uniform float T, B, SC;
float h(float n){ return fract(sin(n*12.9898)*43758.5453); }
void main(){ float id = float(gl_VertexID); vec2 p = vec2(h(id)*2.-1., h(id+.37)*2.-1.)*1.5; vec3 q = vec3(p, h(id+.71));
  if (KIND == 2) { q = vec3(.1+h(id)*.2, h(id+.3)*.2, h(id+.6)*.2); for (int i=0;i<60;i++){ vec3 d = vec3(10.*(q.y-q.x), q.x*(28.-q.z)-q.y, q.x*q.y-2.6667*q.z); q += d*.006; } p = vec2(q.x*.045, (q.z-25.)*.045); }
  else { for (int i=0;i<36;i++){ p = KIND==0 ? vec2(sin(P.x*p.y)+P.z*cos(P.x*p.x), sin(P.y*p.x)+P.w*cos(P.y*p.y)) : vec2(sin(P.x*p.y)-cos(P.y*p.x), sin(P.z*p.x)-cos(P.w*p.y)); } p *= .42; }
  float a = T*.12; p = mat2(cos(a),-sin(a),sin(a),cos(a))*p; p *= SC*(1.+.06*B);
  gl_Position = vec4(p.x*R.y/R.x, p.y, 0., 1.); gl_PointSize = 1.; }""", fragment_shader="#version 330\nout vec4 o; void main(){ o = vec4(1.); }")
        self.vao = ctx.vertex_array(self.pts, [])
        self.tone = _prog(ctx, ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes()), """uniform sampler2D ACC; uniform vec3 C1, C2; uniform float B, GAIN;
void main(){ float d = texture(ACC, vuv).r; float k = log(1.+d*GAIN)/log(1.+40.*GAIN); k = clamp(k, 0., 1.);
  vec3 c = mix(C1*k, mix(C1, C2, smoothstep(.45,.9,k)), k) + vec3(pow(k, 3.))*(.6+.6*B); o = vec4(c, 1.); }""")

    def render(self, t, beat, kind, params, c1, c2):
        ctx = self.ctx; self.facc.use(); ctx.clear(0, 0, 0, 0); ctx.enable(ctx.BLEND | ctx.PROGRAM_POINT_SIZE); ctx.blend_func = (ctx.ONE, ctx.ONE)
        a, b, c, d = params; drift = .08 * math.sin(t * .5) + .05 * beat
        _set(self.pts, P=(a + drift, b - drift * .6, c + drift * .4, d), KIND=int(kind), R=(float(self.w), float(self.h)), T=float(t), B=float(beat), SC=.95 if kind != 2 else 1.0)
        self.vao.render(mode=ctx.POINTS, vertices=self.n); ctx.disable(ctx.BLEND)
        p, vao = self.tone; self.acc.use(0); _set(p, ACC=0, C1=tuple(x / 255 for x in c1), C2=tuple(x / 255 for x in c2), B=float(beat), GAIN=.8); self.fshow.use(); vao.render(mode=5)
        return self.show

CLIFFORD = [(-1.4, 1.6, 1.0, .7), (1.7, 1.7, .6, 1.2), (-1.7, 1.3, -.1, -1.2), (-1.8, -2.0, -.5, -.9), (1.5, -1.8, 1.6, .9)]
DEJONG = [(-2.0, -2.0, -1.2, 2.0), (1.641, 1.902, .316, 1.525), (.970, -1.899, 1.381, -1.506), (-2.7, -.09, -.86, -2.2), (1.4, -2.3, 2.4, -2.1)]
