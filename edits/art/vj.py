#!/usr/bin/env python3
"""VJ layer: GLSL fragment shaders rendered headless (moderngl + EGL; on a CPU-only runner Mesa llvmpipe does it).
Raymarched 3D shapes, tunnels, plasma and a kaleidoscope that eats a cut-out — all driven by time, beat and energy.

    v = VJ(360, 640)                          # render size (upscaled by the caller)
    img = v.frame("boxes", t, beat, energy, seed)          # PIL RGB
    img = v.frame("kaleido", t, beat, energy, seed, tex=pil_rgba)
"""
import numpy as np
from PIL import Image

HEAD = """#version 330
uniform vec2 R; uniform float T, B, E, S; uniform sampler2D TEX;
out vec4 o;
vec3 pal(float x){ vec3 a=vec3(.5), b=vec3(.5), c=vec3(1.), d=fract(vec3(.0,.33,.67)+S*vec3(.13,.29,.41)); return a+b*cos(6.28318*(c*x+d)); }
mat2 rot(float a){ float c=cos(a), s=sin(a); return mat2(c,-s,s,c); }
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453); }
"""

SHADERS = {
# infinite field of rotating neon cubes, camera flying through, beat pumps the size
"boxes": """
float map(vec3 p){
  vec3 id = floor(p/4.); p = mod(p,4.)-2.;
  p.xy *= rot(T*.7+id.z*.5+B*.6); p.yz *= rot(T*.5+id.x);
  vec3 q = abs(p)-vec3(.55+.25*B);
  float box = length(max(q,0.))+min(max(q.x,max(q.y,q.z)),0.);
  return max(box, -(length(p)-.62-.2*B));
}
void main(){
  vec2 uv=(gl_FragCoord.xy-.5*R)/R.y; vec3 ro=vec3(0.,0.,T*6.), rd=normalize(vec3(uv,1.));
  rd.xy*=rot(T*.15+S); float t=0., g=0.;
  for(int i=0;i<64;i++){ float d=map(ro+rd*t); g+=.02/(.02+abs(d)); if(d<.002||t>60.) break; t+=d*.9; }
  vec3 col = pal(t*.04+T*.1)*g*.06*(1.+E) + pal(t*.02+.5)*exp(-t*.06)*.6;
  o=vec4(col,1.);
}""",
# polar tunnel with stripes, hue cycling, beat flashes the rings
"tunnel": """
void main(){
  vec2 uv=(gl_FragCoord.xy-.5*R)/R.y; uv*=rot(T*.3);
  float r=length(uv), a=atan(uv.y,uv.x); float z=.4/r+T*2.5;
  float n=floor(6.+S*10.); float st=step(.5,fract(z*1.2))*.7+.3*step(.5,fract(a*n/6.28318+z*.3));
  vec3 col = pal(z*.08+a*.05)*st*(.3+r*2.)*(1.+1.5*B*step(.85,fract(z*.5)));
  o=vec4(col*smoothstep(0.,.08,r),1.);
}""",
# raymarched twisted torus with iridescent shading
"knot": """
float map(vec3 p){
  p.xz*=rot(T*.6); p.xy*=rot(T*.4+S);
  float a=atan(p.z,p.x); vec2 q=vec2(length(p.xz)-1.2,p.y); q*=rot(a*(2.+floor(S*3.))+T);
  q.x-=.35*(1.+.4*B); return length(q)-.28-.06*sin(a*12.+T*4.)*E;
}
vec3 nor(vec3 p){ vec2 e=vec2(.002,0.); return normalize(vec3(map(p+e.xyy)-map(p-e.xyy),map(p+e.yxy)-map(p-e.yxy),map(p+e.yyx)-map(p-e.yyx))); }
void main(){
  vec2 uv=(gl_FragCoord.xy-.5*R)/R.y; vec3 ro=vec3(0.,0.,-5.6), rd=normalize(vec3(uv,1.5)); float t=0.;
  for(int i=0;i<72;i++){ float d=map(ro+rd*t); if(d<.001||t>10.) break; t+=d*.8; }
  vec3 bg = pal(uv.y*.4+T*.05)*.18*(1.+B);
  if(t<10.){ vec3 p=ro+rd*t, n=nor(p); float f=pow(1.-max(dot(n,-rd),0.),2.);
    vec3 col = pal(dot(n,vec3(.3,.5,.8))+T*.15)*(.25+.75*max(dot(n,normalize(vec3(1,1,-1))),0.)) + pal(f+.4)*f*1.4;
    o=vec4(col,1.); } else o=vec4(bg,1.);
}""",
# posterised plasma, hard colour bands
"plasma": """
void main(){
  vec2 uv=gl_FragCoord.xy/R.y*3.; float t=T*.8;
  float v=sin(uv.x*1.3+t)+sin((uv.y+t)*1.7)+sin((uv.x+uv.y+t)*1.1)+sin(length(uv-vec2(1.5,2.5)+vec2(sin(t),cos(t)))*3.);
  v = floor((v*.25+.5)*(5.+floor(S*5.)))/(5.+floor(S*5.));
  o=vec4(pal(v+T*.07)*(1.+B*.8),1.);
}""",
# neon rings floating in perspective over a grid floor
"rings": """
float map(vec3 p){
  float fl = p.y+1.6;
  vec3 q=p; q.z=mod(q.z+T*5.,3.)-1.5; float ring=length(vec2(length(q.xy)-1.4-.3*B,q.z))-.05;
  return min(fl, ring);
}
void main(){
  vec2 uv=(gl_FragCoord.xy-.5*R)/R.y; vec3 ro=vec3(0.,0.,0.), rd=normalize(vec3(uv,1.)); rd.xy*=rot(sin(T*.4)*.3);
  float t=0., g=0.; vec3 p;
  for(int i=0;i<64;i++){ p=ro+rd*t; float d=map(p); g+=.01/(.01+abs(d-(p.y+1.6>.01?0.:0.))); if(d<.002||t>40.) break; t+=d; }
  vec3 col = pal(t*.05+T*.1)*g*.05;
  if(abs(p.y+1.6)<.01){ vec2 gr=abs(fract(p.xz*.8)-.5); float line=smoothstep(.04,0.,min(gr.x,gr.y)); col+=pal(p.z*.02+T*.2)*line*(1.+B)*exp(-t*.05); }
  o=vec4(col,1.);
}""",
# kaleidoscope of a cut-out texture: the element becomes a mandala
"kaleido": """
void main(){
  vec2 uv=(gl_FragCoord.xy-.5*R)/R.y; float n=floor(5.+S*7.);
  float r=length(uv), a=atan(uv.y,uv.x)+T*.25; a=mod(a,6.28318/n); a=abs(a-3.14159/n);
  vec2 p=vec2(cos(a),sin(a))*r; p*=rot(T*.2); p = p*(1.2-.25*B) + vec2(.5) + vec2(sin(T*.3),cos(T*.25))*.15;
  vec4 c=texture(TEX, fract(p));
  vec3 bg=pal(r*1.5-T*.3)*.35*(1.+B);
  o=vec4(mix(bg, c.rgb*(1.+.6*B), c.a),1.);
}""",
# kaleidoscopic IFS fractal: folded, rotated boxes — crazy 3D that never repeats
"kifs": """
float map(vec3 p){
  p.xz*=rot(T*.3); p.xy*=rot(T*.2+S*3.);
  float sc=1.;
  for(int i=0;i<7;i++){ p=abs(p); if(p.x<p.y) p.xy=p.yx; if(p.x<p.z) p.xz=p.zx; if(p.y<p.z) p.yz=p.zy;
    p = p*1.9 - vec3(1.1+.25*B,1.,.9)*.9; sc*=1.9; p.xy*=rot(.3+S); }
  return (length(max(abs(p)-vec3(.5),0.)))/sc;
}
void main(){
  vec2 uv=(gl_FragCoord.xy-.5*R)/R.y; vec3 ro=vec3(0.,0.,-3.2+sin(T*.4)*.4), rd=normalize(vec3(uv,1.2)); float t=0., g=0.; int it=0;
  for(int i=0;i<80;i++){ float d=map(ro+rd*t); g+=.008/(.01+d); if(d<.0015||t>8.) break; t+=d; it=i; }
  vec3 col = pal(float(it)*.03+T*.12)*g*.08*(1.+B) ;
  if(t<8.) col += pal(t*.3+.2)*(1.-float(it)/80.)*1.2;
  o=vec4(col,1.);
}""",
# menger sponge fly-through
"menger": """
float box(vec3 p, vec3 b){ vec3 q=abs(p)-b; return length(max(q,0.))+min(max(q.x,max(q.y,q.z)),0.); }
float map(vec3 p){
  p.z+=T*1.5; p=mod(p+1.,2.)-1.; float d=box(p,vec3(1.)); float s=1.;
  for(int m=0;m<4;m++){ vec3 a=mod(p*s,2.)-1.; s*=3.; vec3 r=abs(1.-3.*abs(a));
    float da=max(r.x,r.y), db=max(r.y,r.z), dc=max(r.z,r.x); float c=(min(da,min(db,dc))-1.)/s; d=max(d,c); }
  return d;
}
void main(){
  vec2 uv=(gl_FragCoord.xy-.5*R)/R.y; vec3 ro=vec3(.1,.2,0.), rd=normalize(vec3(uv,1.)); rd.xy*=rot(T*.2+S*6.); float t=0.; int it=0;
  for(int i=0;i<80;i++){ float d=map(ro+rd*t); if(d<.001||t>12.) break; t+=d; it=i; }
  vec3 col = pal(t*.15+T*.1+float(it)*.01)*(1.-float(it)/80.)*1.4*(1.+.8*B);
  o=vec4(col*exp(-t*.12),1.);
}""",
# hexagonal tessellation of a cut-out: every cell rotates on a ripple from the centre, tinted per cell, glowing cell edges
"hextile": """
vec4 hexc(vec2 p){ vec2 r=vec2(1.,1.7320508), h=r*.5; vec2 a=mod(p,r)-h, b=mod(p-h,r)-h; vec2 g=dot(a,a)<dot(b,b)?a:b; return vec4(g,p-g); }
void main(){
  vec2 uv=(gl_FragCoord.xy-.5*R)/R.y; uv*=rot(T*.04+S*3.);
  float sc=2.6+2.2*S+.35*sin(T*.37); vec4 h=hexc(uv*sc+vec2(T*.11,T*.23)); vec2 id=h.zw; float d=length(id-vec2(T*.11,T*.23));
  float a=T*.9-d*.7+B*.6+floor(hash(id)*6.)*1.0471976;
  vec2 q=(h.xy*rot(a))*(1.25-.15*B)+.5;
  vec4 tx=(q.x>0.&&q.x<1.&&q.y>0.&&q.y<1.)?texture(TEX,q):vec4(0.);
  vec3 tint=pal(hash(id)*.6+T*.12+d*.08);
  vec3 bg=pal(d*.12-T*.25+.5)*(.18+.1*sin(d*3.-T*4.));
  float hd=max(abs(h.x),dot(abs(h.xy),vec2(.8660254,.5)));
  float edge=smoothstep(.44,.5,hd)*(.6+1.2*B);
  vec3 col=mix(bg,tx.rgb*mix(vec3(1.),tint*1.7,.55),tx.a)+tint*edge;
  o=vec4(col,1.);
}""",
# mirrored square wallpaper (p4m) of a cut-out; random cells split into 2x2 smaller copies, re-rolled every half bar
"wallp": """
void main(){
  vec2 uv=(gl_FragCoord.xy-.5*R)/R.y;
  vec2 p=uv*(2.2+2.4*S)*(1.+.06*sin(T*.5)); p*=rot(T*.06); p+=vec2(T*.15,-T*.1);
  vec2 id=floor(p), f=fract(p);
  if(mod(id.x,2.)>.5) f.x=1.-f.x; if(mod(id.y,2.)>.5) f.y=1.-f.y;
  float lvl=0.; vec2 sid=id;
  if(hash(id+floor(T*.9))>.5){ sid=id*2.+floor(f*2.); f=fract(f*2.); lvl=1.; if(hash(sid+3.1+floor(T*.9))>.7){ sid=sid*2.+floor(f*2.); f=fract(f*2.); lvl=2.; } }
  vec2 q=(f-.5)*rot(B*.35*(hash(sid)-.5))*(1.12-.12*B)+.5;
  vec4 tx=(q.x>0.&&q.x<1.&&q.y>0.&&q.y<1.)?texture(TEX,q):vec4(0.);
  float chk=mod(id.x+id.y,2.);
  vec3 bg=pal(hash(sid)*.4+T*.08+lvl*.2+length(uv)*.3)*(chk>.5?.32:.14);
  vec3 tint=pal(hash(sid)+.3+T*.1);
  o=vec4(mix(bg,tx.rgb*mix(vec3(1.),tint*1.6,.5),tx.a),1.);
}""",
# two layers of truchet arcs at different scales, flowing; beat thickens the lines
"truchet": """
float layer(vec2 p, float k){
  vec2 id=floor(p), f=fract(p)-.5; if(hash(id+k+floor(T*.4+hash(id)*4.))>.5) f.x=-f.x;
  vec2 c=(f.x+f.y>0.)?vec2(.5):vec2(-.5); float d=abs(length(f-c)-.5);
  float w=.07+.06*B; return smoothstep(w,w-.03,d);
}
void main(){
  vec2 uv=(gl_FragCoord.xy-.5*R)/R.y; uv*=rot(.25*sin(T*.21)+S);
  float n=5.+4.*S; vec2 p=uv*n+vec2(0.,T*.5);
  float a=layer(p,0.), b=layer(p*2.+vec2(T*.3,0.),7.);
  vec3 col=pal(floor(p.x)*.05+floor(p.y)*.03+T*.15)*a*1.3+pal(.5+length(uv)+T*.1)*b*.55*(1.-a);
  col+=pal(length(uv)*.6-T*.2)*.06;
  o=vec4(col,1.);
}""",
# op-art moire: two drifting ring sources multiplied, sliced by rotating spokes
"opart": """
void main(){
  vec2 uv=(gl_FragCoord.xy-.5*R)/R.y;
  vec2 c1=vec2(.18*sin(T*.5),.12*cos(T*.3)), c2=vec2(-.16*cos(T*.4),-.14*sin(T*.45));
  float r1=length(uv-c1), r2=length(uv-c2), f=60.+30.*S;
  float s=sign(sin(r1*f-T*3.)*sin(r2*f+T*2.));
  float ang=atan(uv.y,uv.x), sp=sign(sin(ang*(10.+floor(S*14.))+r1*18.-T*2.5));
  vec3 a=pal(r1*.8+T*.1), b=pal(r2*.8+.5-T*.07);
  vec3 col=mix(a,b,.5+.5*s)*(.55+.45*sp)*(1.+.6*B);
  o=vec4(col,1.);
}""",
}

VERT = "#version 330\nin vec2 v; void main(){ gl_Position=vec4(v,0.,1.); }"


class VJ:
    def __init__(self, w=360, h=640):
        import moderngl
        try: self.ctx = moderngl.create_standalone_context(backend="egl")
        except Exception: self.ctx = moderngl.create_standalone_context()
        self.w, self.h = w, h; self.fbo = self.ctx.simple_framebuffer((w, h)); self.fbo.use()
        self.vbo = self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes()); self.progs = {}; self.tex = None; self._texsrc = None
    def _prog(self, name):
        if name not in self.progs:
            p = self.ctx.program(vertex_shader=VERT, fragment_shader=HEAD + SHADERS[name]); self.progs[name] = (p, self.ctx.simple_vertex_array(p, self.vbo, "v"))
        return self.progs[name]
    def _set(self, p, k, v):
        if k in p: p[k].value = v
    def frame(self, name, t, beat=0.0, energy=0.5, seed=0.0, tex=None):
        p, vao = self._prog(name)
        if tex is not None and tex is not self._texsrc:
            im = tex.convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM); im.thumbnail((512, 512))
            if self.tex: self.tex.release()
            self.tex = self.ctx.texture(im.size, 4, im.tobytes()); self.tex.repeat_x = self.tex.repeat_y = True; self.tex.build_mipmaps(); self._texsrc = tex   # mipmaps: small tiles stay clean
        if self.tex is not None and "TEX" in p: self.tex.use(0); p["TEX"].value = 0
        for k, v in (("R", (float(self.w), float(self.h))), ("T", float(t)), ("B", float(beat)), ("E", float(energy)), ("S", float(seed % 1.0))): self._set(p, k, v)
        self.fbo.clear(0, 0, 0, 1); vao.render(mode=5)            # TRIANGLE_STRIP
        return Image.frombytes("RGB", (self.w, self.h), self.fbo.read(components=3)).transpose(Image.FLIP_TOP_BOTTOM)


if __name__ == "__main__":
    import sys, time, os
    out = sys.argv[1] if len(sys.argv) > 1 else "vj_sheet.jpg"; v = VJ(360, 640); tiles = []
    tex = None
    if len(sys.argv) > 2: tex = Image.open(sys.argv[2]).convert("RGBA")
    for name in SHADERS:
        t0 = time.time(); fr = [v.frame(name, 1.3 + k * .7, beat=(.9 if k == 1 else .1), energy=.7, seed=.37, tex=tex) for k in range(3)]
        print(name, f"{(time.time() - t0) / 3 * 1000:.0f} ms/frame"); tiles += fr
    sheet = Image.new("RGB", (360 * 6, 640 * 3))
    for i, im in enumerate(tiles): sheet.paste(im, ((i // 3) * 360, (i % 3) * 640))
    sheet.save(out, quality=85)
