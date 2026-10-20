#!/usr/bin/env python3
"""Nightly montages on free GitHub minutes: a track per style (yt-dlp: YouTube if it ever opens up, else SoundCloud / Bandcamp / archive.org)
+ the atlas library (your cut-outs, the images you kept, your lines) → a beat-cut edit with real datamosh transitions, CRT/xerox/rave post per family.
Results go to R2 (atlas/mont/<id>.mp4 + .jpg + atlas/mont.json); the curator folds them into the static bundle; the Edits tab plays them.
Your keep/kill on each edit weights which styles, recipes and tracks get more rolls.

  python montage.py --cap 6        make up to 6 edits
  python montage.py --probe        only test the audio sources from this runner
"""
import os, sys, io, re, json, time, math, random, hashlib, argparse, subprocess, traceback
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageChops, ImageEnhance

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "art"))
import collage as C, craft as K, glitch as G
_VJ = None
def vj():
    global _VJ
    if _VJ is None:
        import vj as V; _VJ = V.VJ(720, 1280)
    return _VJ
VJ_SHADERS = {"rave": ["boxes", "tunnel", "knot", "plasma", "rings", "kaleido", "kifs", "menger", "truchet", "opart"], "crt": ["tunnel", "rings", "plasma", "kaleido", "kifs", "truchet", "opart"], "dark": ["knot", "boxes", "tunnel", "kaleido", "kifs", "menger", "truchet"]}
TESS = ["hextile", "wallp"]                                             # tessellations built from a cut-out or symbol
def shader_frame(name, t, beat, energy, seed, tex=None, dim=1.0, fam="rave"):
    im = vj().frame(name, t, beat, energy, seed, tex).resize((W, H), Image.BILINEAR)
    if fam == "dark": im = ImageEnhance.Color(im).enhance(.45)
    return ImageEnhance.Brightness(im).enhance(dim) if dim != 1 else im
WORK = os.path.join(HERE, "_work", "montage"); ELS = os.path.join(WORK, "els"); AUD = os.path.join(WORK, "audio"); IMG = os.path.join(WORK, "img"); FR = os.path.join(WORK, "frames")
for d in (WORK, ELS, AUD, IMG, FR): os.makedirs(d, exist_ok=True)
C.ELS = ELS; G.OUT = WORK
SITE = os.environ.get("ATLAS_SITE", "https://atlas.symbiozefpv.com"); TOKEN = os.environ.get("ATLAS_TOKEN", "")
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
S = requests.Session(); S.mount("https://", HTTPAdapter(max_retries=Retry(total=6, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=frozenset(["GET", "PUT", "POST"]), raise_on_status=False)))
AUTH = {"authorization": "Bearer " + TOKEN}
report = {"started": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "probe": [], "errors": [], "made": [], "skipped": []}
CFG = json.load(open(os.path.join(HERE, "styles.json")))
MONT = json.load(open(os.path.join(HERE, "montage.json")))          # families, lines, music queries — edit freely

W, H, FPS = 1080, 1920, 24                                           # native 9:16 1080p
OUT_W, OUT_H = 1080, 1920
CAP_Y = .70                                                           # captions sit above TikTok's bottom UI band
DUR_MIN, DUR_MAX = 12.0, 17.0
KEEP_ITEMS = int(MONT.get("keep", 120)) # newest N unwatched stay on the site, plus every edit you kept or downloaded
WATCHED_TTL = 3600                                                  # seen, not kept, not downloaded → moved to the archive at the next batch after an hour

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def err(where, e): m = f"{where}: {type(e).__name__}: {str(e)[:200]}"; log("ERR", m); report["errors"].append(m)
def obj_get(key):
    r = S.get(SITE + "/api/atlas/cur/obj", params={"o": key}, headers=AUTH, timeout=300); return r.content if r.status_code == 200 else None
def obj_put(key, data, ctype="application/octet-stream"):
    r = S.put(SITE + "/api/atlas/cur/obj", params={"o": key}, data=data, headers={**AUTH, "content-type": ctype}, timeout=600); r.raise_for_status(); return r.json()
def obj_del(key):
    try: S.delete(SITE + "/api/atlas/cur/obj", params={"o": key}, headers=AUTH, timeout=60)
    except Exception: pass
def fam_of(style): return next((f for f, v in MONT["families"].items() if style in v["styles"]), "dark")

# ============================ audio sources ============================
def dl(target, out, extra=()):
    """yt-dlp any url / search expression → mp3 at out (a %(ext)s template). Returns meta or an error."""
    cmd = ["yt-dlp", "-q", "--no-warnings", "--no-playlist", "-f", "ba/b", "-x", "--audio-format", "mp3", "--audio-quality", "5", "--socket-timeout", "20", "--retries", "2",
           "--match-filter", "duration>40 & duration<720 & !is_live", "--print", "after_move:@@%(duration)s\t%(id)s\t%(webpage_url)s\t%(title)s", "-o", out, *extra, target, "--max-downloads", "1"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300); meta = [l[2:] for l in p.stdout.splitlines() if l.startswith("@@") and l.count("\t") >= 3]
        mp3 = out.replace("%(ext)s", "mp3")
        if meta and os.path.exists(mp3):
            d, i, u, t = meta[0].split("\t", 3)
            try: dv = float(d)
            except ValueError: dv = 0.0
            return {"ok": True, "title": t[:120], "dur": dv, "id": i, "url": u, "path": mp3, "bytes": os.path.getsize(mp3)}
        tail = (p.stderr or p.stdout).strip().splitlines(); return {"ok": False, "err": (tail[-1] if tail else f"rc={p.returncode}")[:220]}
    except subprocess.TimeoutExpired: return {"ok": False, "err": "timeout"}

def sc_candidates(q, n=8):
    """SoundCloud search → candidate list with the fields we score on (no download yet)"""
    p = subprocess.run(["yt-dlp", "-j", "--no-warnings", "--socket-timeout", "20", "--retries", "2", f"scsearch{n}:{q}"], capture_output=True, text=True, timeout=180); out = []
    for line in p.stdout.splitlines():
        try: j = json.loads(line)
        except Exception: continue
        out.append({"id": str(j.get("id")), "url": j.get("webpage_url") or j.get("original_url"), "title": j.get("title") or "", "dur": float(j.get("duration") or 0), "genre": (j.get("genre") or ""), "plays": int(j.get("view_count") or 0), "uploader": j.get("uploader") or "",
                    "age": max(1.0, (time.time() - float(j["timestamp"])) / 86400) if j.get("timestamp") else None})
    return out

STOP = {"type", "beat", "music", "the", "a", "of", "and", "mix", "set"}
def score_viral(c, q, want, cfg):
    """going viral now: plays per day since upload dominates; old or tiny tracks are out"""
    txt = " " + " ".join(re.findall(r"[a-z0-9]+", (c["title"] + " " + c["uploader"]).lower())) + " "
    artist = q in cfg.get("artists", []) and all(f" {w} " in txt for w in re.findall(r"[a-z0-9]+", q.lower().replace(" techno", "").replace(" x ", " ")))
    base = score_candidate(c, q, [] if artist else want, min_plays=int(cfg.get("min_plays", 0)))
    if base < 0 or c.get("age") is None or c["age"] > float(cfg.get("max_age_days", 540)): return -1
    return base - math.log10(1 + c["plays"]) * 1.1 + math.log10(1 + c["plays"]) * .5 + math.log10(1 + c["plays"] / c["age"]) * 1.6

def score_candidate(c, q, want, min_plays=None):
    """whole-word matches only (substring matching let 'd-beat' pick Initial D for punk); at least one genre word must hit"""
    txt = " " + " ".join(re.findall(r"[a-z0-9]+", (c["title"] + " " + c["genre"] + " " + c["uploader"]).lower())) + " "
    toks = [t for t in re.findall(r"[a-z0-9]+", q.lower()) if t not in STOP and len(t) > 2]
    hit = sum(1 for t in toks if f" {t} " in txt) / max(1, len(toks))
    gen = sum(1 for w in want if " " + " ".join(re.findall(r"[a-z0-9]+", w.lower())) + " " in txt)
    if not (45 <= c["dur"] <= 720): return -1
    if any(f" {x} " in txt for x in ("podcast", "episode", "interview", "tutorial", "asmr", "audiobook", "eurobeat", "nightcore", "anime", "cover", "covers", "goes pop", "karaoke", "reaction", "tutorial", "maroon", "8d", "ringtone")): return -1
    if want and gen == 0: return -1
    if any(b in txt for b in (" " + " ".join(re.findall(r"[a-z0-9]+", x.lower())) + " " for x in MONT.get("block", []))): return -1
    if min_plays is None: min_plays = int(MONT.get("music", {}).get("_techno", {}).get("min_plays", 0) if MONT.get("music_mode") == "techno" else 0)
    if c["plays"] < min_plays: return -1
    pop = math.log10(1 + c["plays"])                                     # popular inside the scene: plays dominate once it's on-genre
    return hit * 1.2 + gen * .8 + pop * 1.1 + (.5 if 150 <= c["dur"] <= 480 else 0)

def find_track(style, r, used):
    """the style's SoundCloud queries (scored picks) — YouTube tried once, first, only if it worked last run"""
    mq = MONT["music"]["_techno"] if MONT.get("music_mode") == "techno" else (MONT["music"].get(style) or MONT["music"]["_"]); tries = []; want = [w.lower() for w in mq.get("want", [])]
    for u in [p_ for p_ in MONT.get("picks", []) if p_ and p_.startswith("http")]:
        key = hashlib.sha1(u.encode()).hexdigest()[:10]
        if key in used: continue
        res = dl(u, os.path.join(AUD, f"pick_{key}.%(ext)s"))
        if res.get("ok"): res.update(src="pick", q=u, id=key); return res, tries
        tries.append(("pick", u, res.get("err", "?")))
    if MONT.get("_youtube_ok") and mq.get("yt"):
        q = r.choice(mq["yt"]); res = dl(f"ytsearch2:{q}", os.path.join(AUD, f"{style}_yt.%(ext)s"), ("--extractor-args", "youtube:player_client=tv"))
        if res.get("ok") and res["id"] not in used: res["src"] = "youtube"; res["q"] = q; return res, tries
        tries.append(("youtube", q, res.get("err", "used")))
    vir = MONT["music"].get("_viral") if MONT.get("music_mode") == "techno" else None
    passes = []
    if vir and r.random() < float(vir.get("share", .7)):                 # most edits: something going viral in the scene right now
        sc = lambda c, q: score_viral(c, q, want, vir); passes.append((r.sample(vir["sc"], min(5, len(vir["sc"]))), sc, 20))
    passes.append((r.sample(mq["sc"], min(5, len(mq["sc"]))), lambda c, q: score_candidate(c, q, want), 12))
    for qs, scf, n in passes:
        res, more = _sc_pass(style, r, used, qs, scf, n); tries += more
        if res: return res, tries
    return None, tries

def _sc_pass(style, r, used, qs, scf0, n):
    tries = []
    def scf(c, q):                                                      # your deletes/keeps on this uploader's earlier tracks
        b = scf0(c, q); u = UPTASTE.get((c.get("uploader") or "").lower(), 0)
        return -1 if b < 0 or u <= -2 else b + max(-1.5, min(1.0, .6 * u))
    for q in qs:
        try: cands = sc_candidates(q, n)
        except Exception as e: tries.append(("soundcloud", q, str(e)[:80])); continue
        cands = [c for c in cands if c["id"] not in used and c["url"] and int(hashlib.sha1(c["id"].encode()).hexdigest(), 16) % SHARD_OF == SHARD_NO]; cands.sort(key=lambda c: -scf(c, q))
        top = [c for c in cands[:6] if scf(c, q) >= 0]; r.shuffle(top); top.sort(key=lambda c: -scf(c, q) + r.uniform(0, .6))
        for c in top[:3]:
            if scf(c, q) < 0: break
            res = dl(c["url"], os.path.join(AUD, f"{style}_{c['id']}.%(ext)s"))
            if res.get("ok"): res.update(src="soundcloud", q=q, genre=c["genre"], plays=c["plays"], uploader=c["uploader"], per_day=int(c["plays"] / c["age"]) if c.get("age") else None); return res, tries
            tries.append(("soundcloud", c["url"], res.get("err", "?")))
        if not cands: tries.append(("soundcloud", q, "no candidates"))
    return None, tries

# ============================ audio analysis ============================
def pcm(path, sr=22050):
    p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768, sr

def onsets(x, sr, hop=512, n=1024):
    """spectral flux onset strength per hop + peak-picked onset times"""
    nf = max(1, (len(x) - n) // hop); win = np.hanning(n).astype(np.float32); prev = None; flux = np.zeros(nf, np.float32)
    for i in range(nf):
        mag = np.abs(np.fft.rfft(x[i * hop:i * hop + n] * win)); mag = np.log1p(mag * 30)
        if prev is not None: flux[i] = np.maximum(mag - prev, 0).sum()
        prev = mag
    flux = flux / (flux.max() + 1e-9); t = np.arange(nf) * hop / sr
    k = int(0.9 * sr / hop); loc = np.convolve(flux, np.ones(2 * k + 1) / (2 * k + 1), "same"); thr = loc + 0.08
    peaks = []; last = -1
    for i in range(1, nf - 1):
        if flux[i] > thr[i] and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1] and (last < 0 or t[i] - t[last] > 0.12): peaks.append(i); last = i
    rms = np.array([np.sqrt((x[i * hop:i * hop + n] ** 2).mean() + 1e-12) for i in range(nf)], np.float32)
    return t, flux, rms, peaks

def beats(x, sr, hop=512):
    """kick-band onset envelope → tempo (115-185 BPM, autocorrelation) → beat phase. Returns beat times (s), bpm."""
    n = 2048; nf = max(1, (len(x) - n) // hop); win = np.hanning(n).astype(np.float32); f = np.fft.rfftfreq(n, 1 / sr); lo = (f > 35) & (f < 160)
    env = np.zeros(nf, np.float32); prev = None
    for i in range(nf):
        m = np.log1p(np.abs(np.fft.rfft(x[i * hop:i * hop + n] * win))[lo] * 20)
        if prev is not None: env[i] = np.maximum(m - prev, 0).sum()
        prev = m
    env = env - np.convolve(env, np.ones(16) / 16, "same"); env = np.clip(env, 0, None); fps = sr / hop
    seg = env[: int(fps * min(len(env) / fps, 120))]
    ac = np.correlate(seg, seg, "full")[len(seg) - 1:]; best, bpm = -1, 135.0
    for b in np.arange(115, 186, 0.25):
        L = fps * 60 / b; s_ = sum(ac[int(round(L * k))] for k in (1, 2, 4) if int(round(L * k)) < len(ac))
        if s_ > best: best, bpm = s_, b
    period = fps * 60 / bpm; phases = np.arange(0, period, .5)
    score = [env[np.clip((ph + np.arange(0, len(env) - period, period)).astype(int), 0, len(env) - 1)].sum() for ph in phases]
    ph = phases[int(np.argmax(score))]; bt = (ph + np.arange(0, len(env) - 1, period)) / fps
    return bt, float(bpm), env, fps

def pick_drop(bt, kenv, kfps, t, rms, total, dur, r):
    """find a drop: the beat where the kick and loudness come back hard after a quieter stretch. The edit starts exactly
    8 bars (32 beats) or 4 bars (16 beats) before it, and ends 2 to 4 bars into it, because people stay for the drop."""
    bt = np.asarray(bt, np.float64)
    if len(bt) < 48: return None
    kt = np.arange(len(kenv)) / kfps
    def avg(xs, ys, a, b):
        m = (xs >= a) & (xs < b); return float(ys[m].mean()) if m.any() else 0.0
    Eb = np.array([.6 * avg(kt, kenv, bt[k], bt[k + 1]) + .4 * avg(t, rms, bt[k], bt[k + 1]) for k in range(len(bt) - 1)]); Eb /= Eb.max() + 1e-9
    cands = []
    for k in range(16, len(Eb) - 8):                                    # beat k: the 16 beats before vs the 8 after
        pre, post = float(Eb[k - 16:k].mean()), float(Eb[k:k + 8].mean())
        if post > .45 and post - pre > .15 and Eb[k] > Eb[k - 1] + .05: cands.append((post - pre, k))
    if not cands: return None
    cands.sort(reverse=True); best = []
    for sc, k in cands:                                                 # distinct drops only (not neighbouring beats of the same one)
        if all(abs(k - k2) > 8 for _, k2 in best): best.append((sc, k))
        if len(best) == 3: break
    best = [b_ for b_ in best if b_[0] >= best[0][0] * .8]              # only drops nearly as big as the biggest
    _, k = r.choice(best); beat = float(np.median(np.diff(bt)))
    for pre_beats in ([32, 16] if 32 * beat <= 15.5 else [16]):         # 32 beats of build-up when the tempo allows it
        post_beats = r.choice([8, 12, 16]) if pre_beats == 16 else r.choice([12, 16])
        if k - pre_beats < 0 or k + post_beats >= len(bt): continue
        start, end = float(bt[k - pre_beats]), float(bt[k + post_beats])
        if end > total - .5: continue
        return start, float(bt[k]) - start, end - start
    return None

def pick_section(t, flux, rms, peaks, total, dur, r):
    """loudest, busiest window well inside the track; start snapped to a strong onset"""
    hop_t = t[1] - t[0] if len(t) > 1 else .02; w = int(dur / hop_t); lo = int(0.12 * len(t)); hi = max(lo + 1, int(0.80 * len(t)) - w)
    if hi <= lo: return max(0.0, (total - dur) / 2)
    e = rms / (rms.max() + 1e-9) + 0.6 * np.convolve(flux, np.ones(int(0.5 / hop_t)) / int(0.5 / hop_t), "same")
    cs = np.concatenate([[0], np.cumsum(e)]); score = (cs[lo + w:hi + w] - cs[lo:hi]) / w
    i = lo + int(np.argmax(score * np.array([r.uniform(.9, 1.0) for _ in range(len(score))])))
    strong = [p for p in peaks if abs(p - i) < int(1.2 / hop_t)]
    if strong: i = max(strong, key=lambda p: flux[p])
    return float(t[i])

def grid_cuts(bt, start, dur, fam, r, strength_at):
    """cuts on the beat grid: first 2 bars every beat (hook), then mostly every 2 beats, some 4-beat holds,
    a stutter (half-beat double cut) before some bar lines. Bar lines = every 4th beat from the section start."""
    b = [x - start for x in bt if start - 1e-3 <= x <= start + dur - .25]
    if len(b) < 8: return None
    per = float(np.median(np.diff(b))); cuts = [0.0]; i = 0; hold = {"crt": [2, 2, 4], "dark": [2, 4, 4], "rave": [1, 2, 2, 2, 4]}.get(fam, [2])
    while i < len(b) - 1:
        bar_pos = i % 4
        step = 1 if i < 8 else r.choice(hold)
        if step > 1 and bar_pos + step > 4: step = 4 - bar_pos            # land on the bar line
        if i >= 8 and bar_pos == 3 and r.random() < .25 and fam == "rave": cuts.append(b[i] + per / 2)   # stutter into the bar
        i += max(1, step)
        if i < len(b): cuts.append(b[i])
    cuts = sorted(set(round(c, 4) for c in cuts if c < dur - .2))
    strengths = {round(c, 3): (1.0 if k % 4 == 0 else .5) * strength_at(c + start) for k, c in enumerate(b) for _ in [0]}
    bars = [round(c, 3) for k, c in enumerate(b) if k % 4 == 0]
    return cuts, strengths, b, bars, per

def cut_points(t, flux, peaks, start, dur, fam):
    """cut times (relative) on the strongest onsets, min/max gap by family; always a cut at 0"""
    gmin, gmax = MONT["families"][fam]["gap"]; cand = [(float(flux[p]), float(t[p]) - start) for p in peaks if start + 0.1 <= t[p] <= start + dur - 0.5]
    cand.sort(reverse=True); cuts = [0.0]
    for s, ct in cand:
        g = gmin * (.55 if ct < 3.0 else 1)                               # hook: the first 3 s cut harder
        if all(abs(ct - c) >= g for c in cuts): cuts.append(ct)
    cuts.sort(); out = [0.0]
    for c in cuts[1:]:
        while c - out[-1] > gmax * 1.6: out.append(out[-1] + gmax)   # never a dead stretch
        out.append(c)
    while dur - out[-1] > gmax * 1.6: out.append(out[-1] + gmax)
    strengths = {}
    for s, ct in cand: strengths[round(ct, 3)] = s
    return out, strengths

# ============================ library ============================
def sync_library():
    els = json.loads(obj_get("atlas/elements.json") or '{"items":[]}')["items"]; man = json.loads(obj_get("atlas/manifest.json") or '{"styles":{}}')
    flags = json.loads(obj_get("atlas/flags.json") or "{}"); mont = json.loads(obj_get("atlas/mont.json") or '{"items":[]}')
    n = 0
    for e in els:
        p = os.path.join(ELS, e["style"], f"{e.get('kind') or 'thing'}-{e['id']}.png")
        if os.path.exists(p): continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        try:
            rr = S.get(SITE + "/api/atlas/el/" + e["id"], timeout=60)
            if rr.status_code == 200 and rr.content[:4] == b"\x89PNG": open(p, "wb").write(rr.content); n += 1
        except Exception as ex: err("el " + e["id"], ex)
    live = {f"{e.get('kind') or 'thing'}-{e['id']}.png" for e in els}
    for st in os.listdir(ELS):
        for f in os.listdir(os.path.join(ELS, st)):
            if f not in live: os.remove(os.path.join(ELS, st, f))
    log(f"library: {len(els)} elements ({n} new), {sum(len(v) for v in man['styles'].values())} images, {len(mont['items'])} edits so far")
    return els, man, flags, mont

def kept_images(man, style):
    its = man["styles"].get(style, []); ok = [i for i in its if i.get("ok")]; return ok or sorted(its, key=lambda i: -(i.get("cs") or 0))[:40]

def fetch_image(it):
    p = os.path.join(IMG, it["id"] + ".jpg")
    if not os.path.exists(p):
        rr = S.get(SITE + it["url"], timeout=60)
        if rr.status_code != 200: return None
        im = Image.open(io.BytesIO(rr.content)).convert("RGB"); im.thumbnail((1400, 1400)); im.save(p, "JPEG", quality=90)
    return Image.open(p).convert("RGB")

def weights(mont, flags, key):
    """keeps − deletes for a style / family / source: from the live edits plus the taste ledger (survives pruning)"""
    ok, dead = set(flags.get("mok", [])) | set(flags.get("mdl", [])), set(flags.get("mdead", [])); w = {}; seen = set()
    for it in mont["items"]:
        seen.add(it["id"]); k = it.get(key); w[k] = w.get(k, 0) + (1 if it["id"] in ok else 0) - (.7 if it["id"] in dead else 0)
    for t in mont.get("taste", []):
        if t["id"] in seen or (t["v"] < 0 and t.get("why") in ("song", "none")): continue
        k = t.get(key); w[k] = w.get(k, 0) + (1 if t["v"] > 0 else -.7)
    return w

UPTASTE = {}
def uploader_taste(mont):
    """per track uploader: +1 per kept/downloaded edit, −1 per deleted one"""
    d = {}
    for t in mont.get("taste", []):
        u = (t.get("uploader") or "").lower()
        if t["v"] < 0 and t.get("why") in ("video", "none"): continue
        if u: d[u] = d.get(u, 0) + t["v"]
    return d

# ============================ rendering ============================
def cover(im, w=W, h=H, zoom=1.0, px=0.5, py=0.5):
    """fill w×h from im, zoomed, anchored at (px,py) of the source"""
    r = max(w / im.width, h / im.height) * zoom; sw, sh = int(im.width * r), int(im.height * r); big = im.resize((sw, sh), Image.BILINEAR)
    x = int((sw - w) * px); y = int((sh - h) * py); return big.crop((x, y, x + w, y + h))

def grade(img, fam, r):
    """bold: saturated, high contrast; dark family stays moody but keeps colour"""
    if fam == "crt":
        rgb = ImageEnhance.Color(img).enhance(1.1); arr = np.asarray(rgb).astype(np.float32); arr[..., 2] *= 1.12; return Image.fromarray(np.clip((arr - 20) * 1.25, 0, 255).astype(np.uint8))
    if fam == "dark":
        rgb = ImageEnhance.Color(img).enhance(.9); arr = np.asarray(rgb).astype(np.float32); arr[..., 0] *= 1.08; arr[..., 2] *= 1.12; return Image.fromarray(np.clip((arr - 28) * 1.3, 0, 255).astype(np.uint8))
    rgb = ImageEnhance.Color(img).enhance(1.7); return ImageEnhance.Contrast(rgb).enhance(1.3)

def post(img, fam, t, seed, strobe=0.0):
    """per-family finish; grain kept low on purpose — film grain is what makes a 15 s clip 30 MB"""
    if fam == "crt": img = C.chroma(img, 3 + int(2 * abs(math.sin(t * 3)))); img = K.bloom(img, 205, 14, .42); img = ImageEnhance.Contrast(img).enhance(1.2); img = K.phosphor(img, 3, .35); img = C.scanlines(img, 4, .14)
    elif fam == "dark": img = C.chroma(img, 3); img = K.bloom(img, 210, 20, .35); img = ImageEnhance.Contrast(img).enhance(1.15)
    else: img = C.chroma(img, 4 + int(5 * strobe)); img = K.bloom(img, 190, 18, .5); img = ImageEnhance.Contrast(img).enhance(1.12)
    if strobe > 0: img = ImageEnhance.Brightness(img).enhance(1 + 1.1 * strobe)
    img = C.grain(img, 6, seed + int(t * 100))
    return C.vignette(img, .45)

def figure_sprite(path, fam, r):
    f = C.load(path, 1400); f = C.fit(f, h=int(H * r.uniform(.6, .92)))
    rgb = grade(f.convert("RGB"), fam, r).convert("RGBA"); rgb.putalpha(C.alpha_of(f)); return rgb

PALS = [((255, 0, 110), (0, 240, 255)), ((255, 230, 0), (120, 0, 255)), ((0, 255, 120), (255, 0, 60)), ((255, 90, 0), (0, 40, 255)), ((255, 255, 255), (255, 0, 0)), ((0, 255, 255), (255, 0, 255))]
def backdrop(fam, style, s, wall, t=0.0, beat=0.0, energy=.5, seed=0.0):
    """s["bg"] is decided once per shot (wall / plain / shader), so the background doesn't flicker frame to frame"""
    bg = s.get("bg", "plain")
    if bg == "shader": return shader_frame(s["shader"], t, beat, energy, seed, dim=(.8 if fam == "rave" else .6), fam=fam).convert("RGBA")
    if bg == "wall" and wall is not None: b = cover(wall, zoom=1.15, px=s["bpx"], py=s["bpy"]).filter(ImageFilter.GaussianBlur(5)); b = ImageEnhance.Brightness(b).enhance(.5 if fam != "rave" else .7); return b.convert("RGBA")
    a, b2 = s.get("pal", PALS[0]); base = Image.new("RGB", (W, H), tuple(int(c * .12) for c in b2)); gl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(gl)
    d.ellipse([W * .1, H * .15, W * .9, H * .85], fill=tuple(int(c * .55) for c in a) + (255,)); gl = gl.filter(ImageFilter.GaussianBlur(100)); base.paste(gl, (0, 0), gl); return base.convert("RGBA")

def caption_sprite(line, fam, big=False):
    font = {"crt": "vt323", "dark": "unifrakturcook"}.get(fam, "anton")
    return K.tx(line if fam in ("crt", "dark") else line.upper(), 104 if big else 46, K.FONT[font], (255, 255, 255, 255), 2)

# ---------------- frame treatments (applied per shot, part of the motif) ----------------
_ASCII = {}
def ascii_fx(img, cell=10, pal=None):
    """real ASCII: every cell becomes a glyph picked by brightness, tinted with the cell's colour (or a 2-colour palette)"""
    ch = int(cell * 1.6)
    if cell not in _ASCII:
        from PIL import ImageFont
        chars = " .'`:-=+*%#@"; f = ImageFont.truetype(K.FONT["vt323"], int(ch * 1.15)); gl = []
        for c in chars:
            m = Image.new("L", (cell, ch), 0); ImageDraw.Draw(m).text((cell / 2, ch / 2), c, font=f, fill=255, anchor="mm"); gl.append(np.asarray(m, np.float32) / 255)
        _ASCII[cell] = np.stack(gl)
    G_ = _ASCII[cell]; a = np.asarray(img, np.float32); rows, cols = H // ch, W // cell
    blk = a[:rows * ch, :cols * cell].reshape(rows, ch, cols, cell, 3).mean(axis=(1, 3)); lum = blk.mean(-1); lo, hi = np.percentile(lum, 3), np.percentile(lum, 97); lum = np.clip((lum - lo) / max(1, hi - lo), 0, 1)
    idx = np.clip((lum ** .8) * (len(G_) - 1), 0, len(G_) - 1).astype(int); masks = G_[idx]                       # rows, cols, ch, cell
    pal = pal or ((255, 255, 255), (0, 255, 160)); c0, c1 = np.array(pal[0], np.float32), np.array(pal[1], np.float32); col = c1[None, None] * (1 - lum[..., None]) + c0[None, None] * lum[..., None]
    out = masks[..., None] * col[:, :, None, None, :]; out = out.transpose(0, 2, 1, 3, 4).reshape(rows * ch, cols * cell, 3)
    canvas = np.zeros((H, W, 3), np.float32); canvas[:rows * ch, :cols * cell] = out; return Image.fromarray(canvas.astype(np.uint8))

def apply_fx(img, fx, s, t, beat):
    if "mirror2" in fx: a = np.asarray(img); h = a.shape[1] // 2; a = a.copy(); a[:, h:] = a[:, :h][:, ::-1][:, :a.shape[1] - h]; img = Image.fromarray(a)
    if "mirror4" in fx: a = np.asarray(img).copy(); hh, hw = a.shape[0] // 2, a.shape[1] // 2; q = a[:hh, :hw]; a[:hh, a.shape[1] - hw:] = q[:, ::-1]; a[a.shape[0] - hh:, :] = a[:hh][::-1]; img = Image.fromarray(a)
    if "clip" in fx:                                                      # hard clip: crushed, posterised, saturated
        a = np.asarray(img, np.float32); a = np.clip((a - 90) * 2.4 + 90, 0, 255); a = (a // 64) * 64 + 32; img = ImageEnhance.Color(Image.fromarray(a.astype(np.uint8))).enhance(1.8)
    if "duo" in fx:
        c0, c1 = s["pal"]; l = np.asarray(img.convert("L"), np.float32)[..., None] / 255; l = np.clip((l - .5) * 1.8 + .5, 0, 1)
        img = Image.fromarray((np.array(c1) * (1 - l) + np.array(c0) * l).astype(np.uint8))
    if "invert" in fx: img = ImageOps.invert(img)
    if "ascii" in fx: img = ascii_fx(img, s.get("cell", 10), s["pal"] if s.get("apal") else None)
    if "rgbshift" in fx: img = C.chroma(img, int(6 + 18 * beat))
    return img

# ---------------- edit grammar: motifs, repetition, build-up, surprise ----------------
FX_POOL = ["mirror2", "mirror4", "clip", "duo", "ascii", "rgbshift", "invert"]
def motif(r, fam, figs, walls, name):
    share = float(MONT.get("vj_share", {}).get(fam, .4))
    typ = r.choices(["vj", "figure", "wall", "tile", "echo", "burst"], [share * 6, 2.2, 1.4 if walls else 0, 1.2, 1.2, .8])[0]
    m = {"name": name, "type": typ, "fig": r.choice(figs), "wall": r.choice(walls) if walls else None, "extra": r.sample(figs, min(4, len(figs))),
         "shader": r.choice(VJ_SHADERS.get(fam) or ["plasma"]), "solo": r.random() < .3, "sseed": r.random(), "pal": r.choice(PALS),
         "fx": r.sample(FX_POOL, r.choice([0, 1, 1, 2])), "cell": r.choice([8, 10, 12]), "apal": r.random() < .5, "tiles": r.choice([2, 3, 4]),
         "zoom": r.uniform(1.0, 1.08), "dz": r.uniform(-.05, .1), "px": r.random(), "py": r.random(), "dx": r.uniform(-.1, .1), "dy": r.uniform(-.07, .07)}
    p_shader = {"rave": .7, "crt": .35, "dark": .45}.get(fam, .4)
    m["bg"] = "shader" if r.random() < p_shader else ("wall" if r.random() < .5 else "plain")
    if m["bg"] == "shader" and m["type"] != "vj": m["shader"] = r.choice([x for x in VJ_SHADERS.get(fam, []) if x != "kaleido"] or ["plasma"])
    return m

def variant(m, r, k):
    """the same motif, recognisably — mirrored, palette-flipped, re-framed — so repetition stays present but alive"""
    v = dict(m); v["flip"] = (k % 2 == 1); v["px"] = (m["px"] + .23 * k) % 1; v["dx"] = m["dx"] * (-1 if k % 2 else 1)
    if k % 3 == 2: v["pal"] = (m["pal"][1], m["pal"][0])
    if k % 4 == 3 and len(v["fx"]) < 3: v["fx"] = v["fx"] + [r.choice([f for f in FX_POOL if f not in v["fx"]])]
    return v

def make_shots(cuts, dur, fam, style, figs, walls, lines, r, bars=(), per=0.45):
    """Phrases of 4 bars. Each phrase sets a short pattern of motifs (e.g. A B / A A B / A B C B) and repeats it with variations.
    The last bar of a phrase builds (the grid already cuts faster there). The next downbeat resolves the expectation:
      payoff  — back to the hero motif, bigger, with an added treatment, moshed in from the build-up
      surprise — a motif never seen before, or a hard treatment (ascii / invert / mirror4), or a fake-out: one beat of black then the payoff
    Roughly half the phrase boundaries pay off and half surprise, so the viewer learns the pattern and then gets it broken."""
    letters = "ABCDEFG"; M = {"A": motif(r, fam, figs, walls, "A"), "B": motif(r, fam, figs, walls, "B"), "C": motif(r, fam, figs, walls, "C")}
    patterns = [["A", "B"], ["A", "A", "B"], ["A", "B", "A", "C"], ["A", "B", "C", "B"], ["A", "A", "B", "B"]]
    phrase_len = 4 * 4 * per if per else 7.0; bar_len = 4 * per if per else 1.8
    shots = []; pat = r.choice(patterns); pos = 0; seen = {}; cur_phrase = -1; text_left = 1 if lines and r.random() < float(MONT.get("text", {}).get("card_chance", .35)) else 0
    for i, c in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else dur; ph = int((c + 1e-3) // phrase_len); in_bar = int(((c + 1e-3) % phrase_len) // bar_len)
        boundary = ph != cur_phrase and i > 0; cur_phrase = ph; event = None
        if boundary:
            roll = r.random()
            if roll < .45: event = "payoff"
            elif roll < .7: event = "new"
            elif roll < .87: event = "hardfx"
            else: event = "fakeout"
            if r.random() < .5: pat = r.choice(patterns)                    # sometimes the next phrase changes the pattern too
            pos = 0
        if event == "new":
            L = letters[len(M)] if len(M) < len(letters) else r.choice(list(M)); M[L] = motif(r, fam, figs, walls, L); key = L
        elif event == "payoff" or event == "fakeout": key = "A"
        else: key = pat[pos % len(pat)]; pos += 1
        seen[key] = seen.get(key, 0) + 1; m = variant(M[key], r, seen[key] - 1)
        s = dict(m); s.update(t0=c, t1=end, event=event, mosh=False)
        if event == "payoff": s["zoom"] = m["zoom"] * 1.12; s["fx"] = list(dict.fromkeys(m["fx"] + [r.choice(["rgbshift", "clip", "mirror2"])])); s["mosh"] = True
        if event == "hardfx": s["fx"] = [r.choice(["ascii", "invert", "mirror4", "duo"])] + m["fx"][:1]
        if event == "fakeout" and end - c > per * .9:                      # one beat of nothing, then the payoff
            blk = dict(s); blk.update(type="black", t1=c + per, fx=[]); shots.append(blk); s["t0"] = c + per; s["mosh"] = False
        if in_bar == 3 and not boundary and r.random() < .35: s["mosh"] = True   # the build-up bleeds
        if text_left and i > len(cuts) // 3 and in_bar == 1 and r.random() < .3: s["type"] = "text"; s["line"] = r.choice(lines); text_left -= 1
        s["bpx"], s["bpy"] = r.random(), r.random(); shots.append(s)
    return shots

def render(shots, dur, fam, style, cutset, strengths, seed, r, lines, punches=(), hook=None, env=((), ()), bars=(), keyframes=()):
    env_t, env_v = env
    avi = os.path.join(WORK, "clean.avi"); kf = ",".join(f"{c:.3f}" for c in sorted(set(keyframes) | {0.0}))
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libxvid", "-q:v", "4", "-g", "9999", "-bf", "0", "-force_key_frames", kf, avi], stdin=subprocess.PIPE)
    def emit(im): enc.stdin.write(im.convert("RGB").tobytes())
    io_ = MONT.get("intro", {}); intro = None; outro = None
    if io_ and r.random() < float(io_.get("chance", 0)):
        lines_ = r.choice(io_["lines"]); end_ = float(io_.get("secs", 2.4))
        bl = [b for b in bars if b >= end_ - .3]; end_ = bl[0] if bl else end_          # the opener clears on a bar line
        intro = ([x for x in lines_ if x], end_)
    oc = MONT.get("outro", {}); outro = None
    if oc and not intro and r.random() < float(oc.get("chance", 0)):
        bl = [b for b in bars if b <= dur - 1.6]; outro = (r.choice(oc["lines"]), bl[-1] if bl else dur - 2.2)
    d = os.path.join(FR, f"m{seed}"); os.makedirs(d, exist_ok=True)
    for f in os.listdir(d): os.remove(os.path.join(d, f))
    nfr = int(round(dur * FPS)); sprites = {}; walls = {}; barset = list(bars)
    def spr(path):
        if path not in sprites: sprites[path] = figure_sprite(path, fam, r)
        return sprites[path]
    def wall_img(it):
        if it is None: return None
        if it["id"] not in walls:
            im = fetch_image(it); walls[it["id"]] = grade(im, fam, r) if im is not None else None
        return walls[it["id"]]
    # flash words: rare, on bar downbeats only
    fl = MONT.get("abstract", {}).get("flash", []); flashes = {}; fc = float(MONT.get("text", {}).get("flash_chance", .12))
    for b in barset:
        if b > 1.0 and fl and r.random() < fc:
            w_ = r.choice(fl); f0 = int(round(b * FPS))
            for dd in range(2): flashes[f0 + dd] = w_
    si = 0
    for k in range(nfr):
        t = k / FPS
        while si + 1 < len(shots) and t >= shots[si + 1]["t0"] - 1e-6: si += 1
        s = shots[si]; u = (t - s["t0"]) / max(1e-3, s["t1"] - s["t0"]); z = s["zoom"] + s["dz"] * u
        last = max((p for p in punches if p <= t + 1e-6), default=-9); bar_last = max((b for b in barset if b <= t + 1e-6), default=-9)
        z *= 1 + .07 * math.exp(-(t - last) * 10) + .06 * math.exp(-(t - bar_last) * 6)       # punch every beat, bigger on the bar
        beat = math.exp(-(t - last) * 7); energy = float(np.interp(t, env_t, env_v)) if len(env_t) else .5
        wl = wall_img(s.get("wall")); typ = s["type"]
        if typ == "black":
            emit(Image.new("RGB", (W, H), (0, 0, 0))); continue
        fg = spr(s["fig"])
        if typ == "vj":
            if s["shader"] == "kaleido": fr = shader_frame("kaleido", t, beat, energy, s["sseed"], tex=fg, fam=fam).convert("RGBA")
            else:
                fr = shader_frame(s["shader"], t, beat, energy, s["sseed"], fam=fam).convert("RGBA")
                if not s["solo"]: f = fg.resize((max(1, int(fg.width * z * .85)), max(1, int(fg.height * z * .85))), Image.BILINEAR); C.paste(fr, f, W / 2, H * .55)
        elif typ == "wall" and wl is not None:
            fr = cover(wl, zoom=z, px=min(1, max(0, s["px"] + s["dx"] * u)), py=min(1, max(0, s["py"] + s["dy"] * u)))
            if s.get("flip"): fr = ImageOps.mirror(fr)
            fr = fr.convert("RGBA")
        elif typ == "text":
            fr = backdrop(fam, style, s, wl, t, beat, energy, s["sseed"]); big = caption_sprite(s["line"], fam, big=True); big = C.fit(big, w=min(big.width, int(W * .8))); C.paste(fr, big, W / 2, H / 2)
        elif typ == "tile":                                               # step-and-repeat grid, rows offset on the beat
            fr = backdrop(fam, style, s, wl, t, beat, energy, s["sseed"]); n = s["tiles"]; e = C.fit(fg, w=int(W / n * .9))
            for gy in range(int(H / e.height) + 2):
                for gx in range(n + 1):
                    off = (gy % 2) * (W / n / 2) * (1 if int(t / max(per_of(punches), .2)) % 2 else -1)
                    C.paste(fr, e if (gx + gy) % 2 == 0 or not s.get("flip") else ImageOps.mirror(e), gx * W / n + W / n / 2 + off - W / n / 2 * (n % 2 == 0), gy * e.height + e.height / 2 - (u * e.height) % e.height)
        elif typ == "echo":                                               # the element and its trail, decaying copies
            fr = backdrop(fam, style, s, wl, t, beat, energy, s["sseed"]); f = fg.resize((max(1, int(fg.width * z)), max(1, int(fg.height * z))), Image.BILINEAR)
            for j in range(5, 0, -1): C.paste(fr, f, W / 2 + s["dx"] * W * u - j * 26 * math.cos(t * 2), H * .54 + j * 34 * math.sin(t * 2 + j), alpha=.18 + .1 * (5 - j))
            C.paste(fr, f, W / 2 + s["dx"] * W * u, H * .54)
        else:
            fr = backdrop(fam, style, s, wl, t, beat, energy, s["sseed"]); f = fg.resize((max(1, int(fg.width * z)), max(1, int(fg.height * z))), Image.BILINEAR)
            if s.get("flip"): f = ImageOps.mirror(f)
            C.paste(fr, f, W / 2 + s["dx"] * W * u, H * .54 + s["dy"] * H * u)
            if typ == "burst":
                for j, ex in enumerate(s["extra"]):
                    e = spr(ex); e = C.fit(e, w=int(W * .3)); ang = (j * 2.1 + u * 2.5); C.paste(fr, e, W / 2 + math.cos(ang) * W * (.25 + .3 * u), H / 2 + math.sin(ang) * H * (.2 + .3 * u), rot=u * 40 * (1 if j % 2 else -1), alpha=max(0, 1 - u))
        img = apply_fx(fr.convert("RGB"), s.get("fx", []), s, t, beat)
        near_bar = any(abs(b - t) < 1.2 / FPS for b in barset); strobe = .9 if near_bar and fam == "rave" else (.4 if near_bar else 0.0)
        img = post(img, fam, t, seed, strobe)
        w_ = flashes.get(k)
        if w_:
            b = Image.new("RGBA", (W, H), (0, 0, 0, 255)); sp = K.tx(w_, 300, K.FONT[{"crt": "vt323", "dark": "unifrakturcook"}.get(fam, "anton")], (255, 255, 255, 255), 4); sp = C.fit(sp, w=int(W * .86))
            C.paste(b, sp, W / 2, H / 2); img = C.chroma(b.convert("RGB"), 6)
        if intro and t < intro[1]:                                         # series opener: TikTok-native caption, top third, outlined
            b = img.convert("RGBA"); a_ = 1.0 if t < intro[1] - .25 else max(0, (intro[1] - t) / .25); y = H * .26
            for li, line in enumerate(intro[0]):
                sp = K.tx(line, 46, K.FONT["oswald"], (255, 255, 255, int(255 * a_)), 0); sp = C.fit(sp, w=min(sp.width, int(W * .84)))
                ol = K.tx(line, 46, K.FONT["oswald"], (0, 0, 0, int(230 * a_)), 0); ol = C.fit(ol, w=sp.width)
                for ox, oy in ((-3, 0), (3, 0), (0, -3), (0, 3), (-2, -2), (2, 2), (-2, 2), (2, -2)): C.paste(b, ol, W / 2 + ox, y + oy)
                C.paste(b, sp, W / 2, y); y += sp.height * 1.35
            img = b.convert("RGB")
        if outro and t >= outro[1]:
            b = img.convert("RGBA"); y = H * .26
            sp = K.tx(outro[0], 46, K.FONT["oswald"], (255, 255, 255, 255), 0); sp = C.fit(sp, w=min(sp.width, int(W * .84))); ol = K.tx(outro[0], 46, K.FONT["oswald"], (0, 0, 0, 230), 0); ol = C.fit(ol, w=sp.width)
            for ox, oy in ((-3, 0), (3, 0), (0, -3), (0, 3), (-2, -2), (2, 2), (-2, 2), (2, -2)): C.paste(b, ol, W / 2 + ox, y + oy)
            C.paste(b, sp, W / 2, y); img = b.convert("RGB")
        emit(img)
    enc.stdin.close(); enc.wait()
    if enc.returncode: raise RuntimeError("frame encoder failed")
    return d, nfr

def per_of(punches):
    p = [b - a for a, b in zip(punches, punches[1:]) if b > a]; return float(np.median(p)) if p else .45

# ============================ v2: hero symbols over trippy backgrounds, bar-locked sections ============================
import symbols as SYM
HERO_PALS = [((255, 240, 0), (255, 0, 160)), ((0, 255, 200), (120, 0, 255)), ((255, 255, 255), (255, 30, 30)), ((255, 120, 0), (0, 200, 255)),
             ((180, 255, 0), (255, 0, 90)), ((255, 0, 200), (0, 255, 255)), ((255, 255, 255), (0, 120, 255))]

def hero_sprite(h, size, pal, fam, r):
    """h = ("sym", name) or ("cut", path) → RGBA, crisp, glowing"""
    fg, glow = pal
    if h[0] == "sym":
        return SYM.symbol(h[1], size, fg + (255,), (10, 0, 20, 255), font=K.FONT["anton"], glow=glow)
    im = C.load(h[1], 1800); im = C.fit(im, w=size) if im.width >= im.height else C.fit(im, h=size)
    rgb = ImageEnhance.Contrast(ImageEnhance.Color(im.convert("RGB")).enhance(1.6)).enhance(1.25).convert("RGBA"); rgb.putalpha(C.alpha_of(im))
    a = C.alpha_of(im).filter(ImageFilter.GaussianBlur(size * .025)); halo = Image.new("RGBA", im.size, glow + (0,)); halo.putalpha(a.point(lambda v: min(255, int(v * 1.5))))
    out = Image.new("RGBA", im.size, (0, 0, 0, 0)); out.alpha_composite(halo); out.alpha_composite(rgb); return out

def pick_heroes(r, style, fam, mont, n):
    """vector symbols the scene recognises + the library's own logos/symbols/figures; avoid what recent edits used"""
    recent = set(sum((it.get("heroes", []) for it in mont["items"][:12]), []))
    pref = MONT.get("symbols", {}).get(style) or MONT.get("symbols", {}).get("_", SYM.NAMES)
    syms = [("sym", s_) for s_ in pref if "sym:" + s_ not in recent] or [("sym", s_) for s_ in pref]
    others = [("sym", s_) for s_ in SYM.NAMES if s_ not in pref and "sym:" + s_ not in recent]
    cuts = [f for f in C.pool(style, ["logo", "symbol", "person", "face", "speaker", "vinyl", "vehicle"]) if _big_enough(f, 260 * 260) and "cut:" + os.path.basename(f) not in recent]
    cuts += [f for f in sum((C.pool(s2, ["logo", "symbol"]) for s2 in MONT["families"][fam]["styles"] if s2 != style), []) if _big_enough(f, 260 * 260) and "cut:" + os.path.basename(f) not in recent]
    r.shuffle(syms); r.shuffle(others); r.shuffle(cuts)
    out = []; ns = max(2, round(n * .6))
    out += syms[:ns]; out += [("cut", f) for f in cuts[: n - len(out)]]
    if len(out) < n: out += others[: n - len(out)]
    if len(out) < n: out += [("sym", s_) for s_ in r.sample(SYM.NAMES, n - len(out))]
    r.shuffle(out); return out[:n]

def make_sections(bars, per, dur, fam, style, walls, r, mont):
    """Sections of 1 to 3 bars (never a fixed rhythm), one hero each: first bar clean, the rest varied. Build-ups (a cut per beat,
    zoom climbing) arrive at irregular points and always land on a drop: a callback to an earlier hero in a new treatment.
    Shots change by hard cut, datamosh, liquid melt or zoom-through; backgrounds range from shaders and tessellations to a
    landscape and a lattice of 3D shapes built from a tapestry of images. Every cut stays on the beat grid. Returns shots with t0/t1."""
    if len(bars) < 2: bars = [i * 4 * per for i in range(int(dur / (4 * per)) + 1)]
    bar_len = 4 * per; nb = int(round(dur / bar_len))
    heroes = pick_heroes(r, style, fam, mont, max(3, nb // 2 + 1)); shots = []; used = []
    class Deck:
        def __init__(self, items): self.items = list(items); self.pool = []
        def draw(self):
            if not self.pool: self.pool = self.items[:]; r.shuffle(self.pool)
            return self.pool.pop()
    moves = Deck(["center", "coinspin", "orbit", "droste", "slices", "split4", "strobe2", "zoomthrough", "triple", "pixelsort", "shatter", "scan"])
    variants = Deck([["echo"], ["tile"], ["mirror2"], ["duo"], ["rgbshift"], ["invert_beats"], ["mirror4"], ["ascii_bg"], ["posterize"], ["liquid"], ["liquid", "rgbshift"]])
    drops = Deck([["ascii"], ["mirror4"], ["kaleido_hero"], ["ascii", "rgbshift"], ["invert", "mirror2"], ["ascii_bg"], ["kaleido_hero", "duo"], ["posterize", "mirror4"], ["tess_hero"], ["tess_hero", "rgbshift"], ["liquid", "mirror4"]])
    shaders = [x for x in VJ_SHADERS.get(fam, VJ_SHADERS["rave"]) if x != "kaleido"]
    bgs = Deck(["shader", "shader", "tess", "tess", "land", "poly", "kwall", "wall"] if walls else ["shader", "shader", "tess", "tess", "land", "poly"])
    def bg_choice():
        k = bgs.draw()
        if k == "tess": return {"bg": "tess", "shader": r.choice(TESS), "tsrc": r.choice(heroes), "sseed": r.random()}
        if k == "land": return {"bg": "mosaic", "shader": "tapestry", "sseed": r.random()}
        if k == "poly": return {"bg": "mosaic", "shader": "polyfield", "sseed": r.random()}
        if k == "kwall": return {"bg": "kwall", "wall": r.choice(walls), "sseed": r.random()}
        if k == "wall": return {"bg": "wall", "wall": r.choice(walls), "bpx": r.random(), "bpy": r.random()}
        return {"bg": "shader", "shader": r.choice(shaders), "sseed": r.random()}
    def trans(after_drop=False):
        return r.choices(["cut", "mosh", "melt", "zoom", "static"], weights=[3, 3, 3, 1.5, 1] if not after_drop else [1, 4, 3, 2, 1])[0]
    lk = {"look": r.choices(list(LOOKS), list(LOOKS.values()))[0], "raw": None, "acc": r.choice(ACCENTS)}
    def next_look():
        if r.random() < .3: return dict(lk)                                   # sometimes the look carries over, so it doesn't strobe between styles
        lk.update(look=r.choices(list(LOOKS), list(LOOKS.values()))[0], raw=r.choices(list(RAWS), list(RAWS.values()))[0], acc=r.choice(ACCENTS)); return dict(lk)
    b = 0; sec = 0; since_drop = 0
    while b < nb:
        if sec >= len(heroes): heroes.append(r.choice(heroes))
        hero = heroes[sec]; pal = HERO_PALS[(sec * 3 + (hash(style) % 7) + r.randrange(3)) % len(HERO_PALS)]
        bgc = bg_choice(); move = moves.draw(); other = r.choice([h_ for h_ in heroes if h_ != hero] or heroes)
        L = min(nb - b, r.choices([1, 2, 3], weights=[2, 5, 2])[0])
        base = {"hero": hero, "other": other, "pal": pal, "move": move, "fx": [], "zoom": 1.0, "rot": r.uniform(-10, 10), "drift": r.uniform(-.05, .05), **bgc, **next_look()}
        for k in range(L):
            t0 = (b + k) * bar_len; t1 = min(dur, t0 + bar_len)
            if k == 0:
                shots.append({**base, "t0": t0, "t1": t1, "tr": trans(since_drop == 0 and b > 0)}); used.append(hero)
            else:
                half = r.random() < .22                                      # sometimes the variant lands on the half bar instead
                if half and t1 - t0 > per * 2:
                    shots.append({**base, "t0": t0, "t1": t0 + 2 * per, "fx": variants.draw(), "zoom": 1.05, "tr": "cut"})
                    shots.append({**base, "t0": t0 + 2 * per, "t1": t1, "fx": variants.draw(), "zoom": 1.1, "tr": r.choice(["cut", "melt"])})
                else:
                    shots.append({**base, "t0": t0, "t1": t1, "fx": variants.draw(), "zoom": 1.06, "tr": r.choice(["cut", "cut", "melt", "mosh"])})
        b += L; sec += 1; since_drop += L
        # build-up then drop at irregular points: more likely the longer it has been
        if b + 2 <= nb and r.random() < min(.85, .18 * since_drop):
            t0 = b * bar_len; t1 = min(dur, t0 + bar_len); beats_in = r.choice([4, 4, 8]) if per < .42 else 4
            step = bar_len / beats_in
            for k in range(beats_in):
                bt0 = t0 + k * step; alt = k % 2 == 1
                bu = r.choice(["kaleido_hero", "ascii", "invert", "mirror4", "posterize", "tess_hero", "liquid"])
                shots.append({**base, "t0": bt0, "t1": min(t1, bt0 + step), "move": "center" if not alt else r.choice(["strobe2", "triple", "slices"]), "zoom": 1.0 + .5 * k / beats_in,
                              "fx": ([bu] if alt else []) + (["rgbshift"] if k == beats_in - 1 else []), "flash": k == beats_in - 1, "tr": "cut"})
            b += 1
            cb = r.choice(used) if used else hero
            d0 = b * bar_len; d1 = min(dur, d0 + bar_len)
            if d1 - d0 > 1 / FPS:
                shots.append({**base, **bg_choice(), **next_look(), "hero": cb, "move": moves.draw(), "t0": d0, "t1": d1, "fx": drops.draw(), "zoom": 1.15, "drop": True, "tr": r.choice(["mosh", "mosh", "melt", "zoom", "static"])})
                b += 1
            since_drop = 0
    for s_ in shots: s_["mosh"] = s_.get("tr") == "mosh"
    return [s_ for s_ in shots if s_["t1"] - s_["t0"] > 1 / FPS], heroes

_FONT_CACHE = {}
def overlay_text(img, lines, y, size):
    """TikTok-native caption: bold white, thick dark outline, centred"""
    b = img.convert("RGBA")
    for line in lines:
        key = (line, size)
        if key not in _FONT_CACHE:
            sp = K.tx(line, size, K.FONT["oswald"], (255, 255, 255, 255), 1); ol = K.tx(line, size, K.FONT["oswald"], (0, 0, 0, 255), 1)
            sp = C.fit(sp, w=min(sp.width, int(W * .86))); ol = C.fit(ol, w=sp.width)
            stroke = Image.new("RGBA", (ol.width + 16, ol.height + 16), (0, 0, 0, 0))
            for ox in range(-5, 6, 2):
                for oy in range(-5, 6, 2): stroke.alpha_composite(ol, (8 + ox, 8 + oy))
            stroke.alpha_composite(sp, (8, 8)); _FONT_CACHE[key] = stroke
        sp = _FONT_CACHE[key]; b.alpha_composite(sp, (int(W / 2 - sp.width / 2), int(y))); y += sp.height * 1.02
    return b.convert("RGB")

# ---------- colour looks and raw textures, picked per section so one edit moves between them ----------
LOOKS = {"psych": 3, "accent": 2, "duo": 2, "night": 1.5, "acid": 1, "tri": 1}
RAWS = {None: 4, "s_hero": 1.5, "s_around": 1.2, "s_luma": 1.3, "s_rings": 1, "s_hex": 1, "s_scan": 1, "tear": 1.2, "vhs": 1.2}
_RG = {}
def _grids():
    if "r" not in _RG:
        xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)); cx, cy = W / 2, H / 2
        _RG.update(r=np.hypot(xs - cx, ys - cy) / math.hypot(cx, cy), xs=xs, ys=ys, ang=np.arctan2(ys - cy, xs - cx))
        hx = xs / 90.0; hy = ys / 78.0; _RG["hex"] = (np.floor(hx + .5 * (np.floor(hy) % 2)) * 7 + np.floor(hy) * 13).astype(np.int32)
    return _RG

def static_tex(rr, kind_seed, tint=None):
    """TV noise in a few flavours: chunky or fine, grey, per-channel colour, tinted to the palette, or horizontal streaks"""
    f = int(kind_seed * 97) % 5; px = (1, 2, 3, 2, 4)[f]; h_, w_ = H // px + 1, W // px + 1
    if f == 3: n = np.repeat(rr.integers(0, 255, (h_, 1), dtype=np.uint8), w_, 1)[..., None].repeat(3, 2)        # streaks
    elif f == 1: n = rr.integers(0, 255, (h_, w_, 3), dtype=np.uint8)                                                 # colour noise
    else: n = rr.integers(0, 255, (h_, w_, 1), dtype=np.uint8).repeat(3, 2)
    n = n.repeat(px, 0).repeat(px, 1)[:H, :W]
    if tint is not None and f in (0, 4): n = (n.astype(np.float32) * (np.array(tint, np.float32) / 255 * .8 + .2)).astype(np.uint8)
    return n
ACCENTS = [(255, 30, 60), (0, 255, 160), (255, 220, 0), (0, 200, 255), (255, 0, 200), (255, 110, 0)]
def look_fx(img, look, pal, acc):
    if look in (None, "psych"): return img
    a = np.asarray(img, np.float32) / 255; L = a @ np.array([.299, .587, .114], np.float32)
    if look == "accent":                                                     # hard black/white, only the saturated bits keep one vivid colour
        g = np.clip((L - .5) * 2.2 + .5, 0, 1); sat = a.max(2) - a.min(2); m = np.clip((sat - .35) * 3, 0, 1)[..., None]
        out = g[..., None] * (1 - m) + (np.array(acc, np.float32) / 255) * np.clip(L * 1.6, .35, 1)[..., None] * m
    elif look == "duo":                                                      # two strong colours, crushed contrast
        g = np.clip((L - .45) * 2.4 + .5, 0, 1)[..., None]; c0, c1 = (np.array(c, np.float32) / 255 for c in pal); out = c1 * .08 * (1 - g) + c0 * g
    elif look == "night":                                                    # dark, glowing, saturated
        out = np.clip(a ** 1.7 * 1.25, 0, 1); m = out.mean(2, keepdims=True); out = np.clip(m + (out - m) * 1.9, 0, 1)
    elif look == "acid":                                                     # hue rotated and pushed
        out = a[..., [1, 2, 0]]; m = out.mean(2, keepdims=True); out = np.clip(m + (out - m) * 2.0, 0, 1)
    else:                                                                    # tri: three-colour poster
        q = np.digitize(L, [.33, .66]); cols = np.array([(10, 0, 30), pal[1], pal[0]], np.float32) / 255; out = cols[q]
    return Image.fromarray((out * 255).astype(np.uint8))

def raw_fx(img, kind, t, k, rr, mask=None, beat=0.0, sseed=.3, tint=None):
    """static that interacts with the picture: inside the hero, around it, in the shadows, in rings on the kick, in hex cells, in scan bands"""
    if not kind: return img
    a = np.asarray(img).copy(); h, w = a.shape[:2]
    if kind.startswith("s_"):
        g = _grids(); n = static_tex(rr, sseed + (k // 2) * .37 if kind != "s_hero" else sseed, tint)
        if kind == "s_hero" and mask is not None:                             # the hero is a window onto static, edges stay clean
            m = np.asarray(mask, np.float32) / 255; core = np.asarray(mask.filter(ImageFilter.MinFilter(9)), np.float32) / 255
            m = (core * (.75 + .25 * beat))[..., None]
        elif kind == "s_around" and mask is not None:                         # everything but the hero dissolves into static, brightest bits survive
            L = a.mean(2) / 255; m = ((1 - np.asarray(mask, np.float32) / 255) * np.clip(1.3 - L * 1.6, 0, 1) * (.65 + .35 * beat))[..., None]
        elif kind == "s_luma":                                                # the shadows are static, the line moves with the kick
            L = a.mean(2) / 255; th = .22 + .18 * beat + .08 * math.sin(t * 1.3); m = np.clip((th - L) * 6, 0, 1)[..., None]
        elif kind == "s_rings":                                               # rings of static rolling out from the centre on each kick
            ph = (g["r"] * 7 - t * 3.2 + sseed * 5) % 1.0; m = (np.clip(1 - np.abs(ph - .5) * 6, 0, 1) * (.35 + .65 * beat))[..., None]
            if mask is not None: m = m * (1 - np.asarray(mask, np.float32)[..., None] / 255 * .85)
        elif kind == "s_hex":                                                 # random hex cells flip to static, re-rolled each beat
            sel = (((g["hex"] * 2654435761 + int(k / max(1, FPS * .45)) * 97) % 1000) < 230 + 200 * beat); m = sel.astype(np.float32)[..., None] * .9
        else:                                                                 # s_scan: diagonal bands of static sweeping across
            m = (np.sin((g["xs"] * .6 + g["ys"]) * .012 - t * 6) > .6 - .5 * beat).astype(np.float32)[..., None] * .85
        return Image.fromarray((a * (1 - m) + n * m).astype(np.uint8))
    if kind == "static":                                                     # TV static over everything, with a rolling bright band
        n = rr.integers(0, 255, (h // 2, w // 2), dtype=np.uint8).repeat(2, 0).repeat(2, 1)[:h, :w, None]
        al = .14 + .1 * math.sin(t * 7); y0 = int((t * 900) % h); band = slice(y0, min(h, y0 + 70))
        a = (a * (1 - al) + n * al).astype(np.uint8); a[band] = np.clip(a[band].astype(np.int16) + 60, 0, 255).astype(np.uint8)
    elif kind == "tear":                                                     # horizontal tearing in bands, re-rolled every few frames
        r2 = random.Random(k // 3)
        for _ in range(r2.randint(2, 6)):
            y = r2.randrange(0, h - 40); hh = r2.randint(8, 90); a[y:y + hh] = np.roll(a[y:y + hh], r2.randint(-160, 160), axis=1)
    elif kind == "vhs":                                                      # chroma bleed, scan noise and a tracking band
        a[..., 0] = np.roll(a[..., 0], 6, 1); a[..., 2] = np.roll(a[..., 2], -5, 1)
        y0 = int((h * .85 + t * 160) % h); hh = 40; seg = a[y0:y0 + hh]
        a[y0:y0 + hh] = np.clip(seg.astype(np.int16) + rr.integers(-70, 70, seg.shape, dtype=np.int16), 0, 255).astype(np.uint8)
        a[::3] = (a[::3] * .82).astype(np.uint8)
    return Image.fromarray(a)

_GRID = {}
def liquid(img, amp, t, seed=0.0):
    """flowing displacement: two crossing sine fields, sampled per pixel (amp in px)"""
    a = np.asarray(img); h, w = a.shape[:2]
    if (w, h) not in _GRID: _GRID[(w, h)] = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    xs, ys = _GRID[(w, h)]
    dx = amp * (np.sin(ys * .011 + t * 2.3 + seed * 6) + .5 * np.sin((xs + ys) * .007 - t * 1.7))
    dy = amp * (np.sin(xs * .013 - t * 1.9 + seed * 9) + .5 * np.sin((xs - ys) * .006 + t * 2.1))
    ix = np.clip(xs + dx, 0, w - 1).astype(np.int32); iy = np.clip(ys + dy, 0, h - 1).astype(np.int32)
    return Image.fromarray(a[iy, ix])

def build_mosaic(items, sprites, r):
    """1024x1024 tapestry: a 4x4 (sometimes uneven) patchwork of different images and cut-outs, for the landscape and 3D lattice"""
    tiles = [im for im in items if im is not None] + sprites; r.shuffle(tiles)
    if not tiles: return None
    mos = Image.new("RGB", (1024, 1024)); cells = []
    for y in range(4):
        x = 0
        while x < 4:
            wcell = 2 if x < 3 and r.random() < .2 else 1; cells.append((x, y, wcell)); x += wcell
    for k, (x, y, wc) in enumerate(cells):
        src = tiles[k % len(tiles)]; cw, ch = 256 * wc, 256
        if src.mode == "RGBA":
            bg = Image.new("RGB", (cw, ch), tuple(int(c * .25) for c in HERO_PALS[k % len(HERO_PALS)][1])); sp = C.fit(src, w=int(min(cw, ch) * .9)); bg.paste(sp, (cw // 2 - sp.width // 2, ch // 2 - sp.height // 2), sp); tile = bg
        else: tile = ImageEnhance.Color(cover(src, cw, ch)).enhance(1.4)
        mos.paste(tile, (x * 256, y * 256))
    mos.big = True; return mos

def render2(shots, dur, fam, style, seed, r, punches, bars, env, keyframes, song=None, mosaic_src=None, per_beat=.45):
    env_t, env_v = env
    avi = os.path.join(WORK, "clean.avi"); kf = ",".join(f"{c:.3f}" for c in sorted(set(keyframes) | {0.0}))
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libxvid", "-q:v", "3", "-g", "9999", "-bf", "0", "-force_key_frames", kf, avi], stdin=subprocess.PIPE)
    sprites = {}; walls = {}; kaleido_tex = {}
    def hero_im(s):
        k = (s["hero"], s["pal"])
        if k not in sprites: sprites[k] = hero_sprite(s["hero"], int(W * .72), s["pal"], fam, r)
        return sprites[k]
    def wall_im(it):
        if it is None: return None
        if it["id"] not in walls:
            im = fetch_image(it); walls[it["id"]] = ImageEnhance.Color(im).enhance(1.6) if im is not None else None
        return walls[it["id"]]
    series = MONT.get("series", {}); top = series.get("top", []); bottom = series.get("bottom", [])
    si = 0; nfr = int(round(dur * FPS)); barset = list(bars)
    MOS = None
    if any(s_["bg"] == "mosaic" for s_ in shots) and mosaic_src:
        wl, extra = mosaic_src; ims = [wall_im(w_) for w_ in wl[:10]] + [fetch_image(e) for e in extra if e]
        sps = [hero_im({"hero": h_, "pal": HERO_PALS[k_ % len(HERO_PALS)]}) for k_, h_ in enumerate(list(dict.fromkeys(s_["hero"] for s_ in shots)))][:6]
        MOS = build_mosaic(ims, sps, r)
    ph = [(r.uniform(.15, .5), r.uniform(0, 6.3)) for _ in range(4)]               # slow, non-repeating drift for zoom and tilt
    driftz = lambda t: 1 + .035 * math.sin(ph[0][0] * t + ph[0][1]) + .02 * math.sin(ph[1][0] * 2.3 * t + ph[1][1])
    driftr = lambda t: 4 * math.sin(ph[2][0] * t + ph[2][1]) + 2 * math.sin(ph[3][0] * 1.7 * t + ph[3][1])
    last_img = None; cur_si = -1; frozen = None; kstart = 0; NR = np.random.default_rng(int(seed * 1000) % 2**31)
    for k in range(nfr):
        t = k / FPS
        while si + 1 < len(shots) and t >= shots[si + 1]["t0"] - 1e-6: si += 1
        s = shots[si]; u = (t - s["t0"]) / max(1e-3, s["t1"] - s["t0"])
        if si != cur_si: cur_si = si; frozen = last_img; kstart = k
        last = max((p for p in punches if p <= t + 1e-6), default=-9); beat = math.exp(-(t - last) * 8)
        bar_last = max((b for b in barset if b <= t + 1e-6), default=-9); energy = float(np.interp(t, env_t, env_v)) if len(env_t) else .5
        op = s.get("opener")
        if op and t < op["end"]:                                              # the meme opener: full frame, punched on the beats
            if "im" not in op:
                im = fetch_image(op["img"]); op["im"] = ImageEnhance.Contrast(im).enhance(1.1) if im is not None else None
            if op["im"] is not None:
                img = cover(op["im"], zoom=1.04 + .07 * beat + .05 * t / op["end"]); last_img = img
                if top: img = overlay_text(img, top, H * .085, 66)
                if bottom: img = overlay_text(img, bottom, H * .725, 60)
                if song: img = overlay_text(img, [song], H * .805, 40)
                enc.stdin.write(img.tobytes()); continue
        # background
        if s["bg"] == "shader": fr = shader_frame(s["shader"], t, beat, energy, s["sseed"], fam=fam).convert("RGBA")
        elif s["bg"] == "mosaic" and MOS is not None: fr = shader_frame(s["shader"], t, beat, energy, s["sseed"], tex=MOS, fam=fam).convert("RGBA")
        elif s["bg"] == "mosaic": fr = shader_frame("truchet", t, beat, energy, s["sseed"], fam=fam).convert("RGBA")
        elif s["bg"] == "tess":
            tk = (s["tsrc"], s["pal"][::-1])
            if tk not in sprites: sprites[tk] = hero_sprite(s["tsrc"], int(W * .45), s["pal"][::-1], fam, r)
            fr = shader_frame(s["shader"], t, beat, energy, s["sseed"], tex=sprites[tk], dim=.85, fam=fam).convert("RGBA")
        elif s["bg"] == "kwall" and wall_im(s.get("wall")) is not None:
            fr = shader_frame("kaleido", t, beat, energy, s["sseed"], tex=wall_im(s["wall"]).convert("RGBA"), fam=fam).convert("RGBA")
        elif wall_im(s.get("wall")) is not None:
            c0, c1 = s["pal"]; l = np.asarray(cover(wall_im(s["wall"]), zoom=1.1 + .05 * u, px=s.get("bpx", .5), py=s.get("bpy", .5)).convert("L"), np.float32)[..., None] / 255
            fr = Image.fromarray((np.array(c1) * .35 * (1 - l) + np.array(c0) * .5 * l).astype(np.uint8)).convert("RGBA")
        else: fr = shader_frame("plasma", t, beat, energy, s.get("sseed", .3), fam=fam).convert("RGBA")
        # hero
        hmask = None; hi = hero_im(s); z = s["zoom"] * (1 + .09 * beat + .05 * math.exp(-(t - bar_last) * 5)) * (1 + s["drift"] * u) * driftz(t)
        if "tess_hero" in s["fx"]:                                            # drop: the hero tiles the whole background, hero on top
            fr = shader_frame(TESS[int(s.get("sseed", .5) * 7) % len(TESS)], t, beat, energy, s.get("sseed", .5), tex=hi, fam=fam).convert("RGBA")
        if "kaleido_hero" in s["fx"]:
            kt = kaleido_tex.setdefault(id(hi), hi)
            fr = shader_frame("kaleido", t, beat, energy, s.get("sseed", .5), tex=kt, fam=fam).convert("RGBA")
        else:
            mv = s.get("move", "center")
            if mv == "zoomthrough": z *= .35 + 2.6 * u ** 2                           # flies at the camera across the bar
            h = hi.resize((max(1, int(hi.width * z)), max(1, int(hi.height * z))), Image.BICUBIC)
            rot = s["rot"] * math.sin(t * .8) + driftr(t)
            if mv == "coinspin":                                                # flat 3D spin: width follows cos
                sx = abs(math.cos(u * math.pi * 2)) * .95 + .05; h = h.resize((max(1, int(h.width * sx)), h.height), Image.BICUBIC)
                if math.cos(u * math.pi * 2) < 0: h = ImageOps.mirror(h)
            elif mv == "orbit":                                                 # big ghost + small copy orbiting on the beat
                C.paste(fr, h, W / 2, H * .5, alpha=.25)
                h = hi.resize((int(hi.width * .45), int(hi.height * .45)), Image.BICUBIC); a_ = t * 2.4
                C.paste(fr, h, W / 2 + math.cos(a_) * W * .28, H * .5 + math.sin(a_) * H * .18, rot=a_ * 40); C.paste(fr, h, W / 2 - math.cos(a_) * W * .28, H * .5 - math.sin(a_) * H * .18, rot=-a_ * 40)
                h = None
            elif mv == "droste":                                                # the hero inside itself, shrinking in
                for j in range(6, 0, -1):
                    hj = hi.resize((max(1, int(hi.width * z * .85 ** j)), max(1, int(hi.height * z * .85 ** j))), Image.BICUBIC); C.paste(fr, hj, W / 2, H * .5, rot=rot + j * 9 * math.sin(t), alpha=.35 + .1 * (6 - j) / 6)
            elif mv == "split4":                                                # 2x2 of two heroes, swapping on the beat
                o = sprites.get(("other", s["other"], s["pal"]))
                if o is None: o = sprites[("other", s["other"], s["pal"])] = hero_sprite(s["other"], int(W * .72), (s["pal"][1], s["pal"][0]), fam, r)
                flip = int((t - last) < .1) ^ (len([p for p in punches if p <= t]) % 2)
                for qi, (qx, qy) in enumerate(((.25, .3), (.75, .3), (.25, .7), (.75, .7))):
                    src = hi if (qi + flip) % 2 == 0 else o; q = src.resize((int(src.width * .46 * z), int(src.height * .46 * z)), Image.BICUBIC); C.paste(fr, q, W * qx, H * qy)
                h = None
            elif mv == "strobe2":                                               # two heroes alternating on every beat
                o = sprites.get(("other", s["other"], s["pal"]))
                if o is None: o = sprites[("other", s["other"], s["pal"])] = hero_sprite(s["other"], int(W * .72), (s["pal"][1], s["pal"][0]), fam, r)
                if len([p for p in punches if p <= t]) % 2: h = o.resize((max(1, int(o.width * z)), max(1, int(o.height * z))), Image.BICUBIC)
            elif mv == "triple":                                                # three offset copies, RGB-split by hand
                for j, col in enumerate(((255, 0, 80), (0, 255, 220), (255, 255, 0))):
                    tint = Image.new("RGBA", h.size, col + (0,)); tint.putalpha(h.split()[3].point(lambda v: v * 0.55)); C.paste(fr, tint, W / 2 + (j - 1) * 30 * (1 + beat), H * .5 + (j - 1) * 14)
            elif mv == "slices":                                                # horizontal strips sliding apart on the beat
                strips = 12; sh_ = h.height / strips; out_ = Image.new("RGBA", (h.width + 200, h.height), (0, 0, 0, 0))
                for j in range(strips):
                    piece = h.crop((0, int(j * sh_), h.width, int((j + 1) * sh_))); out_.alpha_composite(piece, (int(100 + math.sin(j * 1.7 + t * 6) * 90 * beat), int(j * sh_)))
                h = out_
            elif mv == "pixelsort":
                hs = G.pixel_sort(h.convert("RGB"), lo=40, hi=230, axis=1, seed=int(t * 10), chance=.6).convert("RGBA"); hs.putalpha(h.split()[3]); h = hs
            elif mv == "shatter":                                               # tiles of the hero drifting off-grid
                n_ = 5; tw, th = h.width // n_, h.height // n_; out_ = Image.new("RGBA", (h.width + 240, h.height + 240), (0, 0, 0, 0))
                for gy in range(n_):
                    for gx in range(n_):
                        rr = random.Random(gx * 31 + gy * 7); piece = h.crop((gx * tw, gy * th, (gx + 1) * tw, (gy + 1) * th)); d_ = 120 * u * u
                        out_.alpha_composite(piece, (int(120 + gx * tw + rr.uniform(-1, 1) * d_), int(120 + gy * th + rr.uniform(-1, 1) * d_)))
                h = out_
            elif mv == "scan":                                                  # the hero revealed by a scanning bar
                m_ = Image.new("L", h.size, 0); y_ = int(h.height * ((u * 1.6) % 1.0)); ImageDraw.Draw(m_).rectangle([0, 0, h.width, y_], fill=255)
                h = h.copy(); h.putalpha(ImageChops.multiply(h.split()[3], m_))
            if h is None: pass
            if h is not None and "tile" in s["fx"]:
                small = C.fit(hi, w=int(W * .3)); cols = 4; rows = int(H / small.height) + 2; off = (u * small.height) % small.height
                for gy in range(rows):
                    for gx in range(cols): C.paste(fr, small, (gx + .5 * (gy % 2)) * W / (cols - 1), gy * small.height - off)
            if h is not None and "echo" in s["fx"]:
                for j in range(4, 0, -1): C.paste(fr, h, W / 2 + j * 22 * math.cos(t * 3), H * .5 + j * 30, rot=rot, alpha=.15 + .08 * (4 - j))
            if h is not None:
                C.paste(fr, h, W / 2, H * .5, rot=rot if mv not in ("coinspin", "slices", "shatter") else 0)
                if s.get("raw") in ("s_hero", "s_around", "s_rings"):
                    hm = Image.new("RGBA", (W, H), (0, 0, 0, 0)); C.paste(hm, h, W / 2, H * .5, rot=rot if mv not in ("coinspin", "slices", "shatter") else 0); hmask = hm.split()[3]
        if "ascii_bg" in s["fx"]:                                                # background in ASCII, hero stays clean on top
            bgi = apply_fx(fr.convert("RGB"), ["ascii"], {"pal": s["pal"], "cell": 14, "apal": True}, t, beat).convert("RGBA")
            if h is not None: C.paste(bgi, h, W / 2, H * .5)
            fr = bgi
        img = fr.convert("RGB")
        if "invert_beats" in s["fx"] and (t - last) < .08: img = ImageOps.invert(img)
        if "posterize" in s["fx"]: img = ImageOps.posterize(ImageEnhance.Color(img).enhance(1.8), 2)
        fx = s["fx"]; fs = {"pal": s["pal"], "cell": 16, "apal": True}
        img = apply_fx(img, [f for f in fx if f in ("mirror2", "mirror4", "duo", "invert", "ascii", "rgbshift")], fs, t, beat)
        if "liquid" in fx: img = liquid(img, 14 + 26 * beat, t, s.get("sseed", .3))       # the whole frame flows, harder on the kick
        mf = s.get("meme_flash")
        if mf and 0 <= t - mf["t"] < 2.5 / FPS:                                         # the opener meme comes back for a blink
            if "im" not in mf: mf["im"] = fetch_image(mf["img"])
            if mf["im"] is not None: img = cover(mf["im"], zoom=1.15)
        img = look_fx(img, s.get("look"), s["pal"], s.get("acc", (255, 30, 60)))
        img = raw_fx(img, s.get("raw"), t, k, NR, hmask, beat, s.get("sseed", .3), s.get("acc"))
        pre = img
        n_in = k - kstart; tr = s.get("tr", "cut"); mfr = max(3, min(9, int(per_beat * FPS)))
        if tr == "static" and frozen is not None and n_in < 5:                        # static ring wipe: old frame outside, static band, new frame inside
            g = _grids(); a_ = (n_in + 1) / 5; band = .28
            inner = (g["r"] < a_ * 1.1)[..., None]; ring = ((g["r"] >= a_ * 1.1) & (g["r"] < a_ * 1.1 + band))[..., None]
            nz = static_tex(NR, s.get("sseed", .3) + n_in * .21, s.get("acc"))
            img = Image.fromarray(np.where(inner, np.asarray(img), np.where(ring, nz, np.asarray(frozen))).astype(np.uint8))
        if frozen is not None and n_in < mfr and tr in ("melt", "zoom"):
            a = (n_in + 1) / (mfr + 1)
            if tr == "melt":                                                          # liquid morph: both frames flow into each other
                img = Image.blend(liquid(frozen, 60 * a, t, .7), liquid(img, 60 * (1 - a), t, .2), a)
            else:                                                                     # zoom-through: the old frame rushes past the camera
                zz = 1 + 1.6 * a; ow, oh = int(W * zz), int(H * zz); ox, oy = (ow - W) // 2, (oh - H) // 2
                img = Image.blend(frozen.resize((ow, oh), Image.BILINEAR).crop((ox, oy, ox + W, oy + H)), img, min(1, a * 1.4))
        last_img = pre
        strobe = .8 if any(abs(b - t) < 1.2 / FPS for b in barset) else 0.0
        if s.get("flash") and u > .7: strobe = 1.0
        img = post(img, fam, t, seed, strobe)
        if top: img = overlay_text(img, top, H * .085, 66)
        if bottom: img = overlay_text(img, bottom, H * .725, 60)
        if song: img = overlay_text(img, [song], H * .805, 40)
        enc.stdin.write(img.tobytes())
    enc.stdin.close(); enc.wait()
    if enc.returncode: raise RuntimeError("frame encoder failed")
    return nfr

# ============================ v3: the compositor ============================
_COMP = None
def comp_ctx():
    global _COMP
    if _COMP is None:
        import comp as CP; _COMP = CP.Comp(W, H)
    return _COMP

ANALOG = {0: 3, 1: 2, 2: 1.2, 3: 2, 4: .8}                              # none, VHS, composite, film, CRT
def render3(shots, dur, fam, style, seed, r, punches, bars, env, keyframes, song, photos, cuts_png, opener, refs=()):
    """every section is a compositor graph over real material: atlas photos, cut-outs, the meme, static, noise, a point cloud.
    Graphs come from the recipes (generalised from what reads well) or are fully random; dull or near-black graphs are re-rolled."""
    import comp as CP
    Cx = comp_ctx(); env_t, env_v = env
    avi = os.path.join(WORK, "clean.avi"); kf = ",".join(f"{c:.3f}" for c in sorted(set(keyframes) | {0.0}))
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libxvid", "-q:v", "3", "-g", "9999", "-bf", "0", "-force_key_frames", kf, avi], stdin=subprocess.PIPE)
    series = MONT.get("series", {}); top = series.get("top", []); bottom = series.get("bottom", [])
    R = CP.recipes(r); names = list(R); r.shuffle(names); deck = []
    CALM = ["raw", "raw", "rawpair", "peek", "scan", "collide", "glyphs", "tiles", "diff", "warp"]
    def pick_graph(drop, inten=1.0):
        nonlocal deck
        if inten < .62 and r.random() < .8: return R[r.choice(CALM)](), "calm"      # the build-up: real photos, light treatments
        if r.random() < (.35 if drop else .22): return CP.random_graph(r, r.choice([2, 3, 3, 4])), "random"
        if not deck: deck = names[:] + ["raw", "raw", "rawpair", "glyphs", "collide"]; r.shuffle(deck)   # raw photo moments and references come up often
        nm = deck.pop()
        if drop and nm in ("scan", "diff") and r.random() < .6: nm = r.choice(["cloud", "kaleid", "bend", "tunnel"])
        return R[nm](), nm
    imgs = {}
    def photo(it):
        if it["id"] not in imgs: imgs[it["id"]] = fetch_image(it)
        return imgs[it["id"]]
    meme_im = fetch_image(opener) if opener else None
    ref_ims = [im for im in (fetch_image(x) for x in refs) if im is not None]   # cultural references: memes, film stills, cartoons, icons
    pool = [p_ for p_ in photos if p_]; r.shuffle(pool)
    cut_ims = []
    for f in cuts_png[:10]:
        try: cut_ims.append(C.load(f, 1200))
        except Exception: pass
    # plan per shot: graph, sources, analog, point-cloud shape
    plan = []
    for i, s in enumerate(shots):
        tries = 0
        while True:
            g, nm = pick_graph(s.get("drop"), s.get("inten", 1.0)); tries += 1
            ps = [photo(pool[(i * 2 + k + tries) % len(pool)]) if pool else None for k in range(3)]
            ps = [p_ if p_ is not None else (meme_im or (cut_ims[0] if cut_ims else None)) for p_ in ps]
            cut = (meme_im if (meme_im is not None and r.random() < .15) else (r.choice(cut_ims) if cut_ims else ps[2]))
            srcs = [ps[0], ps[1], cut, ps[2]]
            if ref_ims and r.random() < .65:                            # a reference in most sections, juxtaposed with the scene's own photos
                srcs[2] = r.choice(ref_ims)
                if r.random() < .35: srcs[r.choice([1, 3])] = r.choice(ref_ims)
            if tries > 6 or nm != "random": break
            Cx.compile(g); [Cx.set_source(k, im) for k, im in enumerate(srcs)]
            test = np.asarray(Cx.frame(1.0, .5, .5, .3).resize((108, 192)), np.float32) / 255
            if .07 < test.mean() < .93 and test.std() > .07: break      # not black, not blown out, not flat
        if r.random() < .75:                                            # the whole picture breathes: slow turn + a push on the kick
            g = ("scale", r.uniform(.92, 1.08), r.uniform(.04, .14), ("rotate", r.uniform(-.1, .1), r.uniform(-.18, .18), g))
        inten = s.get("inten", 1.0)
        colour = s.get("drop") or nm in ("fight",) or r.random() < (.08 if inten < .9 else .3)
        plan.append({"g": g, "nm": nm, "srcs": srcs, "swap": r.random() < (.3 + .4 * inten), "mono": not colour, "inv": (inten >= .99) and r.random() < (.55 if s.get("drop") else .25),
                     "bal": (.25 if s.get("drop") else r.uniform(-.3, .2)), "scheme": (r.choice(ACCENTS) if (not colour and r.random() < .2) else None), "an": r.choices(list(ANALOG), list(ANALOG.values()))[0], "ak": r.uniform(.5, 1.2) * (1.3 if s.get("build") else 1),
                     "shape": r.choice([0, 0, 1, 2]), "pa": r.randrange(2), "seed": r.random()})
    si = -1; nfr = int(round(dur * FPS)); barset = list(bars); last = None; teasing = False
    drop_plan = next((plan[k_] for k_, s_ in enumerate(shots) if s_.get("drop") and s_.get("post")), None)
    for k in range(nfr):
        t = k / FPS
        ni = si
        while ni + 1 < len(shots) and t >= shots[ni + 1]["t0"] - 1e-6: ni += 1
        if ni != si:                                                     # new section: new graph, new sources
            si = ni; s = shots[si]; P = plan[si]; Cx.compile(P["g"])
            for kk, im in enumerate(P["srcs"]): Cx.set_source(kk, im)
            Cx.sim_bal = P["bal"]; k0 = k; kicks = 0
        s = shots[si]; P = plan[si]; u = (t - s["t0"]) / max(1e-3, s["t1"] - s["t0"])
        lastp = max((p_ for p_ in punches if p_ <= t + 1e-6), default=-9); beat = math.exp(-(t - lastp) * 8)
        if P.get("swap") and pool and abs(t - lastp) < .5 / FPS and t - s["t0"] > .1 and r.random() < .5:   # on a kick, a new photo drops into the graph
            nxt = photo(r.choice(pool))
            if nxt is not None: Cx.set_source(r.choice([0, 0, 1, 3]), nxt)
        energy = float(np.interp(t, env_t, env_v)) if len(env_t) else .5
        op = s.get("opener")
        if op and t < op["end"] and meme_im is not None:
            img = cover(meme_im, zoom=1.04 + .07 * beat + .05 * t / op["end"]); Cx.set_source(3, meme_im)
        else:
            n_in = k - k0; mfr = max(3, min(10, int(.45 * FPS)))
            tz = s.get("tease")
            if tz is not None and drop_plan is not None and 0 <= t - tz < 3 / FPS:      # a glimpse of what's coming
                if not teasing: Cx.compile(drop_plan["g"]); [Cx.set_source(kk, im) for kk, im in enumerate(drop_plan["srcs"])]; teasing = True
            elif teasing:
                Cx.compile(P["g"]); [Cx.set_source(kk, im) for kk, im in enumerate(P["srcs"])]; teasing = False
            mix = (n_in + 1) / (mfr + 1) if (s.get("tr") in ("melt", "zoom") and n_in < mfr) else 1.0
            img = Cx.frame(t, beat, energy, P["seed"], mix, pc=(u, P["shape"], P["pa"], 1 - P["pa"]))
            if s.get("tr") == "static" and n_in < 4 and last is not None:   # static ring wipe between sections
                g_ = _grids(); a_ = (n_in + 1) / 4; inner = (g_["r"] < a_ * 1.1)[..., None]; ring = ((g_["r"] >= a_ * 1.1) & (g_["r"] < a_ * 1.1 + .28))[..., None]
                img = Image.fromarray(np.where(inner, np.asarray(img), np.where(ring, static_tex(NR3, P["seed"] + n_in * .2), np.asarray(last))).astype(np.uint8))
            mf = s.get("meme_flash")
            if mf and meme_im is not None and 0 <= t - mf["t"] < 2.5 / FPS: img = cover(meme_im, zoom=1.15)
        if abs(t - lastp) < .5 / FPS: kicks += 1
        if P["mono"]:                                                    # hard black and white; a single accent colour sometimes survives in the highlights
            a_ = np.asarray(img, np.float32); L_ = a_ @ np.array([.299, .587, .114], np.float32); g_ = np.clip((L_ - 118) * 2.3 + 128, 0, 255)
            if P["scheme"] is not None:
                sat = a_.max(2) - a_.min(2); m_ = np.clip((sat - 90) / 60, 0, 1)[..., None]
                img = Image.fromarray((g_[..., None] * (1 - m_) + np.array(P["scheme"], np.float32) * (g_[..., None] / 255) * m_).astype(np.uint8))
            else: img = Image.fromarray(g_.astype(np.uint8)).convert("RGB")
        if P["inv"] and kicks % 2 == 1 and t - lastp < 2.5 / FPS: img = ImageOps.invert(img)   # negative on every other kick
        last = img
        strobe = .6 if any(abs(b - t) < 1.2 / FPS for b in barset) and s.get("drop") else 0.0
        if strobe: img = ImageEnhance.Brightness(img).enhance(1 + strobe)
        img = Cx.analog(img, P["an"], t, beat, P["ak"])
        if top: img = overlay_text(img, top, H * .085, 66)
        if bottom: img = overlay_text(img, bottom, H * .725, 60)
        if song: img = overlay_text(img, [song], H * .805, 40)
        enc.stdin.write(img.tobytes())
    enc.stdin.close(); enc.wait()
    if enc.returncode: raise RuntimeError("frame encoder failed")
    return nfr
NR3 = np.random.default_rng(7)

def encode_moshed(frames_dir, nfr, cuts, mosh_cuts, out_mp4, audio_path, start, dur, fam):
    """xvid AVI with I-frames only at cuts → delete the I-frames at the moshed cuts → h264 + the track"""
    avi = os.path.join(WORK, "clean.avi")                                # written by render() straight from memory
    # which I-frames to drop: the ones at moshed cuts (frame index = round(c*FPS)); the first always stays
    drop_at = {int(round(c * FPS)) for c in mosh_cuts if c > 0}
    d = bytearray(open(avi, "rb").read()); movi = d.find(b"movi"); i = movi + 4; out = bytearray(d[:i]); frames = []; idx = 0; after = -99
    while i + 8 <= len(d):
        fourcc = bytes(d[i:i + 4]); size = int.from_bytes(d[i + 4:i + 8], "little")
        if fourcc == b"idx1": break
        chunk = bytes(d[i:i + 8 + size + (size & 1)]); payload = chunk[8:8 + size]
        if fourcc[2:] == b"dc":
            ty = G._vop_type(payload); drop = ty == 0 and idx > 0 and any(abs(idx - a) <= 1 for a in drop_at)
            if drop and frames: frames.append(frames[-1])          # keyframe replaced by the previous frame: frame count (= sync with the track) unchanged
            else: frames.append(chunk)
            idx += 1
        i += 8 + size + (size & 1)
    moshed = os.path.join(WORK, "mosh.avi"); open(moshed, "wb").write(bytes(out + b"".join(frames)))
    vid = os.path.join(WORK, "video.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-err_detect", "ignore_err", "-fflags", "+genpts+igndts", "-i", moshed, "-vf", f"scale={OUT_W}:{OUT_H}:flags=lanczos,fps={FPS}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-maxrate", "7M", "-bufsize", "14M", "-preset", "medium", "-profile:v", "high", "-level", "4.1", "-an", vid], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", vid, "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", audio_path, "-filter_complex", f"[1:a]afade=t=in:st=0:d=0.15,afade=t=out:st={dur-0.6:.3f}:d=0.6,loudnorm=I=-14:TP=-1.5[a]", "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out_mp4], check=True)

# ============================ one edit ============================
_SIZE = {}
def _big_enough(f, min_px=180 * 180):
    if f not in _SIZE:
        try: im = Image.open(f); _SIZE[f] = im.width * im.height
        except Exception: _SIZE[f] = 0
    return _SIZE[f] >= min_px

def pick_meme(man, mont, r):
    """a well-known meme / pop-culture image from the atlas (style "meme"), kept ones first, not one used in the last 25 edits"""
    if r.random() > float(MONT.get("meme_opener", .85)): return None
    recent = {it.get("opener") for it in mont["items"][:25]}
    its = [i for i in man["styles"].get("meme", []) if i["id"] not in recent]
    ok = [i for i in its if i.get("ok")]; pool = ok if len(ok) >= 5 else sorted(its, key=lambda i: -(i.get("cs") or 0))[:60]
    return r.choice(pool) if pool else None

def clean_title(t):
    t = re.sub(r"[\(\[][^\)\]]*(extended|original|mix|edit|remix|tiktok|free|dl|download|bootleg|premiere|out now)[^\)\]]*[\)\]]", lambda m: m.group(0) if re.search(r"remix|bootleg|edit", m.group(0), re.I) and len(m.group(0)) < 40 else "", t, flags=re.I)
    t = re.split(r"\s[|•]\s|\s\|\s?|#", t)[0]
    t = re.sub(r"[\U0001F000-\U0001FFFF\u2600-\u27BF]", "", t); t = re.sub(r"\s{2,}", " ", t).strip(" -–")
    return t[:48] + ("…" if len(t) > 48 else "")

def make_one(style, els, man, flags, mont, used_tracks, seed):
    r = random.Random(seed); fam = fam_of(style)
    recent = set(sum((it.get("els", []) for it in mont["items"][:20]), []))   # elements used in the last 20 edits are rested
    def usable(fs): return [f for f in fs if os.path.basename(f) not in recent and _big_enough(f)]
    own = usable(C.pool(style)); fam_pool = usable(sum((C.pool(s2) for s2 in MONT["families"][fam]["styles"] if s2 != style), [])); every = usable(sum((C.pool(s2) for s2 in os.listdir(C.ELS)), []))
    r.shuffle(own); r.shuffle(fam_pool); r.shuffle(every)
    figs = own[:9] + fam_pool[:4] + every[:3]                           # mostly the style, some of its family, a few wildcards
    if len(figs) < 6: figs += [f for f in (fam_pool + every) if f not in figs][:10]
    figs = list(dict.fromkeys(figs))
    if len(figs) < 3: report["skipped"].append(f"{style}: too few elements"); return None
    walls = kept_images(man, style); r.shuffle(walls); walls = walls[:14]
    if os.environ.get("MONTAGE_AUDIO"): track, tries = {"ok": True, "title": "local test", "dur": 0, "id": "local", "url": "", "path": os.environ["MONTAGE_AUDIO"], "src": "local"}, []
    else: track, tries = find_track(style, r, used_tracks)
    if not track: report["skipped"].append(f"{style}: no track ({'; '.join(f'{a}:{c}' for a, b, c in tries[:4])})"); return None
    log(f"{style}: track {track['src']} {track['title']!r} {track['dur']:.0f}s")
    x, sr = pcm(track["path"]); t, flux, rms, peaks = onsets(x, sr); total = len(x) / sr
    dur = r.uniform(DUR_MIN, min(DUR_MAX, max(DUR_MIN, total - 2))); start = pick_section(t, flux, rms, peaks, total, dur, r)
    bt, bpm, kenv, kfps = beats(x, sr)
    if len(bt): start = float(bt[int(np.argmin(np.abs(bt - start)))])      # section starts on a beat
    drop_rel = None
    dp = pick_drop(bt, kenv, kfps, t, rms, total, dur, r) if len(bt) else None
    if dp: start, drop_rel, dur = dp; log(f"{style}: build-up from {start:.1f}s, drop at +{drop_rel:.1f}s, {dur:.1f}s long")
    strength_at = lambda tt: float(np.interp(tt, np.arange(len(kenv)) / kfps, kenv / (kenv.max() + 1e-9)))
    g = grid_cuts(bt, start, dur, fam, r, strength_at)
    if g:
        cuts, strengths, beat_list, bars, per = g; dur = min(dur, (bars[-1] if len(bars) > 1 else dur))  # end on a bar line (loops cleanly)
        cuts = [c for c in cuts if c < dur - .1]
    else:
        cuts, strengths = cut_points(t, flux, peaks, start, dur, fam); beat_list = [float(t[p]) - start for p in peaks if start <= t[p] <= start + dur]; bars = []; bpm = 0
    pm = MONT["families"][fam]["mosh"]
    lines = MONT.get("abstract", {}).get("cards", [])                   # text is rare and abstract now
    v2 = MONT.get("engine") in ("v2", "v3") and g
    if v2:
        recent_walls = set(sum((it.get("walls", []) for it in mont["items"][:15]), []))
        pool = [w_ for w_ in walls if w_["id"] not in recent_walls] or walls
        for s2 in MONT["families"][fam]["styles"]:                      # widen the image pool with the style's family
            if s2 != style: pool += [w_ for w_ in kept_images(man, s2)[:30] if w_["id"] not in recent_walls]
        r.shuffle(pool); walls = pool[:12]
        shots, heroes = make_sections(bars, per, dur, fam, style, walls, r, mont)
        if drop_rel:                                                    # the arc: tease in the build-up, everything at the drop
            for k_, s_ in enumerate(list(shots)):
                if s_["t0"] < drop_rel - 1e-3 < s_["t1"] - 1e-3:            # a shot always starts exactly on the drop
                    a_, b_ = dict(s_), dict(s_); a_["t1"] = drop_rel; b_["t0"] = drop_rel; b_.update(drop=True, tr=r.choice(["mosh", "static", "zoom", "melt"]), fx=[])
                    shots[k_:k_ + 1] = [a_, b_]; break
            for s_ in shots:
                s_["post"] = s_["t0"] >= drop_rel - 1e-3
                s_["inten"] = 1.0 if s_["post"] else .3 + .45 * (s_["t0"] / drop_rel)
                if not s_["post"] and s_.get("drop"): s_["drop"] = False             # no fake climaxes before the real one
                if abs(s_["t0"] - drop_rel) < 1e-3: s_["drop"] = True
                s_["build"] = (not s_["post"]) and s_["t0"] >= drop_rel - 4 * per - 1e-3
            pre = [s_ for s_ in shots if not s_["post"] and not s_["build"] and s_["t1"] - s_["t0"] > 4 * per - 1e-3]
            for s_ in r.sample(pre, min(len(pre), r.choice([1, 2]))): s_["tease"] = s_["t0"] + per * r.choice([2, 3])   # a 3-frame glimpse of the drop
        opener = pick_meme(man, mont, r)
        if opener:                                                      # pattern interrupt: the meme for one beat (two when fast), then it melts into the edit
            oe = per * (2 if per < .36 else 1); shots[0]["opener"] = {"img": opener, "end": oe}
            shots.insert(1, {**shots[0], "t0": oe, "tr": r.choice(["melt", "melt", "mosh"]), "opener": None}); shots[1]["mosh"] = shots[1]["tr"] == "mosh"; shots[0]["t1"] = oe; shots[0]["tr"] = "cut"
            later = [s_ for s_ in shots[2:] if s_.get("drop")] or shots[len(shots) // 2:]
            for s_ in r.sample(later, min(len(later), r.choice([1, 1, 2]))):  # and comes back: a blink on a beat, or the drop's backdrop
                if r.random() < .5: s_["meme_flash"] = {"img": opener, "t": s_["t0"] + per * r.choice([0, 1, 2])}
                else: s_.update({"bg": "kwall", "wall": opener, "sseed": r.random()})
        mosh_cuts = [s_["t0"] for s_ in shots if s_.get("mosh") and s_["t0"] > 0]
    else:
        shots = make_shots(cuts, dur, fam, style, figs, walls, lines, r, bars, per if g else 0.45)
        mosh_cuts = [s_["t0"] for s_ in shots if s_.get("mosh") and s_["t0"] > 0] + [c for c in cuts[1:] if r.random() < pm * .3]
    cuts = sorted(set([round(s_["t0"], 4) for s_ in shots]))
    log(f"{style}: {dur:.1f}s from {start:.1f}s, {bpm:.1f} bpm, {len(cuts)} cuts on the grid, {len(mosh_cuts)} moshed, fam={fam}")
    punches = [b_ for b_ in beat_list if 0 <= b_ <= dur]
    hook = None
    if MONT.get("text", {}).get("hooks"):
        hooks = MONT.get("hooks", {}).get(style, []) + MONT.get("hooks", {}).get("_", []); hook = r.choice(hooks) if hooks else None
    hop_t = t[1] - t[0]; sel = (t >= start) & (t <= start + dur); ev = rms[sel] / (rms.max() + 1e-9); env = (t[sel] - start, np.clip(ev * 1.4, 0, 1))
    song = ("track · " + clean_title(track["title"])) if MONT.get("text", {}).get("song_name", True) and track.get("title") else None
    if v2 and MONT.get("engine") == "v3":
        photos = list(walls)
        for s2 in r.sample([x for x in CFG if not x.startswith("_") and x not in ("meme",)], 3): photos += kept_images(man, s2)[:4]   # a few images from elsewhere
        refs = r.sample(kept_images(man, "meme"), min(3, len(kept_images(man, "meme")))) + r.sample(kept_images(man, "pop"), min(3, len(kept_images(man, "pop"))))
        d = FR; nfr = render3(shots, dur, fam, style, seed, r, punches, bars, env, cuts, song, photos[:16], figs, opener, refs)
    elif v2: d = FR; nfr = render2(shots, dur, fam, style, seed, r, punches, bars, env, cuts, song, (walls, [opener] if opener else []), per)
    else: d, nfr = render(shots, dur, fam, style, set(cuts), strengths, seed, r, lines, punches, hook, env, bars, keyframes=cuts)
    mid = hashlib.sha1(f"{style}{track['id']}{seed}".encode()).hexdigest()[:20]; mp4 = os.path.join(WORK, mid + ".mp4")
    encode_moshed(d, nfr, cuts, mosh_cuts, mp4, track["path"], start, dur, fam)
    poster = os.path.join(WORK, mid + ".jpg"); subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{min(dur - .2, cuts[1] if len(cuts) > 1 else 0.5) + 0.08:.3f}", "-i", mp4, "-frames:v", "1", "-q:v", "4", poster], check=True)
    item = {"id": mid, "style": style, "fam": fam, "track": {"title": track["title"], "src": track["src"], "id": track["id"], "url": track["url"], "uploader": track.get("uploader", "")}, "start": round(start, 2), "dur": round(dur, 2), "cuts": len(cuts), "moshed": len(mosh_cuts),
            "line": next((s["line"] for s in shots if s.get("type") == "text"), None), "bpm": round(bpm, 1), "heroes": sorted({(h[0] + ":" + (h[1] if h[0] == "sym" else os.path.basename(h[1]))) for h in (heroes if v2 else [])}), "walls": sorted({s_["wall"]["id"] for s_ in shots if s_.get("wall")}) if v2 else [], "els": sorted({os.path.basename(s["fig"]) for s in shots if s.get("fig")} | {os.path.basename(e) for s in shots for e in s.get("extra", [])}), "opener": (shots[0].get("opener") or {}).get("img", {}).get("id") if v2 and shots[0].get("opener") else None, "made": int(time.time()), "bytes": os.path.getsize(mp4), "mp4": f"/atlas/mont/{mid}.mp4", "poster": f"/atlas/mont/{mid}.jpg"}
    obj_put(f"atlas/mont/{mid}.mp4", open(mp4, "rb").read(), "video/mp4"); obj_put(f"atlas/mont/{mid}.jpg", open(poster, "rb").read(), "image/jpeg")
    if not v2:
        for f in os.listdir(d): os.remove(os.path.join(d, f))
    return item

SHARD_NO, SHARD_OF = 0, 1
def make(cap, shard=0, of=1):
    global SHARD_NO, SHARD_OF; SHARD_NO, SHARD_OF = shard, of
    if os.environ.get("MONTAGE_OFFLINE"):                               # local test: elements already in _work/montage/els, no site
        els, man, flags, mont = [], {"styles": {}}, {}, {"items": []}; global obj_put; obj_put = lambda k, d, c=None: log("would put", k, len(d))
    else: els, man, flags, mont = sync_library()
    ws, wf = weights(mont, flags, "style"), weights(mont, flags, "fam"); dead = set(flags.get("mdead", [])); UPTASTE.update(uploader_taste(mont))
    used = {it["track"]["id"] for it in mont["items"] if it["id"] in dead} | {it["track"]["id"] for it in mont["items"][:12]} | {t["track"] for t in mont.get("taste", []) if t["v"] < 0 and t.get("why") in ("song", "both", None) and t.get("track")}
    counts = {s: len(C.pool(s)) for s in CFG if not s.startswith("_")}; recent = [it["style"] for it in mont["items"][:8]]
    styles = [s for s in counts if counts[s] >= 3 or len(man["styles"].get(s, [])) >= 20]   # thin on cut-outs is fine if the wall has images
    boost = MONT.get("boost", {})
    def w(s): return max(.15, (1 + .6 * ws.get(s, 0) + .3 * wf.get(fam_of(s), 0)) * math.log(2 + counts[s]) * (.35 if s in recent else 1)) * float(boost.get(s, 1))
    r = random.Random(int(os.environ.get("GITHUB_RUN_ID", 0)) or int(time.time())); picked = []   # same seed on every shard → same plan, each shard takes its slice
    while len(picked) < cap and styles:                                  # a style can come up twice; each repeat is less likely
        s = r.choices(styles, [w(x) * ((.6 if float(MONT.get("boost", {}).get(x, 1)) > 1 else .3) ** picked.count(x)) for x in styles])[0]; picked.append(s)
    if os.environ.get("MONTAGE_STYLES"): picked = os.environ["MONTAGE_STYLES"].split(",")
    seeds = [r.randrange(10 ** 6) for _ in picked]
    plan = list(zip(picked, seeds))[shard::of]
    log(f"shard {shard}/{of}:", [p_[0] for p_ in plan])
    made = []
    for s, seed in plan:
        try:
            it = make_one(s, els, man, flags, mont, used, seed)
            if it:
                made.append(it); used.add(it["track"]["id"]); report["made"].append(f"{s} · {it['track']['title']} ({it['track']['src']}) · {it.get('bpm')} bpm · {it['dur']}s · {it['cuts']} cuts/{it['moshed']} moshed · {it['bytes']//1024} KB")
                if of == 1: mont["items"].insert(0, it); obj_put("atlas/mont.json", json.dumps({"items": mont["items"]}).encode(), "application/json")
                else:
                    import shutil; od = os.path.join(WORK, "out"); os.makedirs(od, exist_ok=True)
                    for ext in ("mp4", "jpg"): shutil.copy(os.path.join(WORK, f"{it['id']}.{ext}"), os.path.join(od, f"{it['id']}.{ext}"))
        except Exception as e: err(s, e); report["errors"].append(traceback.format_exc()[-600:])
    if of > 1:                                                           # shard: hand the edits to the merge job, touch nothing shared
        od = os.path.join(WORK, "out"); os.makedirs(od, exist_ok=True)
        json.dump({"items": made, "report": report}, open(os.path.join(od, f"shard-{shard}.json"), "w")); return
    finish(mont, flags)

def finish(mont, flags):
    ok = set(flags.get("mok", [])) | set(flags.get("mdl", [])); dead = set(flags.get("mdead", [])); seen = flags.get("mseen") or {}; t = time.time()
    before = mont["items"]; keep, fresh = [], 0
    for it in before:
        i = it["id"]
        if i in ok: keep.append(it); continue
        if i in dead: continue
        if i in seen and t - seen[i] > WATCHED_TTL: continue                 # watched through, left alone: done with it
        if fresh < KEEP_ITEMS: keep.append(it); fresh += 1
    led = mont.setdefault("taste", []); have = {t["id"] for t in led}
    for it in before:                                                   # remember the verdict before the edit itself is gone
        v = 1 if it["id"] in ok else (-1 if it["id"] in dead else 0)
        if v and it["id"] not in have: led.append({"id": it["id"], "v": v, "why": (flags.get("mwhy") or {}).get(it["id"]), "style": it.get("style"), "fam": it.get("fam"), "src": (it.get("track") or {}).get("src"), "track": (it.get("track") or {}).get("id"), "uploader": (it.get("track") or {}).get("uploader", ""), "t": int(t)})
    mont["taste"] = led[-800:]
    gone = {it["id"] for it in before} - {it["id"] for it in keep}
    archive([it for it in before if it["id"] in gone and it["id"] not in dead], flags)   # seen ones go to the archive, small
    for i in gone:                                                          # free the full-size copy
        for ext in ("mp4", "jpg"): obj_del(f"atlas/mont/{i}.{ext}")
    report["pruned"] = len(gone); mont["items"] = keep
    obj_put("atlas/mont.json", json.dumps(mont).encode(), "application/json")
    report["total"] = len(mont["items"])
    try:
        if not os.environ.get("MONTAGE_OFFLINE"): publish(mont, flags)
    except Exception as e: err("publish", e)

ARCH_MAX = float(MONT.get("archive_gb", 2.5)) * 1e9       # with the image library and live edits this keeps storage under ~9 of the free 10 GB
def archive(items, flags):
    """keep every seen edit as a 720p copy (~2-4 MB) with its recipe, track and timing, for later use; oldest go past ARCH_MAX"""
    if not items: return
    idx = json.loads(obj_get("atlas/arch.json") or '{"items":[]}'); have = {a["id"] for a in idx["items"]}; added = 0
    for it in items:
        if it["id"] in have: continue
        data = obj_get(f"atlas/mont/{it['id']}.mp4")
        if not data: continue
        src = os.path.join(WORK, "arch_in.mp4"); dst = os.path.join(WORK, "arch_out.mp4"); open(src, "wb").write(data)
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", "scale=720:1280:flags=lanczos", "-c:v", "libx264", "-crf", "29", "-preset", "medium", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", dst], check=True, timeout=300)
            obj_put(f"atlas/arch/{it['id']}.mp4", open(dst, "rb").read(), "video/mp4")
            jp = obj_get(f"atlas/mont/{it['id']}.jpg")
            if jp: obj_put(f"atlas/arch/{it['id']}.jpg", jp, "image/jpeg")
            meta = {k: it.get(k) for k in ("id", "style", "fam", "track", "bpm", "dur", "start", "heroes", "walls", "made", "opener")}
            meta.update(bytes=os.path.getsize(dst), seen=(flags.get("mseen") or {}).get(it["id"]), why=(flags.get("mwhy") or {}).get(it["id"]), mp4=f"/api/atlas/arch/{it['id']}.mp4", poster=f"/api/atlas/arch/{it['id']}.jpg")
            idx["items"].insert(0, meta); added += 1
        except Exception as e: err("archive " + it["id"], e)
    total = 0; keep = []
    for a in idx["items"]:                                              # newest first; past the cap the oldest leave
        total += a.get("bytes", 0)
        if total <= ARCH_MAX: keep.append(a)
        else:
            for ext in ("mp4", "jpg"): obj_del(f"atlas/arch/{a['id']}.{ext}")
    idx["items"] = keep; obj_put("atlas/arch.json", json.dumps(idx).encode(), "application/json")
    report["archived"] = f"{added} added, {len(keep)} in the archive ({total / 1e9:.2f} GB)"

def cleanup(before):
    """one-off: remove every edit made before `before` (unix time) unless you kept or downloaded it; files deleted, list republished"""
    flags = json.loads(obj_get("atlas/flags.json") or "{}"); mont = json.loads(obj_get("atlas/mont.json") or '{"items":[]}')
    keep_ids = set(flags.get("mok", [])) | set(flags.get("mdl", []))
    gone = [it for it in mont["items"] if it.get("made", 0) < before and it["id"] not in keep_ids]
    for it in gone:
        for ext in ("mp4", "jpg"): obj_del(f"atlas/mont/{it['id']}.{ext}")
    mont["items"] = [it for it in mont["items"] if it not in gone]
    obj_put("atlas/mont.json", json.dumps(mont).encode(), "application/json"); report["total"] = len(mont["items"])
    report["made"].append(f"cleanup: removed {len(gone)} old edits, {len(mont['items'])} left")
    publish(mont, flags)

def merge(dirs):
    """after the shards: collect their edits (mp4/jpg already in R2 and in the artifact dirs), update mont.json once, publish once"""
    import shutil
    flags = json.loads(obj_get("atlas/flags.json") or "{}"); mont = json.loads(obj_get("atlas/mont.json") or '{"items":[]}'); new = []
    for d_ in dirs:
        for f in os.listdir(d_):
            if f.startswith("shard-") and f.endswith(".json"):
                j = json.load(open(os.path.join(d_, f))); new += j["items"]
                report["made"] += j["report"].get("made", []); report["skipped"] += j["report"].get("skipped", []); report["errors"] += j["report"].get("errors", [])
            elif f.endswith((".mp4", ".jpg")): shutil.copy(os.path.join(d_, f), os.path.join(WORK, f))
    new.sort(key=lambda it: -it.get("made", 0)); seen_t = set(); uniq = []
    for it in new:
        tid = it.get("track", {}).get("id")
        if tid in seen_t: continue
        seen_t.add(tid); uniq.append(it)
    mont["items"] = uniq + mont["items"]
    finish(mont, flags)

def publish(mont, flags):
    """Put tonight's edits on the site without waiting for the curator: tar parts atlas/static-9NN.tar appended to atlas/static.json.
    The atlas deploy (triggered by this job's deploy hook) unpacks them; the curator's next bundle folds them in and drops these parts."""
    import tarfile
    idx = json.loads(obj_get("atlas/static.json") or '{"parts":[]}'); built = idx.get("built", 0)
    ok = set(flags.get("mok", [])); items = []
    for it in mont["items"]:
        it = dict(it); it["ok"] = 1 if it["id"] in ok else 0; items.append(it)
    fresh = []
    for it in items:                                                    # everything made since the last full bundle, not just this batch
        if it.get("made", 0) <= built - 60: continue
        for ext in ("mp4", "jpg"):
            fp = os.path.join(WORK, f"{it['id']}.{ext}")
            if not os.path.exists(fp):
                data = obj_get(f"atlas/mont/{it['id']}.{ext}")
                if data: open(fp, "wb").write(data)
        if os.path.exists(os.path.join(WORK, it["id"] + ".mp4")) and os.path.exists(os.path.join(WORK, it["id"] + ".jpg")): fresh.append(it)
    files = [("mont.json", json.dumps({"items": items}).encode())]
    for it in fresh:
        files.append((f"mont/{it['id']}.mp4", open(os.path.join(WORK, it["id"] + ".mp4"), "rb").read()))
        files.append((f"mont/{it['id']}.jpg", open(os.path.join(WORK, it["id"] + ".jpg"), "rb").read()))
    parts, buf, tf, size = [], None, None, 0
    def close():
        nonlocal buf, tf, size
        if tf is None: return
        tf.close(); name = f"atlas/static-{901 + len(parts)}.tar"; obj_put(name, buf.getvalue(), "application/x-tar"); parts.append({"key": name, "bytes": len(buf.getvalue())}); buf = tf = None; size = 0
    for arc, data in files:
        if tf is not None and size + len(data) > 40e6: close()
        if tf is None: buf = io.BytesIO(); tf = tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT)
        ti = tarfile.TarInfo(arc); ti.size = len(data); ti.mtime = int(time.time()); tf.addfile(ti, io.BytesIO(data)); size += len(data) + 512
    close()
    idx = json.loads(obj_get("atlas/static.json") or '{"parts":[]}')          # re-read: the curator may have written meanwhile
    idx["parts"] = [p for p in idx.get("parts", []) if not re.match(r"atlas/static-9\d\d\.tar$", p["key"])] + parts
    obj_put("atlas/static.json", json.dumps(idx).encode(), "application/json")
    report["published"] = f"{len(fresh)} new edits in {len(parts)} part(s); the site picks them up from this run's deploy"; log("publish:", report["published"])

# ============================ probe ============================
def probe():
    """what the viral pool would pick right now: top candidates per query with plays/day"""
    vir = MONT["music"]["_viral"]; want = [w.lower() for w in MONT["music"]["_techno"].get("want", [])]
    for q in vir["sc"][:27]:
        try:
            cands = sc_candidates(q, 12); cands.sort(key=lambda c: -score_viral(c, q, want, vir))
            report["probe"].append({"q": q, "top": [f"{c['title'][:55]} | {c['plays']} plays | {int(c['plays'] / c['age']) if c.get('age') else '?'}/day | {int(c['age']) if c.get('age') else '?'}d → {score_viral(c, q, want, vir):.2f}" for c in cands[:4]]})
        except Exception as e: report["probe"].append({"q": q, "err": str(e)[:160]})

def write_report():
    L = [f"# montage — {report['started']}", ""]
    for r in report["probe"]: L.append(f"- `{r.get('q')}` → " + " ".join(f"{k}={str(v)[:160]}" for k, v in r.items() if k != "q"))
    if report["made"]: L += ["", f"## made ({len(report['made'])}, {report.get('total', '?')} on the site)"] + [f"- {m}" for m in report["made"]]
    if report["skipped"]: L += ["", "## skipped"] + [f"- {m}" for m in report["skipped"]]
    if report["errors"]: L += ["", "## errors"] + [f"- {e}" for e in report["errors"]]
    open(os.path.join(HERE, "last-montage.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--probe", action="store_true"); ap.add_argument("--cap", type=int, default=6)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--of", type=int, default=1); ap.add_argument("--merge", nargs="*"); ap.add_argument("--cleanup", type=int, default=0); a = ap.parse_args()
    try:
        if a.cleanup: cleanup(a.cleanup)
        elif a.probe: probe()
        elif a.merge is not None: merge(a.merge)
        else: make(a.cap, a.shard, a.of)
    except Exception: report["errors"].append(traceback.format_exc()[-900:])
    write_report()
