#!/usr/bin/env python3
"""Lab: render one sample edit per treatment from the real atlas, without publishing anything.
Writes edits/_work/lab/<treatment>.mp4 (720p) and <treatment>.jpg (a contact sheet) for review.

    TREAT=monolith python edits/lab.py
"""
import os, sys, json, random, subprocess, shutil, traceback
os.environ["MONTAGE_LAB"] = "1"
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import montage as M

treat = os.environ.get("TREAT", "monolith"); seed = int(os.environ.get("SEED", "4242"))
out = os.path.join(M.HERE, "_work", "lab"); os.makedirs(out, exist_ok=True)
els, man, flags, mont = M.sync_library()
styles = [s for s in M.CFG if not s.startswith("_") and len(man["styles"].get(s, [])) >= 20 and len(M.C.pool(s)) >= 3]
r = random.Random(seed); note = {"treat": treat}
for attempt in range(4):
    style = r.choice(styles)
    try:
        it = M.make_one(style, els, man, flags, mont, set(), seed + attempt)
    except Exception:
        note.setdefault("errors", []).append(traceback.format_exc()[-800:]); it = None
    if it:
        src = os.path.join(M.WORK, it["id"] + ".mp4"); dst = os.path.join(out, treat + ".mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", "scale=720:1280", "-c:v", "libx264", "-crf", "24", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", dst], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", "fps=1.5,scale=216:384,tile=10x3", "-frames:v", "1", os.path.join(out, treat + ".jpg")], check=True)
        note.update(style=style, track=it["track"]["title"], bpm=it.get("bpm"), dur=it.get("dur")); break
note["errors"] = note.get("errors", []) + M.report.get("errors", [])
json.dump(note, open(os.path.join(out, treat + ".json"), "w"), indent=1)
print(json.dumps(note, indent=1)[:3000])
