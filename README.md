# Afterimage Edits

The edit maker behind the Edits tab of [Afterimage Atlas](https://atlas.symbiozefpv.com), by SYMBIOZE ([@fpv.symbioze](https://instagram.com/fpv.symbioze)).

Each run picks a techno track, finds its beat grid, and cuts a 9:16 edit from the atlas library: real photos and cut-outs, vector symbols, GLSL shader backdrops, ASCII passes and datamosh transitions, all locked to the kick.

- `edits/montage.py`: the engine (`--cap N`, `--shard i --of n`, `--merge dirs...`, `--probe`)
- `edits/montage.json`: families, music queries, captions and look settings. Edit freely.
- `edits/art/`: shaders (`vj.py`), symbols, collage and glitch helpers
- `.github/workflows/edits.yml`: 5 parallel render shards, then one merge job that publishes to the atlas

No audio or video is stored in this repository. Fonts in `edits/art/fonts/ttf` are from Google Fonts under the SIL Open Font License.
