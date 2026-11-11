---
license: other
license_name: metadata-only
viewer: true
configs:
  - config_name: default
    data_files: meta.jsonl
tags: [beat-saber, rhythm-game, music, beatsaver]
size_categories: [1K<n<10K]
---

# BeatGen maps index

Index of the Beat Saber maps used to train [BeatGen](https://github.com/PGSCOM/BeatGen). It contains
references only, without map content or audio: the maps are owned by their authors and BeatSaver's
terms of service do not permit redistribution, so they must be retrieved directly from
[BeatSaver](https://beatsaver.com).

6303 maps:

- 2466 ranked maps with a rating of at least 0.6.
- 3837 unranked maps with a rating of at least 0.85 and at least 50 votes.

Automapped levels, levels declared as AI-generated and levels using Noodle or Mapping Extensions
are excluded.

## Rebuilding the dataset

```bash
git clone https://github.com/PGSCOM/BeatGen && cd BeatGen
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
hf download PGSCOM/beatgen-maps-index meta.jsonl --type dataset --local-dir .
python download.py --from meta.jsonl   # data/raw/<id>.zip, the exact version by hash (~45 GB)
python prepare.py                      # data/proc/<id>.npz (5408 usable maps)
```

If a mapper publishes a new version of a map, the indexed version may become unavailable and that
download will fail.

## Fields

| Field | Content |
|---|---|
| `id` | BeatSaver id (`https://beatsaver.com/maps/<id>`) |
| `hash` | Hash of the map version used |
| `name`, `song`, `artist`, `mapper` | Map title, song, artist and mapper |
| `bpm`, `rating`, `upvotes`, `tags` | BeatSaver data at download time |
| `ranked`, `blRanked` | Ranked on ScoreSaber / BeatLeader |
| `uploaded` | Upload date |
| `diffs` | Difficulties in the Standard characteristic |

Metadata was retrieved from the public BeatSaver API.
