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

The list of Beat Saber maps used to train [BeatGen](https://github.com/PGSCOM/BeatGen). It holds
references only: no maps and no audio. The maps belong to their mappers and BeatSaver's terms don't
allow redistributing them, so you download them yourself from [BeatSaver](https://beatsaver.com).

6303 maps:

- 2466 ranked maps rated 0.6 or higher.
- 3837 unranked maps rated 0.85 or higher, with at least 50 votes.

Automapped maps, maps declared as AI-made and maps using Noodle or Mapping Extensions are excluded.

## Rebuilding the dataset

```bash
git clone https://github.com/PGSCOM/BeatGen && cd BeatGen
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
hf download PGSCOM/beatgen-maps-index meta.jsonl --type dataset --local-dir .
python download.py --from meta.jsonl   # data/raw/<id>.zip, the exact version by hash (~45 GB)
python prepare.py                      # data/proc/<id>.npz (5408 usable maps)
```

If a mapper uploads a new version of a map, the old one may no longer be available and that
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

Metadata comes from the public BeatSaver API.
