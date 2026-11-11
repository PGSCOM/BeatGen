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

Lista de los 6303 mapas de Beat Saber con los que se entrenó
[BeatGen](https://github.com/PGSCOM/BeatGen), sacados de [BeatSaver](https://beatsaver.com):

- 2466 mapas ranked con rating ≥ 0.6.
- 3837 mapas no ranked con rating ≥ 0.85 y al menos 50 votos.

Se excluyen los automappers, los mapas declarados como IA y los que usan Noodle o Mapping
Extensions. De los 6303, `prepare.py` aprovecha 5408; el resto tiene cambios de BPM o formato Info v4.

**Aquí no hay mapas ni audio, solo referencias.** Los mapas son de sus mappers y la música, de
sus autores. Los términos de BeatSaver no permiten redistribuirlos sin permiso, así que cada
uno se los descarga de BeatSaver.

## Reproducir el dataset

```bash
git clone https://github.com/PGSCOM/BeatGen && cd BeatGen
uv venv --python 3.12 .venv && uv pip install -r requirements.txt
hf download PGSCOM/beatgen-maps-index meta.jsonl --type dataset --local-dir .
python download.py --from meta.jsonl   # → data/raw/<id>.zip (la versión exacta, por hash)
python prepare.py                      # → data/proc/<id>.npz
bash gpu.sh train --d 384 --layers 6 --steps 100000   # entrena (en una máquina con GPU)
```

Si un mapper sube una versión nueva de su mapa, la versión antigua puede dejar de estar
disponible y esa descarga fallará. El resto no se ve afectado.

## Campos de `meta.jsonl`

| Campo | Contenido |
|---|---|
| `id` | Id de BeatSaver (`https://beatsaver.com/maps/<id>`) |
| `hash` | Hash de la versión usada |
| `name`, `song`, `artist`, `mapper` | Título del mapa, canción, artista y mapper |
| `bpm`, `rating`, `upvotes`, `tags` | Datos de BeatSaver en el momento de la descarga |
| `ranked`, `blRanked` | Ranked en ScoreSaber / BeatLeader |
| `uploaded` | Fecha de subida |
| `diffs` | Dificultades de la característica Standard |

Los metadatos proceden de la API pública de BeatSaver.
