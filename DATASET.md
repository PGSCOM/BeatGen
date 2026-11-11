---
license: other
license_name: beatsaver-tos
license_link: https://beatsaver.com/policy/tos
viewer: false
tags: [beat-saber, rhythm-game, music, audio]
---

# BeatGen maps (privado)

Datos de entrenamiento de [BeatGen](https://github.com/PGSCOM/BeatGen): mapas ranked de BeatSaver
(rating ≥ 0.75, sin automappers ni mapas declarados como IA).

| Ruta | Contenido |
|---|---|
| `raw/<id>.zip` | Mapa tal cual se descarga de BeatSaver (audio + Info.dat + dificultades). |
| `raw/meta.jsonl` | Una línea por mapa: id, canción, artista, mapper, BPM, rating, tags, ranked, hash, dificultades. |
| `proc/<id>.npz` | Salida de `prepare.py`: `feats` (log-mel por paso de 1/4 de beat, float16), `bpm` y tokens por dificultad. |

## Por qué es privado

- Los mapas son obra de sus mappers. Los términos de BeatSaver exigen permiso explícito por escrito
  para republicar o usar contenido de otro usuario, incluidas obras derivadas.
- El audio de los zips es música con copyright.

Este repositorio es solo copia de trabajo para entrenar. No se redistribuye.
