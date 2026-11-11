# BeatGen

Generador de mapas de Beat Saber con IA: le das una canción y devuelve un zip listo para
`CustomLevels`. A diferencia de Beat Sage o DeepSaber, el mapa es **controlable**: tiene una
intensidad distinta en verso y estribillo, y sigue la estructura de la canción.

```bash
uv venv --python 3.12 .venv && uv pip install -r requirements.txt
python download.py --max 4000 --min-rating 0.6   # mapas ranked de BeatSaver → data/raw (~8 MB/mapa)
python download.py --from meta.jsonl   # o reproducir el dataset publicado (mismas versiones)
python prepare.py                      # → data/proc/*.npz (features por paso + tokens)
hf upload PGSCOM/beatgen-maps data . --type dataset --private      # copia en HF (privada)
hf upload PGSCOM/beatgen-maps DATASET.md README.md --type dataset
hf upload PGSCOM/beatgen-maps-index data/raw/meta.jsonl meta.jsonl --type dataset   # índice público
hf upload PGSCOM/beatgen-maps-index DATASET_PUBLIC.md README.md --type dataset
python train.py --steps 50000          # → ckpt.pt (usa GPU si hay)
python generate.py cancion.mp3 --diffs Hard Expert
python generate.py cancion.mp3 --curve "0:0.5,30:2,75:1"   # intensidad manual (seg:notas/beat)
```

## Cómo funciona

```
audio ─► BPM + fase ─► rejilla (4 pasos/beat) ─► log-mel por paso ─┐
                                                                   ├─► Transformer ─► tokens ─► paridad ─► zip
dificultad + intensidad por paso (secciones o --curve) ────────────┘
```

| Pieza | Archivo | Detalle |
|---|---|---|
| Descarga | `download.py` | Mapas ranked de BeatSaver ordenados por rating; excluye automappers y los declarados como IA. |
| Rejilla | `beatmap.estimate_grid` | Peine armónico sobre los onsets. En 36 mapas ranked, el BPM acierta siempre (salvo octava) y la fase queda a menos de 15 ms en el 83 % de los casos. |
| Tokens | `beatmap.py` | Por paso y mano: 0 = nada, o una de 108 notas (columna × fila × dirección). |
| Modelo | `model.py` | Encoder: audio + dificultad + intensidad. Decoder causal: notas de las dos manos. |
| Control | `generate.py` | La intensidad (notas/beat, 8 niveles) sale de los propios mapas al entrenar. Al generar, se calcula a partir de las secciones de la canción (clustering + energía) o de `--curve`. |
| Jugabilidad | `beatmap.fix_parity` | Alterna forehand/backhand en cada mano y quita los choques entre manos. |

## Estado

**Hecho:** pipeline completo, desde la descarga hasta el zip, probado de punta a punta con 40
mapas. El modelo solo se ha entrenado para comprobar que aprende (300 pasos, 36 canciones, CPU):
la loss baja, pero todavía no respeta la intensidad pedida.

## Plan

Cada fase termina con una condición medible. Mientras no se cumpla, no se pasa a la siguiente.

### Fase 1: dataset

- Descargar unos 3000 mapas ranked (~21 GB) y procesarlos con `prepare.py`.
- Subir `data/` (zips, `meta.jsonl` y `.npz`) al dataset privado
  [PGSCOM/beatgen-maps](https://huggingface.co/datasets/PGSCOM/beatgen-maps). No puede ser
  público (ver `DATASET.md`). La versión pública,
  [PGSCOM/beatgen-maps-index](https://huggingface.co/datasets/PGSCOM/beatgen-maps-index), solo
  lleva `meta.jsonl` con ids y hashes; los mapas se descargan con `download.py --from`.
- Con rating ≥ 0.75 solo hay 2113 mapas ranked (1753 útiles). Con ≥ 0.6 hay 2466, casi todo el
  catálogo ranked, y salen **2006** útiles. El 19 % se descarta, sobre todo por cambios de BPM.
- Recalcular sobre el dataset completo: el `% descartados` (por cambios de BPM o Info v4),
  el `% notas conservadas`, los rangos `NPB` por dificultad y `ONSET_LATENCY`.
- Evaluar la rejilla con canciones que **no** se hayan usado para calibrar.

**Listo cuando:** haya ≥ 2500 canciones procesadas y la fase quede a < 15 ms en ≥ 80 % del conjunto de validación.

### Fase 2: modelo base

- Entrenar en GPU; en CPU no es viable. En otra máquina Ubuntu con GPU:
  ```bash
  git clone https://github.com/PGSCOM/BeatGen && cd BeatGen
  bash gpu.sh train --steps 50000      # venv + .npz de HF + train.py + sube ckpt.pt a PGSCOM/beatgen-ckpt
  bash gpu.sh clean && cd .. && rm -rf BeatGen
  ```
  Después, en local: `hf download PGSCOM/beatgen-ckpt ckpt.pt --local-dir .`
- Crear `eval.py`, que mida sobre las canciones de validación:
  - **Timing:** F1 entre las notas generadas y las del mapa real (tolerancia ±1 paso).
  - **Control:** error entre las notas/beat pedidas y las generadas, sección a sección.
  - **Flujo:** cuántas notas tiene que corregir `fix_parity`; cuantas menos, mejor ha aprendido el modelo.
- Compararlo con una referencia tonta: una nota en cada pico de onset con direcciones alternas.

**Listo cuando:** supere a la referencia en F1 y el error de intensidad sea < 20 %.

### Fase 3: jugabilidad

- Probar mapas en el juego, en Hard y en Expert, y apuntar qué patrones fallan.
- Mejorar la paridad: posición de la muñeca, resets, brazos cruzados y notas que tapan la visión.
- Añadir bombas para los resets y muros básicos.
- **Crítico entrenado con replays de BeatLeader:** sustituir las reglas escritas a mano por
  lo que hacen los jugadores de verdad.
  - Descargar los replays (`.bsor`) de los mapas ranked, limitados a unos 50 por mapa, y
    revisar antes los términos de uso y los límites de peticiones de la API.
  - Sacar etiquetas por nota (tasa de fallo, precisión, ángulos de swing) y alinearlas con los tokens.
  - Entrenar un modelo pequeño que, a partir del contexto de notas anteriores, prediga la
    probabilidad de fallo y la precisión de la siguiente.
  - Normalizar por el nivel del jugador (pp) y por la densidad. Lo que hay que aprender son
    los fallos que sobran respecto a lo esperado, no los fallos sin más: los mapas ranked son
    difíciles a propósito.
  - Usarlo, de menos a más trabajo:
    1. Generar varias versiones de un mapa y quedarse con la más jugable para su dificultad.
    2. Dar menos peso al entrenar a los tramos que se juegan peor de lo esperado.
    3. Afinar el generador con el crítico como recompensa (aprendizaje por refuerzo).
  - Limitación: solo los mapas ranked y populares tienen replays suficientes, así que el
    crítico solo conocerá ese estilo.

**Listo cuando:** se puedan jugar 10 canciones distintas de principio a fin sin ningún patrón
imposible, y elegir el mapa con el crítico baje la tasa de fallo en las pruebas de juego
frente a quedarse con el primero generado.

### Fase 4: control y estructura

- Que las repeticiones se reconozcan: el mismo estribillo debería generar patrones parecidos.
- Control por sección desde la línea de comandos (listar las secciones detectadas y editarlas),
  además de `--curve`.
- Condicionar por estilo usando los tags de BeatSaver (tech, speed, dance…).

**Listo cuando:** dos estribillos iguales den patrones reconocibles como el mismo, y el estilo cambie el mapa de forma visible.

### Fase 5: más datos y más resolución

- Mapas curated o con muy buen rating aunque no sean ranked.
- Soporte de Info v4 y de cambios de BPM.
- Mapas oficiales convertidos al formato custom (revisar antes la licencia).
- Subir la rejilla a 1/8 o 1/12 para streams y tresillos.
- Aumentar el dataset con espejo izquierda/derecha.
- Probar embeddings de un modelo de audio preentrenado en lugar del log-mel.

**Listo cuando:** cada cambio mejore las métricas de `eval.py`; si no las mejora, no entra.

### Fase 6: producto

- Generar en un solo paso (canción → zip), con un `--preview` que muestre la curva de intensidad.
- Si se publica en BeatSaver, marcar los mapas como generados por IA, como exige la plataforma.

## Limitaciones actuales

Están marcadas con `ponytail:` en el código:

- BPM constante: los mapas con cambios de BPM se descartan.
- No se lee Info v4.
- Sin bombas, muros ni sliders.
- Una sola nota por mano y paso.
- Los tresillos se redondean a 1/4 de beat (se conserva el 96 % de las notas).
