# BeatGen

> **Proyecto terminado (noviembre de 2026).** Funciona de punta a punta y genera mapas que se pueden
> jugar, pero no llega a la calidad de un mapa hecho a mano. Este README resume cómo está montado,
> qué se consiguió y qué quedó pendiente.

Generador de mapas de Beat Saber con IA. Le das una canción y devuelve un zip listo para
`CustomLevels`. La diferencia con Beat Sage o DeepSaber es que la intensidad se puede **controlar**:
el mapa es más suave en los versos y más cargado en los estribillos, de forma automática a partir de
la estructura de la canción o a mano con una curva.

## Resultado

- **Medido** (`eval.py`, 60 dificultades de canciones que el modelo no vio al entrenar): el modelo
  supera a una referencia basada en los picos del audio en precisión de timing y cumple el objetivo
  de intensidad.
- **Jugado** ("For Real" de Pharrell Williams, Hard y Expert, en unas Pico 4): la cantidad de notas
  está bien y el mapa se puede jugar, pero cuesta seguirlo. Le falta el flujo y la intención de un
  mapper humano. En versiones anteriores aparecían patrones imposibles y bucles a mitad de canción;
  ambos se corrigieron (ver [Problemas encontrados](#problemas-encontrados)).

| | F1 exacto | F1 ±1 paso | error de intensidad | notas que corrige la paridad |
|---|---|---|---|---|
| **Modelo final** (d=384, 6 capas) | **0.66** | **0.74** | **17 %** | 8 % |
| Modelo pequeño (d=256, 4 capas) | 0.59 | 0.71 | 18 % | 5 % |
| Referencia: notas en los picos del audio | 0.39 | 0.72 | — | — |
| Mapas reales | — | — | — | 4.7 % |

La referencia lo tiene fácil a propósito: pone en cada bloque de 4 compases exactamente el mismo
número de notas que el mapa real.

## Uso

El modelo entrenado no se publica (ver [Datos y licencias](#datos-y-licencias)), así que hay que
reproducir el dataset y entrenarlo:

```bash
git clone https://github.com/PGSCOM/BeatGen && cd BeatGen
uv venv --python 3.12 .venv && uv pip install -r requirements.txt
hf download PGSCOM/beatgen-maps-index meta.jsonl --type dataset --local-dir .
python download.py --from meta.jsonl    # 6303 mapas de BeatSaver, las mismas versiones (~45 GB)
python prepare.py                       # → data/proc/*.npz (5408 canciones, ~2 GB)
bash gpu.sh train --d 384 --layers 6 --steps 100000   # en una máquina con GPU NVIDIA (~75 min)
```

Para generar un mapa:

```bash
python generate.py cancion.mp3 --diffs Hard Expert              # → cancion_beatgen.zip
python generate.py cancion.mp3 --curve "0:0.5,30:2,75:1"         # intensidad a mano (segundo:notas/beat)
python generate.py cancion.mp3 --bpm 128                         # si la detección del BPM falla
```

Genera en CPU en ~1 minuto por dificultad. Para jugarlo, descomprime el zip en
`Beat Saber_Data/CustomLevels/<carpeta>/` (PC) o en `ModData/<juego>/Mods/SongCore/CustomLevels/`
(Quest y Pico con mods).

Opciones del muestreo: `--temp` (1.0 por defecto; por debajo de 1 aparecen bucles) y `--rep`
(0.3 por defecto: reduce la probabilidad de repetir un motivo reciente; 1 lo desactiva).

## Cómo funciona

```
BeatSaver ─► download.py ─► data/raw/*.zip ─► prepare.py ─► data/proc/*.npz ─► train.py (GPU) ─► ckpt.pt
                                                                                                   │
canción.mp3 ─► BPM + fase ─► rejilla ─► audio por paso ─┐                                          │
                                                        ├─► Transformer ◄──────────────────────────┘
dificultad + intensidad (secciones o --curve) ──────────┘        │
                                                                 ▼
                                    tokens ─► notas ─► descruzar + paridad ─► zip
```

### 1. Datos (`download.py`)

Mapas de BeatSaver ordenados por valoración:

- 2466 **ranked** con rating ≥ 0.6, que son prácticamente todo el catálogo ranked.
- 3837 **no ranked** con rating ≥ 0.85 y al menos 50 votos.

Se excluyen los automappers, los declarados como IA y los que usan Noodle o Mapping Extensions,
porque sacan notas de la rejilla. En total son 6303 mapas (45 GB), con sus metadatos en `meta.jsonl`
(canción, mapper, rating, tags, hash de la versión). `download.py --from meta.jsonl` vuelve a
descargar exactamente las mismas versiones.

### 2. Representación (`beatmap.py`, `prepare.py`)

Cada canción se convierte en dos secuencias alineadas sobre una **rejilla de 1/4 de beat**:

- **Audio:** espectrograma log-mel de 80 bandas, promediado en cada paso.
- **Notas:** un token por mano y paso. 0 = nada; 1–108 = columna (4) × fila (3) × dirección (9).

Se descartan los mapas con cambios de BPM o en formato Info v4 (14 %). Quedan **5408 canciones**,
y al encajar las notas en la rejilla se conserva el 94 % de ellas.

### 3. Modelo (`model.py`)

Transformer encoder-decoder que "traduce" audio a notas:

- **Encoder:** audio por paso + posición dentro del compás + dificultad + intensidad (notas/beat
  en 8 niveles, constante por bloques de 4 compases).
- **Decoder:** genera paso a paso las notas de las dos manos, mirando las que ya ha puesto. Así
  aprende patrones de flujo.

La intensidad no hay que etiquetarla: al entrenar se calcula contando las notas del propio mapa.

### 4. Entrenamiento (`train.py`, `gpu.sh`)

Ventanas aleatorias de 64 beats alineadas al compás, AdamW con warmup y coseno, bf16 en GPU. El
modelo final (d=384, 6 capas, 100k pasos) tardó 75 minutos en GPU. `gpu.sh` lo hace todo en otra
máquina y la deja limpia después:

```bash
bash gpu.sh train --d 384 --layers 6 --steps 100000   # venv + entrenar sobre data/proc + subir ckpt a HF
CKPT=otro.pt bash gpu.sh train ...                    # con otro nombre, sin pisar ckpt.pt
bash gpu.sh clean                                     # borra venv, datos, cachés y token de HF
```

### 5. Generación (`generate.py`)

1. **Rejilla:** BPM y fase con un peine armónico sobre la envolvente de onsets: primero una búsqueda
   gruesa por ventanas de 10 s y después una fina sobre toda la canción.
2. **Intensidad:** divide la canción en 8 secciones (clustering de librosa), las ordena por energía
   y reparte entre el mínimo y el máximo de notas/beat de cada dificultad. Los silencios se quedan
   en 0. Con `--curve` se fija a mano.
3. **Muestreo:** autorregresivo en ventanas solapadas, con temperatura 1.0 y un freno a repetir
   secuencias de 4 notas recientes.
4. **Jugabilidad:** intercambia las columnas de las notas simultáneas con los brazos cruzados,
   alterna cada mano entre forehand y backhand (tras 2 beats de pausa puede retomar en cualquier
   sentido) y quita los choques en la misma celda.
5. **Exportación:** `Info.dat` v2, un `.dat` por dificultad, el audio original en OGG y una portada.

### 6. Medición (`calib.py`, `eval.py`)

- `calib.py` ajusta las constantes con el dataset: rangos de notas/beat por dificultad (`NPB`) y
  el retraso de los onsets (`ONSET_LATENCY` = 36 ms). Calibra con una mitad de las canciones y
  evalúa con la otra. La rejilla acierta el BPM en el 97 % de las canciones (salvo octava) y deja
  la fase a menos de 15 ms en el 63 % (error mediano: 8 ms).
- `eval.py` compara lo generado con los mapas reales en canciones de validación: F1 de timing,
  error de intensidad, notas que corrige la paridad y brazos cruzados, frente a la referencia.

| Archivo | Qué hace |
|---|---|
| `download.py` | Descarga mapas y metadatos de BeatSaver, o reproduce un `meta.jsonl`. |
| `beatmap.py` | Formatos de Beat Saber, tokens, audio, rejilla, paridad y exportación (`python beatmap.py` ejecuta su autotest). |
| `prepare.py` | `data/raw/*.zip` → `data/proc/*.npz`. |
| `model.py` | El Transformer. |
| `train.py` | Entrenamiento. |
| `generate.py` | Canción → zip. |
| `calib.py` | Calibración y evaluación de la rejilla sobre el dataset. |
| `eval.py` | Evaluación del modelo frente a mapas reales. |
| `gpu.sh` | Entrenar en otra máquina con GPU sin dejar rastro. |

## Problemas encontrados

Por si se retoma, estos son los errores que más costó ver:

- **Fuga en la intensidad.** Al principio se calculaba con una ventana deslizante de 4 beats, y su
  valor cambiaba justo en los pasos donde había notas. El modelo aprendió a leer las posiciones
  ahí en vez de escuchar el audio: daba un F1 de 0.99 en evaluación y no servía con canciones
  nuevas. Ahora es constante por bloques de 4 compases.
- **`fix_parity` estropeaba mapas buenos.** Tras cada pausa obligaba a empezar con forehand y
  volteaba cadenas enteras: corregía el 39 % de las notas de mapas *reales*. Ahora corrige el 4.7 %.
- **Bucles a mitad de canción.** Cada ventana se condiciona en las notas que el propio modelo puso
  antes. Con temperatura 0.9, un motivo repetido se reforzaba solo, hasta un 49 % de secuencias de
  4 notas repetidas hacia el minuto 2:30 (en mapas reales, ~12 %). El principio de la canción salía
  bien porque aún no había contexto. Se arregló con temperatura 1.0 y el freno `--rep`.
- **El techo de la rejilla.** beat_this, un detector de beats neuronal reciente, no mejora al peine
  armónico (56.5 % frente a 58.7 % de fase < 15 ms). Probablemente el límite está en la referencia:
  no todos los mappers alinean la rejilla con el beat real. Un 13 % de los fallos cae a medio beat.

## Lo que quedó pendiente

- **Jugabilidad:** reglas por posición de la muñeca, notas que tapan la visión, bombas para resets
  y muros.
- **Crítico con replays de BeatLeader:** un modelo que prediga a partir de partidas reales qué
  notas se fallan más de lo esperado. Serviría para elegir la versión más jugable de un mapa o como
  recompensa para afinar el generador. Requiere revisar antes los términos de la API de BeatLeader.
- **Estructura:** que dos estribillos iguales den patrones parecidos, y condicionar por estilo
  usando los tags de BeatSaver (tech, speed, dance…).
- **Datos:** soportar cambios de BPM e Info v4 (recuperaría el 14 % descartado), una rejilla de
  1/8 o 1/12 para streams y tresillos, aumentar el dataset con espejo izquierda/derecha y probar
  embeddings de un modelo de audio preentrenado.
- **Rejilla:** decidir entre la fase φ y φ + medio beat con los onsets de graves (bombo).

Limitaciones actuales, marcadas con `ponytail:` en el código: BPM constante, sin Info v4, sin
bombas, muros ni sliders, una nota por mano y paso, y los tresillos se redondean a 1/4 de beat.

## Datos y licencias

| Qué | Dónde | Acceso |
|---|---|---|
| Código | GitHub [`PGSCOM/BeatGen`](https://github.com/PGSCOM/BeatGen) | Público, licencia MIT |
| Índice: ids, hashes y metadatos de los 6303 mapas | HF [`PGSCOM/beatgen-maps-index`](https://huggingface.co/datasets/PGSCOM/beatgen-maps-index) | Público |
| Mapas, audio y `.npz` | — | No se publican; cada uno los descarga y procesa |
| Modelo entrenado | — | No se publica |

### Por qué no se publican el modelo ni los datos

El código es nuestro y va con licencia MIT. Pero todo lo que sale de los mapas pertenece a otros:

- **Los mapas son de sus mappers.** Los [términos de BeatSaver](https://beatsaver.com/policy/tos)
  exigen permiso explícito por escrito para *"republishing or use of another user's content posted
  on BeatSaver including derivative work"*, y aclaran que dar crédito no basta. Redistribuir los
  zips, aunque sea dentro de un dataset, es justo eso.
- **El audio tiene copyright.** Los zips llevan la canción completa, y quien sube un mapa a
  BeatSaver declara tener sus derechos de distribución; quien lo descarga no los recibe.
- **Los `.npz` tampoco son publicables.** Los tokens son una transcripción directa de las notas de
  cada mapa, es decir, una obra derivada.
- **El modelo es zona gris, y por eso no se publica.** Sus pesos son el resultado de entrenar con
  mapas de otros sin su permiso. Los términos de BeatSaver no hablan de entrenar modelos, pero
  prohíben "usar" el contenido de otros usuarios, incluidas obras derivadas, y un modelo capaz de
  imitar su estilo puede considerarse una. Además, buena parte de la comunidad de mappers está en
  contra de los generadores automáticos: BeatSaver obliga a declararlos y los separa del resto.
  Publicarlo sin preguntar sería usar su trabajo de una forma que no han autorizado. Para hacerlo
  habría que pedir permiso a BeatSaver o, mejor, a los mappers.
- **El índice sí se publica.** Solo lleva ids, hashes y metadatos de la API pública, sin contenido.
  Con él y `download.py --from`, cualquiera puede reconstruir el mismo dataset descargándolo él
  mismo de BeatSaver, que es lo que permiten sus términos.

El dataset privado con los zips y los `.npz` se borró al terminar el proyecto: se puede reconstruir
desde el índice. En una comprobación del 12/11/2026, BeatSaver seguía sirviendo el 100 % de una
muestra de 300 versiones exactas. Con el tiempo se pueden perder algunas, si un mapper sube una
versión nueva de su mapa.

Si algún mapa generado se publica en BeatSaver, hay que marcarlo como hecho con IA, como exige la plataforma.
