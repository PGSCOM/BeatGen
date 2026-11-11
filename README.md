# BeatGen

BeatGen generates Beat Saber levels from audio. Given a song, it estimates the beat grid, predicts
notes with a Transformer conditioned on difficulty and intensity, and exports a level package that
loads as a custom level. Note density follows the structure of the song, with lower intensity in
verses and higher in choruses, and can also be specified manually.

The project concluded in November 2026. The pipeline is complete and the generated levels are
playable, although less readable than human-authored ones. Neither the trained model nor the
dataset is distributed; see [Data and licensing](#data-and-licensing).

## Results

Evaluation with `eval.py` on 60 difficulties from songs excluded from training:

| | F1 (exact) | F1 (±1 step) | Density error | Parity corrections |
|---|---|---|---|---|
| BeatGen (d=384, 6 layers) | **0.66** | **0.74** | **17 %** | 8 % |
| Baseline | 0.39 | 0.72 | | |
| Human-authored maps | | | | 4.7 % |

- **F1** measures agreement between the grid steps containing notes in the generated and the
  reference map.
- **Baseline** places, within each 4-bar block, the same number of notes as the reference map, on
  the steps with the highest spectral flux. Since it is given the true note count, it is a demanding
  reference.
- **Density error** is the relative deviation between requested and generated notes per beat.
- **Parity corrections** is the proportion of notes that post-processing must reorient or remove.

In play testing on Hard and Expert, density matched the music well, but the levels were harder to
follow than human-authored ones: the note flow lacks the deliberate structure a mapper provides.

## Method

```
audio ─► tempo + phase ─► 1/4-beat grid ─► log-mel per step ─┐
                                                             ├─► Transformer ─► notes ─► parity ─► level
difficulty + intensity per section ──────────────────────────┘
```

**Beat grid.** Each song is quantized to a grid of quarter beats. Tempo and phase are estimated with
a harmonic comb over the onset envelope, first coarsely over 10-second windows and then finely over
the full track. On the dataset, the estimated tempo is correct up to octave for 97 % of songs, and
the phase falls within 15 ms of the mapper's grid for 63 % (median error 8 ms).

**Representation.** Each grid step carries an 80-band log-mel spectrum averaged over the step and
one token per hand: either empty or one of 108 notes (4 columns × 3 rows × 9 cut directions).

**Model.** An encoder-decoder Transformer. The encoder receives the audio features, the position
within the bar, the difficulty and the target intensity; the decoder predicts both hands' notes
autoregressively. Intensity is expressed in notes per beat, quantized to 8 levels and held constant
over 4-bar blocks. During training it is derived from each map, so no manual annotation is required.

**Generation.** The song is segmented into 8 sections by agglomerative clustering; sections are
ranked by energy and assigned intensities within the typical range for the selected difficulty,
unless a curve is given with `--curve`. Sampling proceeds over overlapping windows at temperature
1.0, with a penalty on recently repeated 4-note motifs. Post-processing resolves crossed arms on
simultaneous notes, enforces forehand/backhand alternation per hand and removes overlapping notes.

## Usage

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Training requires an NVIDIA GPU.

```bash
git clone https://github.com/PGSCOM/BeatGen && cd BeatGen
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Rebuild the dataset from the public index (approx. 45 GB)
hf download PGSCOM/beatgen-maps-index meta.jsonl --type dataset --local-dir .
python download.py --from meta.jsonl
python prepare.py

# Train (approx. 75 minutes on a single GPU)
python train.py --d 384 --layers 6 --steps 100000

# Generate
python generate.py song.mp3 --diffs Hard Expert
python generate.py song.mp3 --curve "0:0.5,30:2,75:1"   # notes per beat from each timestamp (s)
python generate.py song.mp3 --bpm 128                   # override tempo detection
```

The output archive is extracted into `Beat Saber_Data/CustomLevels/<folder>/` on PC, or into
`ModData/<game>/Mods/SongCore/CustomLevels/` on modded standalone headsets. `gpu.sh` runs training
on a remote machine and removes all artifacts afterwards.

| File | Purpose |
|---|---|
| `download.py` | Retrieves maps and metadata from BeatSaver, or an exact set listed in `meta.jsonl` |
| `prepare.py` | Converts map archives into per-step features and tokens (`data/proc/*.npz`) |
| `beatmap.py` | Map formats, tokenization, audio features, beat grid, parity and export |
| `model.py` | Transformer model |
| `train.py` | Training loop |
| `generate.py` | Level generation from an audio file |
| `eval.py` | Evaluation against human-authored maps on validation songs |
| `calib.py` | Dataset statistics and beat grid accuracy |

## Dataset

6303 maps from [BeatSaver](https://beatsaver.com):

- 2466 ranked maps with a rating of at least 0.6.
- 3837 unranked maps with a rating of at least 0.85 and at least 50 votes.

Automapped levels, levels declared as AI-generated and levels using Noodle or Mapping Extensions
were excluded. 5408 maps are usable; the remainder use variable tempo or the Info v4 format.
Quantization to the grid preserves 94 % of notes.

## Notable issues

- **Information leak through the intensity signal.** Intensity was initially computed with a sliding
  window, whose value changes exactly at note positions. The model learned to infer notes from it,
  reaching a spurious F1 of 0.99. Holding intensity constant over 4-bar blocks removed the leak.
- **Parity correction degrading valid maps.** The correction forced a forehand after every pause and
  inverted entire note chains, altering 39 % of notes in human-authored maps. The revised rule alters 4.7 %.
- **Repetition in the second half of songs.** Since each window is conditioned on previously
  generated notes, at temperature 0.9 a repeated motif reinforced itself, reaching 49 % repeated
  4-note sequences around minute 2:30, against roughly 12 % in human-authored maps. Sampling at
  temperature 1.0 with a repetition penalty reduces this to 3-6 %.
- **Beat tracking ceiling.** A recent neural beat tracker (beat_this) did not outperform the harmonic
  comb (56.5 % vs. 58.7 % within 15 ms). The limiting factor is likely the reference itself, as not
  all mappers align their grid precisely with the audio.

## Limitations

- Constant tempo only; Info v4 maps are not supported.
- No bombs, walls, arcs or chains.
- At most one note per hand per grid step; triplets are rounded to the nearest quarter beat.
- Parity is determined by cut direction alone, without modelling wrist position.
- Repeated sections, such as choruses, do not receive consistent patterns.

## Data and licensing

The source code is released under the MIT license. Material derived from the maps is owned by third
parties and is therefore not distributed:

- **Maps.** BeatSaver's [terms of service](https://beatsaver.com/policy/tos) require explicit written
  permission for "republishing or use of another user's content posted on BeatSaver including
  derivative work", and state that attribution alone is not sufficient.
- **Audio.** The map archives contain complete commercial recordings.
- **Processed data.** The token sequences are a direct transcription of each map and constitute a
  derivative work.
- **Trained model.** The weights result from training on third-party work without the authors'
  consent. The terms do not address model training explicitly, but a model that reproduces mappers'
  style may reasonably be regarded as a derivative work, and a significant part of the mapping
  community opposes automated mapping (BeatSaver requires AI-generated levels to be declared).
  Releasing it would require permission from BeatSaver or from the mappers.

Only the [index](https://huggingface.co/datasets/PGSCOM/beatgen-maps-index) is published: map ids,
version hashes and public metadata, without map content. `download.py --from` uses it to rebuild the
same dataset directly from BeatSaver. In November 2026, all maps in a sample of 300 remained
available; individual versions may become unavailable as mappers publish updates.

Levels generated with BeatGen must be declared as AI-generated if uploaded to BeatSaver.
