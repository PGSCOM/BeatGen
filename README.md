# BeatGen

BeatGen generates Beat Saber maps from audio. You give it a song and it returns a zip you can drop
into `CustomLevels`. Note density follows the structure of the song (calmer verses, busier
choruses), or you can set it by hand.

The project is finished (November 2026). It works end to end and the maps are playable, but they
are harder to follow than maps made by people. The trained model and the dataset are not published;
see [Data and licensing](#data-and-licensing).

## Results

Evaluated with `eval.py` on 60 difficulties from songs held out of training:

| | F1 (exact) | F1 (±1 step) | Density error | Parity fixes |
|---|---|---|---|---|
| BeatGen (d=384, 6 layers) | **0.66** | **0.74** | **17 %** | 8 % |
| Baseline | 0.39 | 0.72 | | |
| Human maps | | | | 4.7 % |

- **F1** compares the grid steps that have notes in the generated map and in the human one.
- **Baseline** places as many notes as the human map has in each 4-bar block, on the steps with the
  most spectral flux. It knows the real note count, so it is a fairly strong baseline.
- **Density error** is how far the generated notes per beat are from the requested ones.
- **Parity fixes** is the share of notes the post-processing has to flip or remove.

In play testing (Hard and Expert) the density felt right, but the maps are hard to follow: the flow
lacks the intent a human mapper puts into it.

## How it works

```
song ─► BPM + phase ─► 1/4-beat grid ─► log-mel per step ─┐
                                                          ├─► Transformer ─► notes ─► parity ─► zip
difficulty + intensity per section ───────────────────────┘
```

**Grid.** The song is laid on a grid of quarter beats. BPM and phase come from a harmonic comb over
the onset envelope: a coarse search over 10-second windows, then a fine one over the whole song. On
the dataset it finds the right BPM (up to octave) for 97 % of songs and lands within 15 ms of the
mapper's grid for 63 % (median error 8 ms).

**Representation.** Each grid step has an 80-band log-mel spectrum averaged over the step, and one
token per hand: empty, or one of 108 notes (4 columns × 3 rows × 9 cut directions).

**Model.** An encoder-decoder Transformer. The encoder sees the audio, the position in the bar, the
difficulty and the requested intensity. The decoder writes the notes for both hands step by step.
Intensity is notes per beat in 8 levels, held constant over 4-bar blocks. During training it is
measured from the map itself, so no labelling is needed.

**Generation.** The song is split into 8 sections by clustering, the sections are ranked by energy,
and each one gets an intensity within the usual range for the difficulty (or from `--curve`).
Sampling runs over overlapping windows at temperature 1.0, with a penalty on repeating recent
4-note motifs. Afterwards, crossed arms on simultaneous notes are swapped, each hand alternates
forehand and backhand, and overlapping notes are dropped.

## Usage

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Training needs an NVIDIA GPU.

```bash
git clone https://github.com/PGSCOM/BeatGen && cd BeatGen
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Rebuild the dataset from the public index (about 45 GB of downloads)
hf download PGSCOM/beatgen-maps-index meta.jsonl --type dataset --local-dir .
python download.py --from meta.jsonl
python prepare.py

# Train (about 75 minutes on a single GPU)
python train.py --d 384 --layers 6 --steps 100000

# Generate
python generate.py song.mp3 --diffs Hard Expert
python generate.py song.mp3 --curve "0:0.5,30:2,75:1"   # notes per beat from each second
python generate.py song.mp3 --bpm 128                   # if BPM detection is off
```

To play it, unzip the result into `Beat Saber_Data/CustomLevels/<folder>/` on PC, or into
`ModData/<game>/Mods/SongCore/CustomLevels/` on modded Quest or Pico.

`gpu.sh` runs the training on another machine and cleans up afterwards.

| File | Purpose |
|---|---|
| `download.py` | Downloads maps and metadata from BeatSaver, or an exact list from `meta.jsonl` |
| `prepare.py` | Turns map zips into per-step features and tokens (`data/proc/*.npz`) |
| `beatmap.py` | Map formats, tokens, audio features, grid estimation, parity, export |
| `model.py` | The Transformer |
| `train.py` | Training |
| `generate.py` | Song to zip |
| `eval.py` | Compares generated maps with human ones on validation songs |
| `calib.py` | Dataset statistics and grid accuracy |

## Dataset

6303 maps from [BeatSaver](https://beatsaver.com):

- 2466 ranked maps rated 0.6 or higher.
- 3837 unranked maps rated 0.85 or higher, with at least 50 votes.

Automapped maps, maps declared as AI-made and maps using Noodle or Mapping Extensions are excluded.
5408 of them are usable; the rest use variable BPM or the newer Info v4 format. Snapping notes to
the grid keeps 94 % of them.

## What went wrong along the way

- **The intensity signal leaked note positions.** It was first computed with a sliding window, which
  changes value exactly where notes are. The model learned to read notes from it and scored an F1 of
  0.99 that meant nothing. Holding it constant over 4-bar blocks fixed that.
- **The parity fix was breaking good maps.** It forced a forehand after every pause and flipped whole
  chains, changing 39 % of the notes in human maps. Now it changes 4.7 %.
- **The model looped halfway through songs.** Each window is conditioned on the notes already placed,
  so at temperature 0.9 a repeated motif fed itself: up to 49 % of 4-note sequences were repeats by
  minute 2:30, against about 12 % in human maps. Temperature 1.0 and the repetition penalty bring it
  down to 3-6 %.
- **Beat tracking hits a ceiling.** A recent neural beat tracker (beat_this) did no better than the
  comb (56.5 % vs 58.7 % within 15 ms). The limit is probably the reference itself: not every mapper
  aligns the grid to the actual beat.

## Limitations

- Constant BPM only, and no Info v4 maps.
- No bombs, walls, arcs or chains.
- One note per hand per step, and triplets are rounded to the nearest quarter beat.
- Parity only looks at cut direction, not wrist position.
- Repeated choruses don't get matching patterns.

## Data and licensing

The code is MIT licensed. Everything derived from the maps belongs to someone else, so none of it is
published:

- **Maps.** BeatSaver's [terms](https://beatsaver.com/policy/tos) require explicit written permission
  for "republishing or use of another user's content posted on BeatSaver including derivative work",
  and say that crediting the author is not enough.
- **Audio.** The zips contain full commercial songs.
- **Processed data.** The tokens are a direct transcription of each map, so they are a derivative work too.
- **The trained model.** Its weights come from other people's work used without their permission. The
  terms don't mention model training, but a model that imitates mappers' style can reasonably be
  seen as a derivative work, and many mappers object to automatic mapping (BeatSaver requires AI-made
  maps to be declared). Publishing it would need permission from BeatSaver or the mappers.

What is published is the [index](https://huggingface.co/datasets/PGSCOM/beatgen-maps-index): ids,
version hashes and public metadata, with no map content. With it, `download.py --from` rebuilds the
same dataset straight from BeatSaver. In November 2026 every map in a sample of 300 was still
available; some versions may disappear over time as mappers update their maps.

Maps made with BeatGen must be marked as AI-generated if uploaded to BeatSaver.
