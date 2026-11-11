# BeatGen

Generates Beat Saber levels from audio with a Transformer conditioned on difficulty and intensity.
Note density follows the structure of the song, or a curve given by hand.

The project concluded in November 2026. Generated levels are playable but less readable than
human-authored ones. The trained model and dataset are not distributed (see [licensing](#data-and-licensing)).

## Results

60 held-out difficulties, evaluated with `eval.py`:

| | F1 (exact) | F1 (±1 step) | Density error |
|---|---|---|---|
| BeatGen | **0.66** | **0.74** | **17 %** |
| Spectral-flux baseline | 0.39 | 0.72 | |

The baseline places the reference map's note count per 4-bar block on the steps with the highest spectral flux.

## Method

1. **Beat grid.** Tempo and phase from a harmonic comb over the onset envelope, quantized to 1/4 beat.
2. **Features.** 80-band log-mel per grid step; one token per hand (empty or 4 columns × 3 rows × 9 directions).
3. **Model.** Encoder-decoder Transformer. The encoder takes audio, bar position, difficulty and intensity;
   the decoder predicts both hands autoregressively.
4. **Intensity.** Notes per beat over 4-bar blocks, taken from each map during training and from
   energy-ranked song sections (or `--curve`) at generation.
5. **Post-processing.** Forehand/backhand alternation, uncrossed arms, no overlapping notes.

Trained on 5408 BeatSaver maps (ranked, or highly rated with at least 50 votes; no automapped or AI-declared levels).

## Usage

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt

hf download PGSCOM/beatgen-maps-index meta.jsonl --type dataset --local-dir .
python download.py --from meta.jsonl    # ~45 GB
python prepare.py
python train.py --d 384 --layers 6 --steps 100000    # ~75 min on one GPU

python generate.py song.mp3 --diffs Hard Expert
python generate.py song.mp3 --curve "0:0.5,30:2,75:1"    # notes per beat from each second
```

## Limitations

Constant tempo only, no bombs or walls, one note per hand per quarter beat, and repeated choruses
don't get matching patterns.

## Data and licensing

Code is MIT. Nothing derived from the maps is published:

- **Maps and processed data.** BeatSaver's [terms](https://beatsaver.com/policy/tos) require written
  permission for any republishing or derivative use of other users' content.
- **Audio.** The maps include commercial recordings.
- **Model.** It is trained on mappers' work without their consent and arguably a derivative work.
  Releasing it would require permission from BeatSaver or the mappers.

The [index](https://huggingface.co/datasets/PGSCOM/beatgen-maps-index) lists map ids and version
hashes only, so the dataset can be rebuilt from BeatSaver with `download.py --from`.
Levels made with BeatGen must be declared as AI-generated on BeatSaver.
