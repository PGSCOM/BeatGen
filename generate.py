"""Canción -> zip de Beat Saber.

uso: python generate.py cancion.mp3 --diffs Hard Expert
     python generate.py cancion.mp3 --curve "0:1,45:2.5,90:1.5"   # notas/beat desde cada segundo
"""
import argparse, pathlib
import numpy as np
import librosa
import torch
from beatmap import (DIFFS, DENS_EDGES, SPB, load_audio, estimate_grid, step_features,
                     tokens_to_notes, fix_parity, export_zip)
from model import BeatGen, BOS

# notas/beat por ventana de 4 beats (percentiles 15 y 85, calib.py sobre 5408 mapas)
NPB = {"Easy": (0.5, 1), "Normal": (0.5, 1.5), "Hard": (0.75, 2), "Expert": (1, 2.5), "ExpertPlus": (1.25, 3)}


def sections(feats, n):
    """Segmenta la canción en n secciones; devuelve inicios (pasos) y energía relativa 0..1 por sección."""
    bounds = list(librosa.segment.agglomerative(feats.T, n)) + [len(feats)]
    energy = np.array([feats[a:b].mean() for a, b in zip(bounds, bounds[1:])])
    rank = energy.argsort().argsort() / max(1, n - 1)
    return bounds, rank


def auto_curve(feats, diff, n_sections):
    lo, hi = NPB[diff]
    bounds, rank = sections(feats, n_sections)
    npb = np.zeros(len(feats))
    for (a, b), r in zip(zip(bounds, bounds[1:]), rank):
        npb[a:b] = lo + (hi - lo) * r
    npb[feats.mean(1) < 0.15] = 0  # silencios
    return npb, bounds, rank


def manual_curve(spec, n, bpm, offset):
    pts = sorted((float(t), float(v)) for t, v in (p.split(":") for p in spec.split(",")))
    t = offset + np.arange(n) * 60 / bpm / SPB
    return np.array([next((v for s, v in reversed(pts) if s <= x), pts[0][1]) for x in t])


@torch.no_grad()
def sample(model, feats, diff, dens, W, temp):
    T, hop = len(feats), W // 2
    tok = torch.zeros(T, 2, dtype=torch.long)
    for s in range(0, T, hop):
        e = min(s + W, T)
        mem, cond = model.encode(torch.tensor(feats[None, s:e]), torch.tensor([diff]), torch.tensor(dens[None, s:e]))
        for t in range(0 if s == 0 else min(hop, e - s), e - s):
            prev = torch.cat([torch.full((1, 2), BOS), tok[s:s + t]])[None]
            logits = model.decode(prev, mem, cond)[0, -1] / temp
            tok[s + t] = torch.multinomial(logits.softmax(-1), 1)[:, 0]
        if e == T:
            break
    return tok.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--diffs", nargs="+", default=["Expert"], choices=DIFFS)
    ap.add_argument("--curve", help='intensidad manual "seg:notas_por_beat,..."; por defecto, automática por secciones')
    ap.add_argument("--sections", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--bpm", type=float, help="forzar BPM si la detección falla (p.ej. mitad/doble)")
    ap.add_argument("--out")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    model = BeatGen(**ck["cfg"]).eval()
    model.load_state_dict(ck["model"])

    y = load_audio(a.audio)
    bpm, offset = estimate_grid(y)
    if a.bpm:
        offset, bpm = offset % (60 / a.bpm), a.bpm
    n = int((len(y) / 22050 - offset) * bpm / 60 * SPB)
    feats = step_features(y, bpm, offset, n)
    print(f"BPM {bpm:.2f}, primer beat {offset:.3f}s, {n} pasos")

    maps = {}
    for diff in a.diffs:
        if a.curve:
            npb = manual_curve(a.curve, n, bpm, offset)
        else:
            npb, bounds, rank = auto_curve(feats, diff, a.sections)
            print(diff, "secciones:", ", ".join(f"{offset + b * 60 / bpm / SPB:.0f}s→{npb[b]:.1f}" for b in bounds[:-1]))
        dens = np.digitize(npb, DENS_EDGES)
        tok = sample(model, feats, DIFFS.index(diff), dens, ck["cfg"]["max_len"], a.temp)
        maps[diff] = fix_parity(tokens_to_notes(tok, beat0=offset * bpm / 60))
        print(f"{diff}: {len(maps[diff])} notas ({len(maps[diff]) / (n / SPB):.2f}/beat)")

    out = a.out or pathlib.Path(a.audio).stem + "_beatgen.zip"
    export_zip(out, a.audio, bpm, maps, name=pathlib.Path(a.audio).stem)
    print("→", out)


if __name__ == "__main__":
    main()
