"""Compares generated maps with the human ones on validation songs, and with a spectral-flux baseline.

usage: python eval.py --ckpt ckpt.pt --n 60
"""
import argparse, glob, random
import numpy as np
import torch
from beatmap import DIFFS, DENS_BLOCK, density_levels, tokens_to_notes, fix_parity, note_token, crossed
from model import BeatGen
from generate import sample

WIN = DENS_BLOCK


def f1(pred, real, tol=1):
    p, r = np.flatnonzero(pred), np.flatnonzero(real)
    if not len(p) or not len(r):
        return float(len(p) == len(r))
    used, tp = set(), 0
    for i in p:
        j = r[np.searchsorted(r, i - tol):np.searchsorted(r, i + tol, "right")]
        j = [x for x in j if x not in used]
        if j:
            used.add(j[0]); tp += 1
    return 2 * tp / (len(p) + len(r))


def npb_err(pred, real):
    n = len(real) // WIN * WIN
    a, b = pred[:n].reshape(-1, WIN).sum(1), real[:n].reshape(-1, WIN).sum(1)
    m = b > 0
    return float(np.mean(np.abs(a[m] - b[m]) / b[m])) if m.any() else 0.0


def baseline(feats, real):
    """Notes on spectral-flux peaks, as many per block as the human map has."""
    flux = np.r_[0, np.maximum(0, np.diff(feats.astype(np.float32), axis=0)).sum(1)]
    tok = np.zeros((len(feats), 2), np.int64)
    k = 0
    for s in range(0, len(feats), WIN):
        n = int(real[s:s + WIN].sum())
        for i in sorted(s + np.argsort(-flux[s:s + WIN])[:n]):
            h = k % 2
            tok[i, h] = note_token(1 + h, 0, 1 if (k // 2) % 2 == 0 else 0)
            k += 1
    return tok


def cross_rate(tok):
    notes = tokens_to_notes(tok)
    return 2 * len(crossed(notes)) / max(1, len(notes))


def parity_fix_rate(tok):
    notes = tokens_to_notes(tok)
    if not notes:
        return 0.0
    fixed = set(fix_parity(notes))
    return 1 - sum(n in fixed for n in notes) / len(notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--data", default="data/proc")
    ap.add_argument("--n", type=int, default=60, help="number of difficulties to evaluate")
    ap.add_argument("--steps", type=int, default=1024, help="grid steps per song, from the start")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--rep", type=float, default=0.3)
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.data}/*.npz"))
    random.Random(0).shuffle(files)
    val = files[:max(1, len(files) // 20)]
    ck = torch.load(a.ckpt, map_location="cpu")
    model = BeatGen(**ck["cfg"]).eval()
    model.load_state_dict(ck["model"])
    torch.manual_seed(0)

    rows = {k: [] for k in DIFFS}
    items = [(f, k) for f in val for k in DIFFS if k in np.load(f)]
    random.Random(1).shuffle(items)
    for f, k in items[:a.n]:
        d = np.load(f)
        feats, real = d["feats"][:a.steps].astype(np.float32), d[k][:a.steps]
        tok = sample(model, feats, DIFFS.index(k), density_levels(real), ck["cfg"]["max_len"], a.temp, a.rep)
        base = baseline(feats, (real > 0).any(1))
        ra, ga, ba = (real > 0).any(1), (tok > 0).any(1), (base > 0).any(1)
        rows[k].append((f1(ga, ra, 0), f1(ga, ra), f1(ba, ra, 0), f1(ba, ra), npb_err((tok > 0).sum(1), (real > 0).sum(1)), parity_fix_rate(tok), cross_rate(tok), cross_rate(real)))
        print(f"{f.split('/')[-1]} {k}: F1 {rows[k][-1][0]:.2f} (baseline {rows[k][-1][2]:.2f})", flush=True)

    print(f"\n{'difficulty':<11} {'n':>3} {'F1':>6} {'F1±1':>6} {'base':>6} {'base±1':>6} {'NPB err':>8} {'parity':>8} {'crossed':>8} {'human':>7}")
    allr = [r for v in rows.values() for r in v]
    for k, v in list(rows.items()) + [("TOTAL", allr)]:
        if v:
            m = np.mean(v, 0)
            print(f"{k:<11} {len(v):>3} {m[0]:>6.2f} {m[1]:>6.2f} {m[2]:>6.2f} {m[3]:>6.2f} {m[4]:>8.0%} {m[5]:>8.0%} {m[6]:>8.1%} {m[7]:>7.1%}")


if __name__ == "__main__":
    main()
