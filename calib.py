"""Calibra y evalúa sobre el dataset: notas conservadas, NPB por dificultad, BPM/fase de la rejilla.

uso: python calib.py [--max 0]
ONSET_LATENCY se ajusta con la mitad de las canciones (id par en hash) y se evalúa con la otra mitad.
"""
import argparse, pathlib, zlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import beatmap as bm


def one(path):
    try:
        m = bm.load_map(path.read_bytes())
        if m is None:
            return None
        y = bm.load_audio(m["audio"])
        n = int((len(y) / bm.SR - m["offset"]) * m["bpm"] / 60 * bm.SPB)
        kept = {k: (len(v), int((bm.notes_to_tokens(v, n) > 0).sum())) for k, v in m["diffs"].items()}
        bpm_e, off_e = bm.estimate_grid(y)
        return path.stem, m["bpm"], m["offset"], bpm_e, off_e, kept
    except Exception:
        return None


def npb_windows(tok):
    """Notas/beat en ventanas de 4 beats no vacías."""
    c = (tok > 0).sum(1)
    w = 4 * bm.SPB
    c = c[:len(c) // w * w].reshape(-1, w).sum(1) / 4
    return c[c > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--proc", default="data/proc")
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    files = sorted(pathlib.Path(a.raw).glob("*.zip"))[:a.max or None]
    with ProcessPoolExecutor(8) as ex:
        rs = [r for r in ex.map(one, files, chunksize=4) if r]
    print(f"{len(rs)} mapas válidos de {len(files)} ({1 - len(rs) / len(files):.0%} descartados)")

    tot = np.array([v for r in rs for v in r[5].values()]).sum(0)
    print(f"notas conservadas al tokenizar: {tot[1] / tot[0]:.1%}")

    win = {k: [] for k in bm.DIFFS}
    for r in rs:
        f = pathlib.Path(a.proc) / f"{r[0]}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        for k in bm.DIFFS:
            if k in d:
                win[k].append(npb_windows(d[k]))
    for k, v in win.items():
        if v:
            v = np.concatenate(v)
            print(f"NPB {k}: p15 {np.percentile(v, 15):.2f}  p50 {np.median(v):.2f}  p85 {np.percentile(v, 85):.2f}")

    # rejilla: BPM correcto salvo octava y error de fase (s) frente a la rejilla del mapa
    ok, err, split = [], [], []
    for sid, bpm, off, bpm_e, off_e in (r[:5] for r in rs):
        ratio = np.log2(bpm_e / bpm)
        good = abs(ratio - round(ratio)) < 0.003
        ok.append(good)
        if good:
            p = 60 / max(bpm, bpm_e)  # el peine más fino contiene al otro
            e = (off_e - off + bm.ONSET_LATENCY) % p  # sin la compensación actual, para recalibrarla
            err.append(e - p if e > p / 2 else e)
            split.append(zlib.crc32(sid.encode()) % 2)
    print(f"BPM correcto (salvo octava): {np.mean(ok):.1%}")
    err, split = np.array(err), np.array(split)
    lat = np.median(err[split == 0])
    val = np.abs(err[split == 1] - lat)
    print(f"ONSET_LATENCY recalibrada (mitad A): {lat * 1000:.1f} ms (actual {bm.ONSET_LATENCY * 1000:.0f} ms)")
    print(f"fase en mitad B: <15 ms {np.mean(val < 0.015):.1%}, <30 ms {np.mean(val < 0.03):.1%}, mediana {np.median(val) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
