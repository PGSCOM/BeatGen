"""data/raw/*.zip -> data/proc/<id>.npz with feats (T, N_MELS) and tokens (T, 2) per difficulty.

usage: python prepare.py [--workers 8]
"""
import argparse, pathlib, traceback
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from beatmap import load_map, load_audio, step_features, notes_to_tokens, SR, SPB


def process(args):
    src, dst = args
    if dst.exists():
        return "done"
    try:
        m = load_map(src.read_bytes())
        if m is None:
            return "skipped"
        y = load_audio(m["audio"])
        n = int((len(y) / SR - m["offset"]) * m["bpm"] / 60 * SPB)
        feats = step_features(y, m["bpm"], m["offset"], n)
        toks = {k: notes_to_tokens(v, n) for k, v in m["diffs"].items() if v}
        np.savez(dst, feats=feats.astype(np.float16), bpm=m["bpm"], **toks)
        return "ok"
    except Exception:
        return "error " + src.name + "\n" + traceback.format_exc(limit=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/proc")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, out / (p.stem + ".npz")) for p in sorted(pathlib.Path(a.raw).glob("*.zip"))]
    stats = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(process, jobs, chunksize=4):
            if r.startswith("error"):
                print(r)
            k = r.split()[0]
            stats[k] = stats.get(k, 0) + 1
    print(stats)


if __name__ == "__main__":
    main()
