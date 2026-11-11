"""Entrena BeatGen sobre data/proc/*.npz con ventanas aleatorias de la rejilla.

uso: python train.py --steps 50000 --batch 32
"""
import argparse, glob, math, random, time
import numpy as np
import torch
import torch.nn.functional as F
from beatmap import DIFFS, SPB, density_levels
from model import BeatGen, BOS

BAR = 4 * SPB


def load(files):
    items = []  # (feats, tokens, diff_idx, dens)
    for f in files:
        d = np.load(f)
        for k in DIFFS:
            if k in d:
                items.append((d["feats"], d[k], DIFFS.index(k), density_levels(d[k])))
    return items


def batch(items, n, W):
    f, t, df, de = [], [], [], []
    for feats, tok, di, dens in random.choices(items, k=n):
        T = len(tok)
        s = random.randrange(0, max(1, T - W) // BAR + 1) * BAR  # alineado al compás
        pad = lambda a: np.pad(a[s:s + W], [(0, W - len(a[s:s + W]))] + [(0, 0)] * (a.ndim - 1))
        f.append(pad(feats).astype(np.float32)); t.append(pad(tok)); de.append(pad(dens)); df.append(di)
    t = torch.tensor(np.stack(t))
    prev = torch.cat([torch.full_like(t[:, :1], BOS), t[:, :-1]], 1)
    return torch.tensor(np.stack(f)), torch.tensor(df), torch.tensor(np.stack(de)), prev, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/proc")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--window", type=int, default=256, help="pasos de rejilla (múltiplo de 32)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    assert a.window % (2 * BAR) == 0

    files = sorted(glob.glob(f"{a.data}/*.npz"))
    random.Random(0).shuffle(files)
    nv = max(1, len(files) // 20)  # 5% de canciones para validación
    val, tr = load(files[:nv]), load(files[nv:])
    print(f"{len(files)} canciones, {len(tr)} dificultades de entrenamiento, {len(val)} de validación")

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = BeatGen(d=a.d, layers=a.layers, max_len=a.window).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    step0 = 0
    if a.resume:
        ck = torch.load(a.out, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step0 = ck["step"]
    sched = lambda s: min(1, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps)))
    amp = torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda")

    def loss_of(b):
        feats, diff, dens, prev, tgt = (x.to(dev) for x in b)
        with amp:
            logits = model(feats, diff, dens, prev)
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1))

    t0 = time.time()
    for step in range(step0, a.steps):
        model.train()
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        loss = loss_of(batch(tr, a.batch, a.window))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == a.steps - 1:
            model.eval()
            with torch.no_grad():
                vl = sum(loss_of(batch(val, a.batch, a.window)).item() for _ in range(4)) / 4
            print(f"paso {step} loss {loss.item():.4f} val {vl:.4f} {time.time() - t0:.0f}s", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1, "cfg": model.cfg}, a.out)


if __name__ == "__main__":
    main()
