"""Encoder (audio por paso de rejilla + dificultad + intensidad) / decoder causal (tokens de ambas manos)."""
import torch
from torch import nn
from beatmap import N_TOK, N_DENS, N_MELS, DIFFS, SPB

BOS = N_TOK


class BeatGen(nn.Module):
    def __init__(self, d=256, heads=4, layers=4, max_len=1024, dropout=0.1):
        super().__init__()
        self.cfg = dict(d=d, heads=heads, layers=layers, max_len=max_len, dropout=dropout)
        self.audio = nn.Linear(N_MELS, d)
        self.pos = nn.Embedding(max_len, d)
        self.phase = nn.Embedding(4 * SPB, d)  # posición dentro del compás
        self.diff = nn.Embedding(len(DIFFS), d)
        self.dens = nn.Embedding(N_DENS, d)
        self.tok = nn.ModuleList([nn.Embedding(N_TOK + 1, d) for _ in range(2)])
        self.tf = nn.Transformer(d, heads, layers, layers, 4 * d, dropout, batch_first=True, norm_first=True)
        self.out = nn.Linear(d, 2 * N_TOK)

    def encode(self, feats, diff, dens):
        """feats (B,T,N_MELS), diff (B,), dens (B,T). La ventana debe empezar en múltiplo de 4*SPB."""
        p = torch.arange(feats.shape[1], device=feats.device)
        cond = self.pos(p) + self.phase(p % (4 * SPB)) + self.diff(diff)[:, None]
        return self.tf.encoder(self.audio(feats) + cond + self.dens(dens)), cond

    def decode(self, prev, memory, cond):
        """prev (B,L,2): tokens desplazados con BOS delante. Devuelve logits (B,L,2,N_TOK)."""
        L = prev.shape[1]
        x = self.tok[0](prev[..., 0]) + self.tok[1](prev[..., 1]) + cond[:, :L]
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=prev.device)
        h = self.tf.decoder(x, memory, tgt_mask=mask, tgt_is_causal=True)
        return self.out(h).view(*h.shape[:2], 2, N_TOK)

    def forward(self, feats, diff, dens, prev):
        memory, cond = self.encode(feats, diff, dens)
        return self.decode(prev, memory, cond)
