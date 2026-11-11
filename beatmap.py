"""Formato Beat Saber <-> tokens sobre rejilla de beats, features de audio, paridad y export.

Token por paso de rejilla y por mano: 0 = nada, 1..108 = nota (x 0-3, y 0-2, dirección 0-8).
"""
import io, json, struct, zipfile, zlib
import numpy as np
import librosa
import soundfile as sf

SPB = 4  # pasos por beat  # ponytail: 1/4 de beat; tresillos y streams 1/8 se redondean. Subir a 8 o 12 si el modelo base lo pide
SR, HOP, N_MELS = 22050, 512, 80
ONSET_LATENCY = 0.036  # s; retraso de la envolvente de onsets frente a la rejilla de los mapas (calib.py, 5408 mapas)
DIFFS = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
N_TOK = 1 + 4 * 3 * 9
DENS_EDGES = [0.1, 0.5, 1, 1.5, 2, 3, 4]  # notas por beat -> 8 niveles de intensidad
N_DENS = len(DENS_EDGES) + 1
OPP = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 7: 4, 5: 6, 6: 5, 8: 8}  # dirección opuesta


def note_token(x, y, d):
    return 1 + (x * 3 + y) * 9 + d


def token_note(t):
    t -= 1
    return t // 27, (t // 9) % 3, t % 9


# ---------- audio ----------

def load_audio(src):
    """src: ruta o bytes (.egg = ogg). Devuelve mono a SR."""
    y, sr = sf.read(io.BytesIO(src) if isinstance(src, bytes) else src, dtype="float32", always_2d=True)
    return librosa.resample(y.mean(1), orig_sr=sr, target_sr=SR)


def step_features(y, bpm, offset, n_steps):
    """Log-mel promediado en cada paso de rejilla -> (n_steps, N_MELS) en [0,1]."""
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, hop_length=HOP, n_mels=N_MELS), ref=np.max)
    S = (S + 80) / 80
    F = S.shape[1]
    c = np.concatenate([np.zeros((N_MELS, 1)), np.cumsum(S, 1)], 1)
    dt = 60 / bpm / SPB
    t = offset + np.arange(n_steps) * dt
    a = np.clip(librosa.time_to_frames(t - dt / 2, sr=SR, hop_length=HOP), 0, F - 1)
    b = np.clip(np.maximum(a + 1, librosa.time_to_frames(t + dt / 2, sr=SR, hop_length=HOP)), 1, F)
    return ((c[:, b] - c[:, a]) / (b - a)).T.astype(np.float32)


def _comb(bpms, x, t):
    """Energía del peine armónico (beat, corchea, semicorchea) de la envolvente de onsets x."""
    return sum(np.abs(np.exp(2j * np.pi * np.outer(h * bpms / 60, t)) @ x) for h in (1, 2, 4))


def estimate_grid(y, lo=90.0, hi=180.0):
    """BPM constante en [lo, hi) y offset (s) del primer beat.
    Grueso: peine no coherente en ventanas de 10 s (tolera deriva). Fino: peine coherente sobre toda la canción."""
    # ponytail: la octava es ambigua (75 vs 150 da igual al jugar); usar --bpm si la canción lo necesita
    o = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    o = o - o.mean()
    fps = SR / HOP
    W = int(10 * fps)
    coarse = np.arange(lo, hi, 0.1)
    c = _comb(coarse, o[:len(o) // W * W].reshape(-1, W).T, np.arange(W) / fps).sum(1)
    t = np.arange(len(o)) / fps
    fine = coarse[c.argmax()] + np.arange(-0.3, 0.3, 0.002)
    bpm = fine[_comb(fine, o, t).argmax()]
    period = 60 / bpm
    phis = np.linspace(0, period, 400, endpoint=False)  # fase que maximiza la correlación con el peine
    score = sum((np.exp(2j * np.pi * h * t / period) @ o * np.exp(-2j * np.pi * h * phis / period)).real for h in (1, 2, 4))
    return bpm, (phis[score.argmax()] - ONSET_LATENCY) % period


DENS_BLOCK = 16 * SPB  # 4 compases


def density_levels(tokens):
    """Notas/beat por bloques de 4 compases -> nivel 0..N_DENS-1 por paso.
    Constante dentro del bloque: una ventana deslizante cambia justo donde hay notas y le chiva al modelo su posición."""
    c = (tokens > 0).sum(1)
    n = -(-len(c) // DENS_BLOCK) * DENS_BLOCK
    per_beat = np.pad(c, (0, n - len(c))).reshape(-1, DENS_BLOCK).sum(1) / (DENS_BLOCK / SPB)
    return np.repeat(np.digitize(per_beat, DENS_EDGES), DENS_BLOCK)[:len(c)]


# ---------- lectura de mapas ----------

def _bpm_changes(info_bpm, d):
    ev = d.get("bpmEvents") or [{"m": e["_floatValue"]} for e in d.get("_events", []) if e.get("_type") == 100]
    ev += d.get("_customData", {}).get("_BPMChanges", [])
    return any(abs(e.get("m", e.get("_BPM", info_bpm)) - info_bpm) > 1e-3 for e in ev)


def read_notes(d):
    """(beat, x, y, mano, dir) de un .dat v2 o v3; ignora bombas, muros y notas fuera de rejilla."""
    if "colorNotes" in d:
        raw = [(n["b"], n.get("x", 0), n.get("y", 0), n.get("c", 0), n.get("d", 0)) for n in d["colorNotes"]]
    else:
        raw = [(n["_time"], n["_lineIndex"], n["_lineLayer"], n["_type"], n["_cutDirection"]) for n in d.get("_notes", [])]
    return [r for r in raw if r[3] in (0, 1) and 0 <= r[1] < 4 and 0 <= r[2] < 3 and 0 <= r[4] < 9]


def load_map(zip_bytes):
    """Devuelve dict con audio, bpm y {dificultad: lista de notas}, o None si no sirve."""
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    names = {n.lower(): n for n in z.namelist()}
    info = json.loads(z.read(names["info.dat"]))
    if "_beatsPerMinute" not in info:  # ponytail: Info v4 no soportado, añadir si se pierden muchos mapas
        return None
    bpm, offset = info["_beatsPerMinute"], info.get("_songTimeOffset", 0)
    diffs = {}
    for s in info["_difficultyBeatmapSets"]:
        if s["_beatmapCharacteristicName"] != "Standard":
            continue
        for b in s["_difficultyBeatmaps"]:
            fn = names.get(b["_beatmapFilename"].lower())
            if fn is None or b["_difficulty"] not in DIFFS:
                continue
            d = json.loads(z.read(fn))
            if _bpm_changes(bpm, d):  # ponytail: mapas con cambios de BPM descartados
                return None
            diffs[b["_difficulty"]] = read_notes(d)
    if not diffs:
        return None
    return {"audio": z.read(names[info["_songFilename"].lower()]), "bpm": bpm, "offset": offset, "diffs": diffs}


def notes_to_tokens(notes, n_steps):
    tok = np.zeros((n_steps, 2), np.int64)
    for beat, x, y, h, d in notes:
        i = round(beat * SPB)
        if 0 <= i < n_steps and tok[i, h] == 0:  # ponytail: una nota por mano y paso (sin stacks/sliders)
            tok[i, h] = note_token(x, y, d)
    return tok


def tokens_to_notes(tok, beat0=0.0):
    out = []
    for i, h in zip(*np.nonzero(tok)):
        x, y, d = token_note(int(tok[i, h]))
        out.append((float(beat0 + i / SPB), x, y, int(h), d))
    return out


# ---------- jugabilidad ----------

def swing(d, hand):
    """1 = forehand, 0 = backhand, None = punto."""
    if d in (1, 6, 7):
        return 1
    if d in (0, 4, 5):
        return 0
    if d == 8:
        return None
    return int(d == (2 if hand == 1 else 3))  # corte horizontal hacia dentro = forehand


def fix_parity(notes, reset_beats=2.0):
    """Alterna forehand/backhand por mano; quita choques de ambas manos en la misma celda."""
    # ponytail: paridad por dirección, sin tener en cuenta posición/rotación de muñeca. Basta para un baseline jugable
    notes = sorted(notes)
    out, last, taken = [], {0: (1, -1e9), 1: (1, -1e9)}, set()
    for beat, x, y, h, d in notes:
        if (beat, x, y) in taken:
            continue
        p, lb = last[h]
        want = 1 if beat - lb >= reset_beats else 1 - p
        s = swing(d, h)
        if s is not None and s != want:
            d = OPP[d]
        last[h] = (want, beat)
        taken.add((beat, x, y))
        out.append((beat, x, y, h, d))
    return out


# ---------- export ----------

NJS = {"Easy": 10, "Normal": 10, "Hard": 12, "Expert": 16, "ExpertPlus": 18}


def _png(w=256, h=256, rgb=(24, 24, 24)):
    raw = b"".join(b"\0" + bytes(rgb) * w for _ in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def export_zip(path, audio_path, bpm, maps, name="BeatGen"):
    """maps: {dificultad: [(beat, x, y, mano, dir)]}. Escribe zip listo para CustomLevels."""
    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    ogg = io.BytesIO()
    with sf.SoundFile(ogg, "w", sr, data.shape[1], format="OGG", subtype="VORBIS") as f:
        for i in range(0, len(data), 1 << 15):  # por bloques: escribir todo de golpe revienta libsndfile
            f.write(data[i:i + (1 << 15)])
    info = {
        "_version": "2.1.0", "_songName": name, "_songSubName": "", "_songAuthorName": "",
        "_levelAuthorName": "BeatGen", "_beatsPerMinute": round(bpm, 3), "_songTimeOffset": 0,
        "_shuffle": 0, "_shufflePeriod": 0.5, "_previewStartTime": 12, "_previewDuration": 10,
        "_songFilename": "song.egg", "_coverImageFilename": "cover.png",
        "_environmentName": "DefaultEnvironment", "_allDirectionsEnvironmentName": "GlassDesertEnvironment",
        "_difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
            {"_difficulty": k, "_difficultyRank": 2 * DIFFS.index(k) + 1, "_beatmapFilename": f"{k}.dat",
             "_noteJumpMovementSpeed": NJS[k], "_noteJumpStartBeatOffset": 0}
            for k in DIFFS if k in maps]}],
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Info.dat", json.dumps(info, indent=1))
        z.writestr("song.egg", ogg.getvalue())
        z.writestr("cover.png", _png())
        for k, notes in maps.items():
            z.writestr(f"{k}.dat", json.dumps({"_version": "2.2.0", "_events": [], "_obstacles": [], "_notes": [
                {"_time": round(b, 4), "_lineIndex": x, "_lineLayer": yy, "_type": h, "_cutDirection": d}
                for b, x, yy, h, d in notes]}))


if __name__ == "__main__":  # autocomprobación
    assert all(token_note(note_token(x, y, d)) == (x, y, d) for x in range(4) for y in range(3) for d in range(9))
    notes = [(0, 1, 0, 0, 1), (0.5, 2, 0, 1, 1), (1, 1, 0, 0, 1), (1.25, 2, 1, 1, 0), (2, 1, 0, 0, 8)]
    assert tokens_to_notes(notes_to_tokens(notes, 16)) == sorted(notes)
    fixed = fix_parity(notes)
    assert [n[4] for n in fixed if n[3] == 0] == [1, 0, 8]  # down, (down->up), punto
    assert [n[4] for n in fixed if n[3] == 1] == [1, 0]
    assert fix_parity([(0, 1, 0, 0, 1), (0, 1, 0, 1, 1)]) == [(0, 1, 0, 0, 1)]  # choque de celda
    assert density_levels(np.array([[1, 0]] * 64))[16] == N_DENS - 1  # 4 notas/beat -> nivel máximo
    assert density_levels(np.array([[1, 0], [0, 0], [0, 0], [0, 0]] * 16))[16] == DENS_EDGES.index(1) + 1
    rnd = (np.random.default_rng(0).random((128, 2)) < 0.3).astype(int)
    assert len(set(density_levels(rnd)[:64])) == 1  # sin fugas: constante dentro del bloque
    print("ok")
