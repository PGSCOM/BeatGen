"""Descarga mapas ranked de BeatSaver a data/raw/<id>.zip.

uso: python download.py --max 3000 --min-rating 0.75
"""
import argparse, json, pathlib, time, urllib.parse, urllib.request

API = "https://api.beatsaver.com/search/text/{page}?"
UA = {"User-Agent": "BeatGen-dataset/0.1"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--max", type=int, default=3000)
    ap.add_argument("--min-rating", type=float, default=0.75)
    ap.add_argument("--leaderboard", default="Ranked", help="Ranked | All | ScoreSaber | BeatLeader")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    q = urllib.parse.urlencode({"leaderboard": a.leaderboard, "sortOrder": "Rating",
                                "minRating": a.min_rating, "pageSize": 100})
    n, page = 0, 0
    while n < a.max:
        docs = json.loads(get(API.format(page=page) + q))["docs"]
        if not docs:
            break
        for m in docs:
            if m.get("automapper") or m.get("declaredAi", "None") != "None":
                continue
            v = m["versions"][0]
            if not any(d["characteristic"] == "Standard" for d in v["diffs"]):
                continue
            f = out / f"{m['id']}.zip"
            if not f.exists():
                try:
                    f.write_bytes(get(v["downloadURL"]))
                except OSError as e:
                    print("fallo", m["id"], e)
                    continue
                time.sleep(0.05)
            n += 1
            if n >= a.max:
                break
        print(f"página {page}: {n} mapas")
        page += 1


if __name__ == "__main__":
    main()
