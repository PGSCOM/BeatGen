"""Downloads maps from BeatSaver to data/raw/<id>.zip and their metadata to data/raw/meta.jsonl.

usage: python download.py --from meta.jsonl
       python download.py --max 3000 --min-rating 0.6
       python download.py --leaderboard All --min-rating 0.85 --min-votes 50
"""
import argparse, json, pathlib, time, urllib.parse, urllib.request

API = "https://api.beatsaver.com/search/text/{page}?"
CDN = "https://r2cdn.beatsaver.com/{hash}.zip"
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
    ap.add_argument("--min-votes", type=int, default=0, help="minimum number of votes (up + down)")
    ap.add_argument("--from", dest="src", help="meta.jsonl of a published dataset: download exactly those versions")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.src:
        for m in map(json.loads, open(a.src)):
            f = out / f"{m['id']}.zip"
            if not f.exists():
                try:
                    f.write_bytes(get(CDN.format(hash=m["hash"])))
                except OSError as e:
                    print("failed", m["id"], e)
                time.sleep(0.05)
        return
    q = urllib.parse.urlencode({"leaderboard": a.leaderboard, "sortOrder": "Rating",
                                "minRating": a.min_rating, "pageSize": 100})
    n, page = 0, 0
    mp = out / "meta.jsonl"
    seen = {json.loads(l)["id"] for l in open(mp)} if mp.exists() else set()
    meta = open(mp, "a")
    while n < a.max:
        docs = json.loads(get(API.format(page=page) + q))["docs"]
        if not docs:
            break
        for m in docs:
            if m.get("automapper") or m.get("declaredAi", "None") != "None":
                continue
            if m["stats"]["upvotes"] + m["stats"]["downvotes"] < a.min_votes:
                continue
            v = m["versions"][0]
            if not any(d["characteristic"] == "Standard" for d in v["diffs"]):
                continue
            if any(d.get("ne") or d.get("me") for d in v["diffs"]):  # Noodle / Mapping Extensions
                continue
            f = out / f"{m['id']}.zip"
            if not f.exists():
                try:
                    f.write_bytes(get(v["downloadURL"]))
                except OSError as e:
                    print("failed", m["id"], e)
                    continue
                time.sleep(0.05)
            if m["id"] not in seen:
                seen.add(m["id"])
                meta.write(json.dumps({"id": m["id"], "name": m["name"], "song": m["metadata"].get("songName"),
                                   "artist": m["metadata"].get("songAuthorName"), "mapper": m["uploader"]["name"],
                                   "bpm": m["metadata"].get("bpm"), "rating": m["stats"].get("score"),
                                   "upvotes": m["stats"].get("upvotes"), "tags": m.get("tags", []),
                                   "ranked": m.get("ranked"), "blRanked": m.get("blRanked"),
                                   "uploaded": m.get("uploaded"), "hash": v["hash"],
                                   "diffs": [d["difficulty"] for d in v["diffs"] if d["characteristic"] == "Standard"]},
                                  ensure_ascii=False) + "\n")
                meta.flush()
            n += 1
            if n >= a.max:
                break
        print(f"page {page}: {n} maps")
        page += 1


if __name__ == "__main__":
    main()
