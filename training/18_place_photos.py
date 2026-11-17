"""
18_place_photos.py -- pick up to K good photos per place for the app's card (laptop; pandas).

Inputs : brain/photo_scores<places>.npz (17_photo_scores.py, GPU server), brain/usa_europe_geotagged.csv
Score per photo (higher = better picture of the place):
    typicality  - 0.6 * people - 0.6 * junk - 0.08 * (people / food / pet words in its tags or title)
Photos the classifier calls mostly people or junk (> 0.5) are used only when a place has nothing else.
At most one photo per photographer, so the K photos are different views.
Photo links are stored as "server/id_secret" and served from live.staticflickr.com (Flickr's current host).

  .venv\\Scripts\\python.exe training\\18_place_photos.py
Outputs: brain/place_photos<places>.npz  (place index, rank, photo key, score; nothing user-level)
"""
import argparse, os, re
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--brain", default="brain")
ap.add_argument("--places", default="_v2")
ap.add_argument("--k", type=int, default=6)
ap.add_argument("--chunk", type=int, default=1_000_000)
ap.add_argument("--nrows", type=int, default=None, help="smoke test")
a = ap.parse_args()
out_path = f"{a.brain}/place_photos{a.places}.npz" if a.nrows is None else os.environ.get("TMP", ".") + "/place_photos_smoke.npz"
assert not os.path.exists(out_path), f"{out_path} exists; refusing to overwrite"

S = np.load(f"{a.brain}/photo_scores{a.places}.npz")
keep = S["place"] >= 0
sid = S["photo_id"][keep].astype(np.int64)
o = np.argsort(sid)
sid = sid[o]
s_place = S["place"][keep][o].astype(np.int32)
s_typ = S["typicality"][keep][o].astype(np.float32)
s_peo = S["people"][keep][o].astype(np.float32)
s_jnk = S["junk"][keep][o].astype(np.float32)
print(f"{len(sid):,} scored photos at places", flush=True)

WORDS = re.compile(r"\b(me|myself|self|selfie|selfportrait|portrait|portraits|family|friends?|people|person|girls?|boys?|"
                   r"woman|women|man|men|kids?|children|child|baby|babies|wedding|bride|groom|party|birthday|mom|mum|dad|"
                   r"wife|husband|son|daughter|sister|brother|grandma|grandpa|group|face|faces|smile|cute|dog|dogs|cat|cats|"
                   r"puppy|kitten|pet|food|dinner|lunch|breakfast|brunch|meal|drink|drinks|beer|wine|cocktail|coffee|"
                   r"cake|dessert|car|cars|bus|train interior|hotel room|room|bed)\b")
URL = re.compile(r"staticflickr\.com/(\d+/\d+_[0-9a-f]+)")

cols = {"uid": [], "flag": [], "key": [], "row": []}
done = 0
for ch in pd.read_csv(f"{a.brain}/usa_europe_geotagged.csv", usecols=["photoid", "uid", "title", "usertags", "downloadurl"],
                      chunksize=a.chunk, nrows=a.nrows, dtype=str, keep_default_na=False):
    pid = pd.to_numeric(ch["photoid"], errors="coerce").values
    ok = np.isfinite(pid)
    pid = pid[ok].astype(np.int64)
    pos = np.clip(np.searchsorted(sid, pid), 0, len(sid) - 1)
    hit = sid[pos] == pid
    if hit.any():
        sub = ch[ok][hit]
        text = (sub["title"].str.lower() + " " + sub["usertags"].str.lower().str.replace("+", " ", regex=False)
                .str.replace(",", " ", regex=False))
        key = sub["downloadurl"].str.extract(URL, expand=False).fillna("")
        cols["uid"].append(pd.util.hash_array(sub["uid"].values).astype(np.uint64))
        cols["flag"].append(text.str.contains(WORDS).values)
        cols["key"].append(key.values)
        cols["row"].append(pos[hit])
    done += len(ch)
    if done % (10 * a.chunk) == 0:
        print(f"    {done:,} rows read", flush=True)

row = np.concatenate(cols["row"])
D = pd.DataFrame({"place": s_place[row], "typ": s_typ[row], "peo": s_peo[row], "jnk": s_jnk[row],
                  "flag": np.concatenate(cols["flag"]), "uid": np.concatenate(cols["uid"]), "key": np.concatenate(cols["key"])})
D = D[D["key"] != ""].drop_duplicates("key")
D["score"] = D["typ"] - 0.6 * D["peo"] - 0.6 * D["jnk"] - 0.08 * D["flag"]
D["bad"] = (D["peo"] > 0.5) | (D["jnk"] > 0.5)
print(f"{len(D):,} photos with a link; flagged by tags {D['flag'].mean():.1%}, mostly people/junk {D['bad'].mean():.1%}", flush=True)

# good photos first, then by score; one per photographer; top K per place
D = D.sort_values(["place", "bad", "score"], ascending=[True, True, False])
D = D.drop_duplicates(["place", "uid"])
D["rank"] = D.groupby("place").cumcount()
D = D[D["rank"] < a.k]
np.savez_compressed(out_path, place=D["place"].values.astype(np.int32), rank=D["rank"].values.astype(np.int8),
                    key=D["key"].values.astype(str), score=D["score"].values.astype(np.float16), bad=D["bad"].values)
per = D.groupby("place").size()
print(f"saved {out_path}: {len(D):,} photos for {len(per):,} places; places with >= 3 photos {int((per >= 3).sum()):,}, "
      f"with {a.k}: {int((per >= a.k).sum()):,}")
