"""
15_place_details.py -- extra per-place facts for the swipe app's place card (runs on the laptop, pandas only).

For every place in the bundle (same order as bundle/pois.npz):
  * n_photos, n_visitors        : photos and distinct photographers (all users, not only modelled ones)
  * visitors_by_month (P, 12)   : distinct photographers per calendar month (month of datetaken)
  * best_months                 : e.g. "MAY-JUN & SEP" when the place has enough visitors, else ""
  * city, region, country, cc   : the place's town (towns.py: a big nearby city beats a closer suburb)
  * tags (P, 5)                 : top decoded tags from poi_tags<places>.csv
Photographer ids are only hashed in memory to count distinct people; nothing user-level is written.

Needs: brain/usa_europe_geotagged.csv, brain/photo_poi_mapping<places>.csv, brain/poi_tags<places>.csv,
       staging/geonames/{cities15000.txt, admin1CodesASCII.txt, countryInfo.txt}, bundle/pois.npz
  .venv\\Scripts\\python.exe training\\15_place_details.py
Outputs: brain/poi_details<places>.npz (a new file; nothing existing is overwritten)
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--brain", default="brain")
ap.add_argument("--places", default="_v2")
ap.add_argument("--bundle", default="bundle")
ap.add_argument("--geonames", default="staging/geonames")
ap.add_argument("--chunk", type=int, default=1_000_000)
ap.add_argument("--nrows", type=int, default=None, help="smoke test: read only this many CSV rows, write to the scratch path")
ap.add_argument("--min-visitors", type=int, default=12, help="fewer distinct visitors -> no best-months line")
a = ap.parse_args()

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
out_path = f"{a.brain}/poi_details{a.places}.npz" if a.nrows is None else os.environ.get("TMP", ".") + "/poi_details_smoke.npz"
assert not os.path.exists(out_path), f"{out_path} exists; refusing to overwrite"

B = np.load(f"{a.bundle}/pois.npz")
poi_ids, lat, lon = B["poi_ids"], B["lat"].astype(np.float64), B["lon"].astype(np.float64)
P = len(poi_ids)

# ---------- photos -> place index ----------
print("[1/4] photo -> place mapping", flush=True)
pm = pd.read_csv(f"{a.brain}/photo_poi_mapping{a.places}.csv", usecols=["photoid", "poi_id"]).drop_duplicates("photoid")
pm = pm[np.isin(pm["poi_id"].values, poi_ids)]
ph_ids = pm["photoid"].values.astype(np.int64)
o = np.argsort(ph_ids); ph_ids = ph_ids[o]
ph_p = np.searchsorted(poi_ids, pm["poi_id"].values[o]).astype(np.int64)
print(f"    {len(ph_ids):,} photos at {P:,} places", flush=True)

# ---------- stream the raw CSV: photographer + month ----------
print("[2/4] streaming the raw photo CSV (photoid, uid, datetaken)", flush=True)
keys, mkeys = [], []
n_photos = np.zeros(P, dtype=np.int64)
done = 0
for ch in pd.read_csv(f"{a.brain}/usa_europe_geotagged.csv", usecols=["photoid", "uid", "datetaken"], chunksize=a.chunk, nrows=a.nrows,
                      dtype={"uid": str, "datetaken": str}, keep_default_na=False):
    pid = pd.to_numeric(ch["photoid"], errors="coerce").values
    ok = np.isfinite(pid)
    pid = pid[ok].astype(np.int64)
    pos = np.clip(np.searchsorted(ph_ids, pid), 0, len(ph_ids) - 1)
    hit = ph_ids[pos] == pid
    if hit.any():
        p = ph_p[pos[hit]]
        uh = pd.util.hash_array(ch["uid"].values[ok][hit]).astype(np.uint64) >> np.uint64(24)   # 40-bit person hash
        mon = pd.to_numeric(pd.Series(ch["datetaken"].values[ok][hit]).str[5:7], errors="coerce").values
        np.add.at(n_photos, p, 1)
        k = (p.astype(np.uint64) << np.uint64(40)) | uh
        keys.append(np.unique(k))
        good = np.isfinite(mon) & (mon >= 1) & (mon <= 12)
        mk = k[good] * np.uint64(16) + (mon[good].astype(np.uint64) - np.uint64(1))
        mkeys.append(np.unique(mk))
    done += len(ch)
    if done % (10 * a.chunk) == 0:
        print(f"    {done:,} rows read", flush=True)
keys = np.unique(np.concatenate(keys))
n_visitors = np.bincount((keys >> np.uint64(40)).astype(np.int64), minlength=P)
mkeys = np.unique(np.concatenate(mkeys))
mp = ((mkeys // np.uint64(16)) >> np.uint64(40)).astype(np.int64)
mm = (mkeys % np.uint64(16)).astype(np.int64)
vbm = np.zeros((P, 12), dtype=np.int32)
np.add.at(vbm, (mp, mm), 1)
print(f"    {n_photos.sum():,} photos matched; median visitors per place {int(np.median(n_visitors))}", flush=True)


def best_months(row, n):
    """months with at least 70% of the busiest month's visitors, as ranges: 'MAY-JUN & SEP'"""
    if n < a.min_visitors or row.max() == 0:
        return ""
    on = row >= 0.7 * row.max()
    if on.sum() > 6:                      # busy all year: no useful season
        return "ALL YEAR"
    idx = list(np.nonzero(on)[0])
    runs, cur = [], [idx[0]]
    for m in idx[1:]:
        if m == cur[-1] + 1:
            cur.append(m)
        else:
            runs.append(cur); cur = [m]
    runs.append(cur)
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == 11:   # wrap DEC-JAN
        runs[0] = runs.pop() + runs[0]
    parts = [MONTHS[r[0]] if len(r) == 1 else f"{MONTHS[r[0]]}-{MONTHS[r[-1]]}" for r in runs]
    return " & ".join(parts[:3])


best = np.array([best_months(vbm[i], n_visitors[i]) for i in range(P)])

# ---------- town ----------
print("[3/4] town for each place (GeoNames, population >= 15,000)", flush=True)
from towns import assign
_t = assign(lat, lon, a.geonames)
city, region, country, cc, near_km = _t["city"], _t["region"], _t["country"], _t["cc"], _t["town_km"]

# ---------- tags ----------
print("[4/4] top tags", flush=True)
T = pd.read_csv(f"{a.brain}/poi_tags{a.places}.csv", keep_default_na=False, dtype={"tag": str})
T = T[T["p"] < P]
T["r"] = T.groupby("p").cumcount()
T = T[T["r"] < 5]
tags = np.full((P, 5), "", dtype=object)
tags[T["p"].values, T["r"].values] = T["tag"].values
tags = tags.astype(str)

np.savez_compressed(out_path, poi_ids=poi_ids, n_photos=n_photos.astype(np.int32), n_visitors=n_visitors.astype(np.int32),
                    visitors_by_month=vbm, best_months=best, city=city, region=region, country=country, cc=cc,
                    town_km=near_km, tags=tags)
print("saved", out_path)
for i in np.argsort(-n_visitors)[:8]:
    print(f"  {B['name'][i][:30]:30s} {city[i]}, {region[i]}, {cc[i]} ({near_km[i]:.0f} km)  photos={n_photos[i]} "
          f"visitors={n_visitors[i]} best={best[i]} tags={list(tags[i])}")
