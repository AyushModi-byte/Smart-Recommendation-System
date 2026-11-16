"""
00a_split_places.py -- split oversized "places" (NB5 DBSCAN chains whole city centres into one POI).

1. Places with radius <= --max-km keep their id.
2. Larger places are re-clustered by density peaks (a grid version of mean-shift, bandwidth --bw-m
   metres): each peak becomes a place and every photo joins its nearest peak. Anything still larger than --max-km
   is cut into --grid-m grid cells.
3. Fuzzy memberships: unchanged places keep theirs. New places get one predicted from their mean
   visual embedding by a linear model fitted on the unchanged places (hold-out accuracy is printed).

Writes NEW files only (the v1 files are not touched):
  brain/photo_poi_mapping_v2.csv         photoid, longitude, latitude, poi_id
  brain/poi_parent_v2.csv                poi_id, parent_poi_id   (new places only)
  brain/poi_fuzzy_memberships_v2.npy, brain/poi_cluster_ids_v2.npy

  python training/00a_split_places.py --brain brain
"""
import argparse, glob, json, os, re, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge

ap = argparse.ArgumentParser()
ap.add_argument("--brain", default="brain")
ap.add_argument("--max-km", type=float, default=0.25)
ap.add_argument("--bw-m", type=float, default=75.0)
ap.add_argument("--grid-m", type=float, default=150.0)
a = ap.parse_args()
B = a.brain
t0 = time.time()


def radius_km(df, key):
    g = df.groupby(key)
    lat0, lon0 = g["latitude"].transform("mean").values, g["longitude"].transform("mean").values
    dy = (df["latitude"].values - lat0) * 111.32
    dx = (df["longitude"].values - lon0) * 111.32 * np.cos(np.radians(lat0))
    r = pd.Series(np.sqrt(dx ** 2 + dy ** 2), index=df.index)
    return r.groupby(df[key]).max()


def report(df, key, title):
    r = radius_km(df, key)
    n = df.groupby(key).size()
    print(f"  {title}: places={len(n):,}  radius km p50={r.median():.3f} p99={r.quantile(.99):.3f} max={r.max():.2f}  "
          f"largest place={n.max():,} photos ({n.max() / len(df):.2%})  places>{a.max_km}km: {(r > a.max_km).sum():,} "
          f"holding {n[r > a.max_km].sum() / len(df):.1%} of photos", flush=True)
    return r, n


def density_peaks(X):
    """Mean-shift equivalent on a grid (fast): smooth a 25 m photo histogram, take local maxima at least
    --bw-m apart that have >= 3 photos within --bw-m, and give every photo to its nearest peak."""
    cell = 25.0
    ij = np.floor((X - X.min(0)) / cell).astype(np.int64) + 4
    H = np.zeros(ij.max(0) + 5, dtype=np.float64)
    np.add.at(H, (ij[:, 0], ij[:, 1]), 1.0)
    D = ndimage.gaussian_filter(H, sigma=a.bw_m / (2 * cell), mode="constant")
    k = 2 * int(round(a.bw_m / cell)) + 1
    is_max = (D == ndimage.maximum_filter(D, size=k, mode="constant")) & (D > 1e-9)
    lab_img, n_lab = ndimage.label(is_max)                       # a flat top counts once
    if n_lab == 0:
        return np.zeros(len(X), dtype=np.int64)
    peaks = np.array(ndimage.center_of_mass(is_max, lab_img, np.arange(1, n_lab + 1)))
    peaks = (peaks - 4 + 0.5) * cell + X.min(0)
    tree = cKDTree(X)
    support = tree.query_ball_point(peaks, a.bw_m, return_length=True)
    if (support >= 3).any():
        peaks = peaks[support >= 3]
    return cKDTree(peaks).query(X, k=1)[1].astype(np.int64)


print("[1/4] loading mapping", flush=True)
df = pd.read_csv(f"{B}/photo_poi_mapping.csv")
df = df.drop_duplicates("photoid").reset_index(drop=True)
r, n = report(df, "poi_id", "before")
big = r.index[r > a.max_km].values
print(f"  to split: {len(big):,} places, {n.loc[big].sum():,} photos", flush=True)

print("[2/4] density peaks inside each oversized place", flush=True)
new_id = df["poi_id"].values.copy()
parent = {}
next_id = int(df["poi_id"].max()) + 1
sub = np.nonzero(df["poi_id"].isin(big).values)[0]                 # row positions in df
rows_of = {k: sub[v] for k, v in df.iloc[sub].groupby("poi_id").indices.items()}
for k, pid in enumerate(sorted(big, key=lambda p: -n.loc[p])):
    idx = rows_of[pid]
    lat, lon = df["latitude"].values[idx], df["longitude"].values[idx]
    y = (lat - lat.mean()) * 111320.0
    x = (lon - lon.mean()) * 111320.0 * np.cos(np.radians(lat.mean()))
    X = np.stack([x, y], axis=1)
    lab = density_peaks(X)
    # safety net: any sub-place still too large is cut into grid cells
    out = lab.astype(np.int64).copy()
    for c in np.unique(lab):
        m = lab == c
        cx, cy = X[m].mean(0)
        if np.sqrt(((X[m] - [cx, cy]) ** 2).sum(1)).max() > a.max_km * 1000.0:
            cell = np.floor(X[m] / a.grid_m).astype(np.int64)
            out[m] = (c + 1) * 10_000_000 + (cell[:, 0] - cell[:, 0].min()) * 3000 + (cell[:, 1] - cell[:, 1].min())
    uniq, inv = np.unique(out, return_inverse=True)
    new_id[idx] = next_id + inv
    for j in range(len(uniq)):
        parent[next_id + j] = int(pid)
    next_id += len(uniq)
    if k < 5 or k % 500 == 0:
        print(f"    {k + 1}/{len(big)}  place {pid}: {len(idx):,} photos -> {len(uniq):,} places  [{time.time() - t0:.0f}s]", flush=True)

df["poi_id"] = new_id
r2, n2 = report(df, "poi_id", "after ")
df.to_csv(f"{B}/photo_poi_mapping_v2.csv", index=False)
pd.DataFrame({"poi_id": list(parent.keys()), "parent_poi_id": list(parent.values())}).to_csv(f"{B}/poi_parent_v2.csv", index=False)

print("[3/4] mean visual embedding per place (aligned with quintuplet_embeddings.npy)", flush=True)
old = pd.read_csv(f"{B}/photo_poi_mapping.csv", usecols=["photoid", "poi_id"]).drop_duplicates("photoid")
old_of = pd.Series(old["poi_id"].values, index=old["photoid"].values)
files = sorted(glob.glob(f"{B}/ids/ids_chunk_*.npy"), key=lambda f: int(re.findall(r"(\d+)\.npy$", f)[0]))
pids = []
for f in files:
    ids = np.load(f, allow_pickle=True).astype(np.int64)
    pids.append(ids[old_of.reindex(ids).notna().values])
pids = np.concatenate(pids)
vis = np.load(f"{B}/quintuplet_embeddings.npy", mmap_mode="r")
assert len(pids) == vis.shape[0], (len(pids), vis.shape)
new_of = pd.Series(df["poi_id"].values, index=df["photoid"].values)
row_poi = new_of.reindex(pids).values.astype(np.int64)
ids_v2, inv = np.unique(row_poi, return_inverse=True)
S = np.zeros((len(ids_v2), vis.shape[1]), dtype=np.float64)
for s in range(0, len(pids), 400_000):
    k = inv[s:s + 400_000]
    M = sp.csr_matrix((np.ones(len(k)), (k, np.arange(len(k)))), shape=(len(ids_v2), len(k)))
    S += M @ np.asarray(vis[s:s + 400_000], dtype=np.float64)
mean_vis = (S / np.bincount(inv, minlength=len(ids_v2))[:, None]).astype(np.float32)

print("[4/4] memberships for the new places", flush=True)
mem = np.load(f"{B}/poi_fuzzy_memberships.npy").astype(np.float32)
mem_ids = np.load(f"{B}/poi_cluster_ids.npy").astype(np.int64)
pos = pd.Series(np.arange(len(ids_v2)), index=ids_v2)
kept = np.isin(mem_ids, ids_v2)                                     # clustered places that were not split
Xk, Yk = mean_vis[pos.loc[mem_ids[kept]].values], mem[kept]
rng = np.random.RandomState(0)
perm = rng.permutation(len(Xk))
cut = int(0.9 * len(Xk))
model = Ridge(alpha=1.0).fit(Xk[perm[:cut]], Yk[perm[:cut]])


def predict(X):
    p = np.clip(model.predict(X), 1e-4, None)
    return (p / p.sum(1, keepdims=True)).astype(np.float32)


ph = predict(Xk[perm[cut:]])
acc = float((ph.argmax(1) == Yk[perm[cut:]].argmax(1)).mean())
mae = float(np.abs(ph - Yk[perm[cut:]]).mean())
print(f"  membership predictor on held-out unchanged places: top-cluster agreement {acc:.1%}, MAE {mae:.3f} "
      f"(chance {np.bincount(Yk.argmax(1)).max() / len(Yk):.1%})", flush=True)
model = Ridge(alpha=1.0).fit(Xk, Yk)
par = pd.Series(parent)
child = par.index[par.isin(mem_ids)].values                         # new places whose parent had a membership
child = child[np.isin(child, ids_v2)]
mem_child = predict(mean_vis[pos.loc[child].values])
out_ids = np.concatenate([mem_ids[kept], child])
out_mem = np.concatenate([mem[kept], mem_child])
o = np.argsort(out_ids)
np.save(f"{B}/poi_cluster_ids_v2.npy", out_ids[o])
np.save(f"{B}/poi_fuzzy_memberships_v2.npy", out_mem[o])
stats = {"places_before": int(len(n)), "places_after": int(len(n2)), "split": int(len(big)), "new_places": int(len(parent)),
         "largest_before": int(n.max()), "largest_after": int(n2.max()), "radius_max_after_km": float(r2.max()),
         "places_with_membership_v2": int(len(out_ids)), "membership_predictor_top1": acc, "membership_predictor_mae": mae,
         "max_km": a.max_km, "bw_m": a.bw_m, "grid_m": a.grid_m, "seconds": round(time.time() - t0, 1)}
json.dump(stats, open(f"{B}/split_places_v2.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
