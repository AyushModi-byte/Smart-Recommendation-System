"""
Does "people like you" (the swipe app's taste matching) find people with genuinely similar taste?

The app's taste vector is mean(embedding of liked places) - mean(embedding of all places), normalised
(server/engine.py Engine.taste), and users are matched by cosine. Tested here on real Flickr users from the
training visits (no test visits used):

1. Self-retrieval. Each user's places are split at random into two halves A and B. A good taste vector
   should find the same person: rank of B(u) among everyone's B for the query A(u). Random = N/2.
2. Same city. The same, but only against users whose places centre within 50 km of the user's, so matching by
   location alone cannot succeed. Compared with a location-only matcher on the full set.
3. Few likes. A(u) has only 3, 5 or 10 places, as for a new app user.
4. Scenery agreement. For each user's 5 nearest neighbours (full taste), how often they share the user's most
   frequent scenery type, against the chance rate for random pairs.

Run from the repo root:  python training/19_people_match.py --data data_v4 --out runs/people_match
"""
import argparse, json, os, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_v4")
ap.add_argument("--out", default="runs/people_match")
ap.add_argument("--users", type=int, default=3000)
ap.add_argument("--min_places", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(a.seed)
t0 = time.time()

z = np.load(f"{a.data}/prepared.npz", allow_pickle=False)
emb = z["poi_emb"].astype(np.float32)
lat, lon = z["poi_lat"].astype(np.float64), z["poi_lon"].astype(np.float64)
cluster = z["poi_cluster"].astype(np.int64)
tu, tp = z["train_u"].astype(np.int64), z["train_p"].astype(np.int64)
gmean = emb.mean(0)
print(f"{len(emb):,} places, {len(tu):,} train visits  ({time.time() - t0:.0f}s)", flush=True)

# distinct places per user
key = np.unique(tu * len(emb) + tp)
u_of, p_of = key // len(emb), key % len(emb)
starts = np.r_[0, np.flatnonzero(np.diff(u_of)) + 1]
counts = np.diff(np.r_[starts, len(u_of)])
ok = np.flatnonzero(counts >= a.min_places)
pick = rng.choice(ok, size=min(a.users, len(ok)), replace=False)
places = [p_of[starts[i]:starts[i] + counts[i]] for i in pick]
N = len(places)
print(f"{len(ok):,} users with >= {a.min_places} places; testing {N:,}", flush=True)


def taste(ix):
    v = emb[ix].mean(0) - gmean
    return v / max(np.linalg.norm(v), 1e-8)


def centre(ix):                                           # mean position on the unit sphere
    la, lo = np.radians(lat[ix]), np.radians(lon[ix])
    v = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1).mean(0)
    return v / max(np.linalg.norm(v), 1e-12)


def ranks(S):                                             # rank (1 = best) of the diagonal in each row
    return 1 + (S > np.diag(S)[:, None]).sum(1)


def summary(r, n=None):
    n = n if n is not None else N
    return {"recall@1": float((r <= 1).mean()), "recall@10": float((r <= 10).mean()),
            "median_rank": float(np.median(r)), "candidates": int(n),
            "random_recall@10": float(min(10, n) / n), "random_median_rank": float((n + 1) / 2)}


A, B, CA, CB, cap = [], [], [], [], 200                    # the app sends at most 200 likes
for ix in places:
    s = rng.permutation(ix)
    h = len(s) // 2
    A.append(s[:h][:cap]); B.append(s[h:][:cap])
TA = np.stack([taste(x) for x in A]); TB = np.stack([taste(x) for x in B])
CA = np.stack([centre(x) for x in A]); CB = np.stack([centre(x) for x in B])
res = {"users": N, "min_places": a.min_places, "places_per_user_median": float(np.median([len(x) for x in places]))}

# 1. self-retrieval, taste vs location only
res["self_retrieval_taste"] = summary(ranks(TA @ TB.T))
res["self_retrieval_location_only"] = summary(ranks(CA @ CB.T))

# 2. within 50 km: taste must separate people who photograph the same city
km = 6371 * np.arccos(np.clip(CA @ CB.T, -1, 1))
St = TA @ TB.T
r_city, n_city = [], []
for i in range(N):
    m = km[i] <= 50
    m[i] = True
    if m.sum() < 20:
        continue
    r_city.append(1 + (St[i, m] > St[i, i]).sum()); n_city.append(m.sum())
r_city, n_city = np.array(r_city), np.array(n_city)
res["same_city_taste"] = {"users_with_20+_neighbours": int(len(r_city)),
                          "recall@1": float((r_city <= 1).mean()), "recall@10": float((r_city <= 10).mean()),
                          "random_recall@1": float((1 / n_city).mean()), "random_recall@10": float((np.minimum(10, n_city) / n_city).mean()),
                          "median_candidates": float(np.median(n_city)),
                          "median_rank_percentile": float(np.median(r_city / n_city))}

# 3. few likes on the query side
res["few_likes"] = {}
for k in (3, 5, 10):
    Tk = np.stack([taste(rng.choice(x, size=min(k, len(x)), replace=False)) for x in A])
    res["few_likes"][str(k)] = summary(ranks(Tk @ TB.T))

# 4. scenery agreement of the 5 nearest neighbours (full histories)
TF = np.stack([taste(x[:cap]) for x in places])
dom = np.array([np.bincount(cluster[x], minlength=cluster.max() + 1).argmax() for x in places])
S = TF @ TF.T
np.fill_diagonal(S, -np.inf)
nn = np.argsort(-S, 1)[:, :5]
share = np.bincount(dom, minlength=cluster.max() + 1) / N
res["scenery_agreement_top5"] = {"same_main_scenery": float((dom[nn] == dom[:, None]).mean()),
                                 "random_pairs": float((share ** 2).sum()),
                                 "main_scenery_shares": [round(float(x), 3) for x in share]}
res["seconds"] = round(time.time() - t0, 1)

print(json.dumps(res, indent=2))
with open(f"{a.out}/results.json", "w") as f:
    json.dump(res, f, indent=2)
