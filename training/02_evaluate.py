"""
02_evaluate.py -- FINAL test-set evaluation (run once per chosen model; do not tune on it).

Protocol: every test user has one held-out last visit. The model ranks ALL POIs (no candidate pool);
the user's own train+val POIs are removed from the ranking. History for the distance features =
train + val visits (all earlier than the test visit). Ties are resolved by expected rank.

  CUDA_VISIBLE_DEVICES=7 python training/02_evaluate.py --data data_v2 --run v4_lowlr
Outputs: runs/<run>/test_results.json  (+ printed table)
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NIL, GeoCtx, score_all, build_csr, evaluate, summarize, popularity_scorer, nearest_scorer, geopop_scorer

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_v2")
ap.add_argument("--run", required=True)
ap.add_argument("--travel-km", type=float, default=50.0, help="test visit this far from all history = 'new place' case")
a = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_dir = os.path.join("runs", a.run)
cfg = json.load(open(os.path.join(run_dir, "config.json")))
d = np.load(os.path.join(a.data, "prepared.npz"), allow_pickle=True)
n_users, n_pois = len(d["user_ids"]), len(d["poi_ids"])
U = torch.as_tensor(d["user_emb"], device=dev)
P = torch.as_tensor(d["poi_emb"], device=dev)
tr_u, tr_p = d["train_u"].astype(np.int64), d["train_p"].astype(np.int64)
va_u, va_p = d["val_u"].astype(np.int64), d["val_p"].astype(np.int64)
te_u, te_p = d["test_u"].astype(np.int64), d["test_p"].astype(np.int64)

# history = train + val, chronological per user (val is later than all train visits by construction)
h_u = np.concatenate([tr_u, va_u]); h_p = np.concatenate([tr_p, va_p])
o = np.argsort(h_u, kind="stable"); h_u, h_p = h_u[o], h_p[o]
pop = np.bincount(h_p, minlength=n_pois).astype(np.float64)
geo = GeoCtx(d["poi_lat"], d["poi_lon"], h_u, h_p, n_users, dev, M=cfg["geo_m"], pop=pop, v2=cfg.get("feat_v", 1) == 2)
hist_csr = build_csr(h_u, h_p, n_users)

no_clean = torch.as_tensor(d["poi_no_clean"].astype(bool), device=dev)
in_inv = ~d["poi_no_clean"].astype(bool)[te_p]               # test targets that existed before the hold-out
print(f"test visits to POIs with no earlier photos (not recommendable, excluded): {(~in_inv).mean():.1%}", flush=True)
te_u, te_p = te_u[in_inv], te_p[in_inv]

model = NIL(h=cfg["hidden"], drop=(cfg["drop"], cfg["drop"] + 0.1, cfg["drop"]), geo_dim=cfg.get("geo_dim", 5)).to(dev)
model.load_state_dict(torch.load(os.path.join(run_dir, "best.pth"), map_location=dev))
model.eval()


def model_scorer(ub):
    return score_all(model, model.user_tower(U[ub]), model.poi_tower(P), geo, ub)


scorers = {"AuraPath (this model)": model_scorer, "Popularity": popularity_scorer(pop),
           "Nearest place": nearest_scorer(geo, n_pois), "Popular within 25 km": geopop_scorer(geo, pop, n_pois)}
ranks = {}
for name, sc in scorers.items():
    m, r = evaluate(sc, te_u, te_p, [hist_csr], dev, bs=8, invalid=no_clean)
    ranks[name] = r
    print(f"done {name}", flush=True)

# subsets
warm = np.isin(te_p, np.unique(h_p))
with torch.no_grad():
    dt = []
    for s in range(0, len(te_u), 1024):
        u = torch.as_tensor(te_u[s:s + 1024], device=dev)
        p = torch.as_tensor(te_p[s:s + 1024], device=dev)[:, None]
        dt.append(geo.dist(u, p).min(-1).values[:, 0].cpu().numpy())
    dt = np.concatenate(dt)
travel = dt > a.travel_km
subsets = {"all test users with a known target POI": np.ones(len(te_u), bool), "target seen in history by others (warm)": warm,
           "target never visited in train/val (cold POI)": ~warm,
           f"new place (> {a.travel_km:.0f} km from all history)": travel, "local (<= %.0f km)" % a.travel_km: ~travel}

results = {"share_of_test_visits_to_brand_new_pois_excluded": float((~in_inv).mean())}
for sub, mask in subsets.items():
    results[sub] = {"n": int(mask.sum())}
    for name, r in ranks.items():
        results[sub][name] = summarize(r[mask]) if mask.sum() else {}


# paired bootstrap CI of NDCG@20 gain over nearest-place baseline
def nd20(r):
    return np.where(r <= 20, 1.0 / np.log2(r + 1.0), 0.0)


diff = nd20(ranks["AuraPath (this model)"]) - nd20(ranks["Nearest place"])
rng = np.random.RandomState(0)
bs = np.array([diff[rng.randint(0, len(diff), len(diff))].mean() for _ in range(1000)])
results["bootstrap_ndcg20_gain_vs_nearest"] = {"mean": float(diff.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
json.dump(results, open(os.path.join(run_dir, "test_results.json"), "w"), indent=1)

keys = ["HR@10(=Recall@10)", "HR@20(=Recall@20)", "Precision@10", "NDCG@10", "NDCG@20", "MRR(=MAP)"]
for sub in subsets:
    print(f"\n=== {sub}  (n={results[sub]['n']:,}) ===")
    print(f"{'method':26s}" + "".join(f"{k.split('(')[0]:>12s}" for k in keys))
    for name in ranks:
        m = results[sub][name]
        if m:
            print(f"{name:26s}" + "".join(f"{m[k]:12.4f}" for k in keys))
print("\nNDCG@20 gain over nearest-place baseline: %.4f  95%% CI [%.4f, %.4f]" % (
    results["bootstrap_ndcg20_gain_vs_nearest"]["mean"], *results["bootstrap_ndcg20_gain_vs_nearest"]["ci95"]))
print("saved", os.path.join(run_dir, "test_results.json"))
