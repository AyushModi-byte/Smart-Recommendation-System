"""
14_swipe_check.py -- can "accepted places" stand in for a user's own photos? (offline check for the swipe app)

In the app a new user has no photos: their taste is built from the places they accept. This script evaluates the
trained model on the usual test users with that substitution, no retraining:
  * own photos            : the normal model input (mean of the user's own train photos)
  * all visited places    : taste = mean embedding of every place in the user's history
  * last k accepted       : a new app user who accepted only k places: taste AND location history from the last k
                            places of their history, and only those k are excluded from the ranking
Baselines (popular within 25 km, nearest) are computed from the same restricted history, so each k is a fair comparison.

  CUDA_VISIBLE_DEVICES=0 python training/14_swipe_check.py --data data_v4 --run p1_split
Outputs: runs/<run>/swipe_check.json
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NIL, GeoCtx, score_all, build_csr, evaluate, nearest_scorer, geopop_scorer, RunData

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_v4")
ap.add_argument("--run", default="p1_split")
ap.add_argument("--n", type=int, default=6000, help="test users sampled (same sample for every variant)")
ap.add_argument("--ks", default="1,3,5,10")
a = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_dir = os.path.join("runs", a.run)
cfg = json.load(open(os.path.join(run_dir, "config.json")))
d = RunData(a.data, run_dir)
n_users, n_pois = len(d["user_ids"]), len(d["poi_ids"])
U = torch.as_tensor(d["user_emb"], device=dev)
P = torch.as_tensor(d["poi_emb"], device=dev)
tr_u, tr_p = d["train_u"].astype(np.int64), d["train_p"].astype(np.int64)
va_u, va_p = d["val_u"].astype(np.int64), d["val_p"].astype(np.int64)
te_u, te_p = d["test_u"].astype(np.int64), d["test_p"].astype(np.int64)
# history = train + val, in time order within each user (train is time-sorted, val is the later visit)
h_u = np.concatenate([tr_u, va_u]); h_p = np.concatenate([tr_p, va_p])
o = np.argsort(h_u, kind="stable"); h_u, h_p = h_u[o], h_p[o]
pop = np.bincount(h_p, minlength=n_pois).astype(np.float64)       # place popularity stays global
v2 = cfg.get("feat_v", 1) == 2
no_clean_np = d["poi_no_clean"].astype(bool)
no_clean = torch.as_tensor(no_clean_np, device=dev)
ok = ~no_clean_np[te_p]
te_u, te_p = te_u[ok], te_p[ok]
sel = np.random.RandomState(0).permutation(len(te_u))[:a.n]
te_u, te_p = te_u[sel], te_p[sel]

model = NIL(user_dim=U.shape[1], poi_dim=P.shape[1], h=cfg["hidden"], drop=(cfg["drop"], cfg["drop"] + 0.1, cfg["drop"]),
            geo_dim=cfg.get("geo_dim", 5)).to(dev)
model.load_state_dict(torch.load(os.path.join(run_dir, "best.pth"), map_location=dev))
model.eval()
with torch.no_grad():
    PT = model.poi_tower(P)


def last_k(k):
    """keep only the last k history places of each user (k=0 -> all)"""
    if k <= 0:
        return h_u, h_p
    starts = np.r_[0, np.nonzero(np.diff(h_u))[0] + 1]
    ends = np.r_[starts[1:], len(h_u)]
    pos_from_end = np.empty(len(h_u), dtype=np.int64)
    for s, e in zip(starts, ends):
        pos_from_end[s:e] = np.arange(e - s)[::-1]
    keep = pos_from_end < k
    return h_u[keep], h_p[keep]


def place_taste(u, p):
    """mean place embedding over each user's (restricted) history -> (n_users, dim)"""
    s = np.zeros((n_users, P.shape[1]), dtype=np.float64)
    np.add.at(s, u, d["poi_emb"][p].astype(np.float64))
    c = np.bincount(u, minlength=n_users)[:, None]
    return torch.as_tensor((s / np.maximum(c, 1)).astype(np.float32), device=dev)


def run(name, Uv, hu, hp, res, baselines):
    geo = GeoCtx(d["poi_lat"], d["poi_lon"], hu, hp, n_users, dev, M=cfg["geo_m"], pop=pop, v2=v2)
    hist = build_csr(hu, hp, n_users)
    scorers = {"AuraPath": lambda ub: score_all(model, model.user_tower(Uv[ub]), PT, geo, ub)}
    if baselines:
        scorers["Popular within 25 km"] = geopop_scorer(geo, pop, n_pois)
        scorers["Nearest place"] = nearest_scorer(geo, n_pois)
    res[name] = {}
    for sn, sc in scorers.items():
        m, _ = evaluate(sc, te_u, te_p, [hist], dev, bs=8, invalid=no_clean)
        res[name][sn] = m
        print(f"{name:28s}{sn:24s}{m['HR@10(=Recall@10)']:8.4f}{m['HR@20(=Recall@20)']:8.4f}{m['NDCG@20']:9.4f}"
              f"{m['MRR(=MAP)']:8.4f}", flush=True)


res = {"n_users": int(len(te_u))}
print(f"{len(te_u):,} test users\n{'variant':28s}{'method':24s}{'HR@10':>8s}{'HR@20':>8s}{'NDCG@20':>9s}{'MRR':>8s}")
run("own photos (normal)", U, h_u, h_p, res, baselines=True)
run("all visited places", place_taste(h_u, h_p), h_u, h_p, res, baselines=False)
for k in [int(x) for x in a.ks.split(",")]:
    hu, hp = last_k(k)
    run(f"last {k} accepted", place_taste(hu, hp), hu, hp, res, baselines=True)
json.dump(res, open(os.path.join(run_dir, "swipe_check.json"), "w"), indent=1)
print("saved", os.path.join(run_dir, "swipe_check.json"))
