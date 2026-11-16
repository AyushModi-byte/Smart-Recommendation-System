"""
04_ablate_content.py -- how much does the VISUAL signal (MEAL embeddings + fuzzy memberships) actually contribute?
Evaluates the trained model on the same test users with the content deliberately damaged (no retraining):
  normal | users shuffled (taste of another user) | POI content neutral (global mean) | both
If the score barely drops, improving the quintuplet/MEAL embeddings cannot raise the metric much.

  CUDA_VISIBLE_DEVICES=7 python training/04_ablate_content.py --data data_v3 --run v7
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NIL, GeoCtx, score_all, build_csr, evaluate, RunData

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_v3")
ap.add_argument("--run", default="v7")
ap.add_argument("--n", type=int, default=4000)
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
h_u = np.concatenate([tr_u, va_u]); h_p = np.concatenate([tr_p, va_p])
o = np.argsort(h_u, kind="stable"); h_u, h_p = h_u[o], h_p[o]
geo = GeoCtx(d["poi_lat"], d["poi_lon"], h_u, h_p, n_users, dev, M=cfg["geo_m"], pop=np.bincount(h_p, minlength=n_pois),
             v2=cfg.get("feat_v", 1) == 2)
hist = build_csr(h_u, h_p, n_users)
no_clean = torch.as_tensor(d["poi_no_clean"].astype(bool), device=dev)
ok = ~d["poi_no_clean"].astype(bool)[te_p]
te_u, te_p = te_u[ok], te_p[ok]
sel = np.random.RandomState(0).permutation(len(te_u))[:a.n]
te_u, te_p = te_u[sel], te_p[sel]

model = NIL(user_dim=U.shape[1], poi_dim=P.shape[1], h=cfg["hidden"], drop=(cfg["drop"], cfg["drop"] + 0.1, cfg["drop"]), geo_dim=cfg.get("geo_dim", 5)).to(dev)
model.load_state_dict(torch.load(os.path.join(run_dir, "best.pth"), map_location=dev))
model.eval()
perm = torch.as_tensor(np.random.RandomState(1).permutation(n_users), device=dev)
P_neutral = torch.as_tensor(d["g_content"], device=dev)[None].expand_as(P).contiguous()

variants = {"normal": (U, P), "user taste shuffled": (U[perm], P), "POI content neutral": (U, P_neutral),
            "both damaged": (U[perm], P_neutral)}
print(f"{'variant':24s}{'HR@10':>8s}{'HR@20':>8s}{'NDCG@20':>9s}{'MRR':>8s}")
res = {}
for name, (Uv, Pv) in variants.items():
    with torch.no_grad():
        PT = model.poi_tower(Pv)
    sc = lambda ub: score_all(model, model.user_tower(Uv[ub]), PT, geo, ub)
    m, _ = evaluate(sc, te_u, te_p, [hist], dev, bs=8, invalid=no_clean)
    res[name] = m
    print(f"{name:24s}{m['HR@10(=Recall@10)']:8.4f}{m['HR@20(=Recall@20)']:8.4f}{m['NDCG@20']:9.4f}{m['MRR(=MAP)']:8.4f}", flush=True)
json.dump(res, open(os.path.join(run_dir, "ablation_content.json"), "w"), indent=1)
