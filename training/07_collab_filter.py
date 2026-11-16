"""
07_collab_filter.py -- is there collaborative signal? (ROADMAP Phase 2, step 1)

Item-item collaborative filtering under the same protocol as 02_evaluate.py:
  sim(i, j) = sum_u w_u [u visited i][u visited j] / (sqrt(n_i n_j) + shrink),  w_u = 1 / log(2 + visits of u)
  score(u, j) = sum over the user's history i of sim(i, j)
Reported: CF alone, the trained model alone, and model + lam * CF with lam tuned on VALIDATION users
(history = train, similarities from train; model scores z-scored per user) and applied once to TEST (history = train + val).

  CUDA_VISIBLE_DEVICES=0 python training/07_collab_filter.py --data data_v4 --run p1_split
Outputs: runs/<run>/collab_results.json
"""
import argparse, json, os, sys
import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NIL, GeoCtx, score_all, build_csr, evaluate, summarize, RunData

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_v4")
ap.add_argument("--run", required=True)
ap.add_argument("--cap", type=int, default=200, help="most recent visits per user used to build similarities")
ap.add_argument("--shrink", type=float, default=2.0)
ap.add_argument("--val-users", type=int, default=3000)
ap.add_argument("--bs", type=int, default=32)
a = ap.parse_args()
LAMS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

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
no_clean_np = d["poi_no_clean"].astype(bool)
no_clean = torch.as_tensor(no_clean_np, device=dev)
h_u = np.concatenate([tr_u, va_u]); h_p = np.concatenate([tr_p, va_p])
o = np.argsort(h_u, kind="stable"); h_u, h_p = h_u[o], h_p[o]

model = NIL(user_dim=U.shape[1], poi_dim=P.shape[1], h=cfg["hidden"], drop=(cfg["drop"], cfg["drop"] + 0.1, cfg["drop"]), geo_dim=cfg.get("geo_dim", 5)).to(dev)
model.load_state_dict(torch.load(os.path.join(run_dir, "best.pth"), map_location=dev))
model.eval()
with torch.no_grad():
    PT = model.poi_tower(P)
v2 = cfg.get("feat_v", 1) == 2


def build_sim(u, p):
    """u, p: visits grouped by user, chronological. Returns (history csr U x P, similarity csr P x P)."""
    cnt = np.bincount(u, minlength=n_users)
    start = np.concatenate([[0], np.cumsum(cnt)])[u]
    from_end = cnt[u] - 1 - (np.arange(len(u)) - start)
    X = sp.csr_matrix((np.ones(len(u), dtype=np.float32), (u, p)), shape=(n_users, n_pois))
    X.data[:] = 1.0
    k = from_end < a.cap
    Xc = sp.csr_matrix((np.ones(k.sum(), dtype=np.float32), (u[k], p[k])), shape=(n_users, n_pois))
    Xc.data[:] = 1.0
    w = 1.0 / np.log(2.0 + cnt)
    C = (sp.diags(w.astype(np.float32)) @ Xc).T.tocsr() @ Xc
    C.setdiag(0); C.eliminate_zeros()
    n_i = np.asarray(Xc.sum(0)).ravel()
    C = C.tocoo()
    C.data = C.data / (np.sqrt(n_i[C.row] * n_i[C.col]) + a.shrink)
    S = C.tocsr().astype(np.float32)
    print(f"    visits={len(u):,} (capped {int(k.sum()):,})  similarity nnz={S.nnz:,}  places with a neighbour={int((S.getnnz(1) > 0).sum()):,}", flush=True)
    return X, S


def scorers(geo, X, S):
    def cf(ub):
        return torch.as_tensor((X[ub.cpu().numpy()] @ S).toarray(), device=dev)

    def mod(ub):
        return score_all(model, model.user_tower(U[ub]), PT, geo, ub)
    return cf, mod


def z(x):
    return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp(min=1e-6)


def run(users, targets, geo, X, S, csr, lams):
    cf, mod = scorers(geo, X, S)
    cache = {}

    def both(ub):
        key = int(ub[0]), len(ub)
        if key not in cache:
            cache.clear()
            c = cf(ub)                                  # raw: mostly zeros, so it is not z-scored
            cache[key] = (c, z(mod(ub)), c)
        return cache[key]
    out = {}
    out["collaborative filtering alone"] = evaluate(lambda ub: both(ub)[0].clone(), users, targets, [csr], dev, bs=a.bs, invalid=no_clean)[1]
    for lam in lams:
        out[lam] = evaluate(lambda ub: both(ub)[1] + lam * both(ub)[2], users, targets, [csr], dev, bs=a.bs, invalid=no_clean)[1]
    return out


nd20 = lambda r: float(np.where(r <= 20, 1.0 / np.log2(r + 1.0), 0.0).mean())

print("[1/2] VALIDATION: similarities from train, history = train", flush=True)
Xtr, Str = build_sim(tr_u, tr_p)
geo_tr = GeoCtx(d["poi_lat"], d["poi_lon"], tr_u, tr_p, n_users, dev, M=cfg["geo_m"], pop=np.bincount(tr_p, minlength=n_pois), v2=v2)
ok = ~no_clean_np[va_p]
vu, vp = va_u[ok], va_p[ok]
sel = np.random.RandomState(0).permutation(len(vu))[:a.val_users]
vr = run(vu[sel], vp[sel], geo_tr, Xtr, Str, build_csr(tr_u, tr_p, n_users), LAMS)
for lam in LAMS:
    print(f"    lam={lam:<5} val NDCG@20 {nd20(vr[lam]):.4f}", flush=True)
print(f"    CF alone     val NDCG@20 {nd20(vr['collaborative filtering alone']):.4f}", flush=True)
best = max(LAMS, key=lambda l: nd20(vr[l]))
print(f"    best lam = {best}", flush=True)

print("[2/2] TEST: similarities from train + val, history = train + val", flush=True)
Xh, Sh = build_sim(h_u, h_p)
geo_h = GeoCtx(d["poi_lat"], d["poi_lon"], h_u, h_p, n_users, dev, M=cfg["geo_m"], pop=np.bincount(h_p, minlength=n_pois), v2=v2)
ok = ~no_clean_np[te_p]
tu, tp = te_u[ok], te_p[ok]
tr = run(tu, tp, geo_h, Xh, Sh, build_csr(h_u, h_p, n_users), sorted({0.0, best}))
has_nb = np.asarray((Xh[tu] @ Sh).getnnz(1) > 0).ravel()
tgt_nb = np.asarray(Sh.getnnz(1) > 0).ravel()[tp]
names = {"collaborative filtering alone": tr["collaborative filtering alone"], "model alone": tr[0.0], f"model + {best} * CF": tr[best]}
res = {"best_lam": best, "val_ndcg20": {str(l): nd20(vr[l]) for l in LAMS}, "n_test": int(len(tu)),
       "test_users_with_any_cf_score": float(has_nb.mean()), "test_targets_with_a_neighbour": float(tgt_nb.mean()), "methods": {}}
print(f"\n    test users with any CF score: {has_nb.mean():.1%}   test targets that have a neighbour: {tgt_nb.mean():.1%}")
print(f"{'method':32s}{'HR@10':>8s}{'HR@20':>8s}{'NDCG@20':>9s}{'MRR':>8s}")
for n, r in names.items():
    m = summarize(r)
    res["methods"][n] = m
    print(f"{n:32s}{m['HR@10(=Recall@10)']:8.4f}{m['HR@20(=Recall@20)']:8.4f}{m['NDCG@20']:9.4f}{m['MRR(=MAP)']:8.4f}")
diff = np.where(tr[best] <= 20, 1.0 / np.log2(tr[best] + 1.0), 0.0) - np.where(tr[0.0] <= 20, 1.0 / np.log2(tr[0.0] + 1.0), 0.0)
rng = np.random.RandomState(0)
bs = np.array([diff[rng.randint(0, len(diff), len(diff))].mean() for _ in range(1000)])
res["ndcg20_gain_over_model"] = {"mean": float(diff.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
print("NDCG@20 gain of the blend over the model: %.4f  95%% CI [%.4f, %.4f]" % (diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
json.dump(res, open(os.path.join(run_dir, "collab_results.json"), "w"), indent=1)
print("saved", os.path.join(run_dir, "collab_results.json"))
