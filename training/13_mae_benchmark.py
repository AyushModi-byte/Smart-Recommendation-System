"""
13_mae_benchmark.py -- the proposal's MAE benchmark, defined for a ranking model.

The proposal targets "a 60% reduction in MAE". The model ranks places and does not predict ratings, so MAE is
defined on visit probability: each test user gets their held-out place plus 99 places they did not visit
(uniform over valid places outside their history). A method spreads probability 1 over the 100 candidates;
MAE = mean |1[visited] - p| over all candidates.
  * AuraPath: softmax of the model's scores, one temperature tuned on VALIDATION users (by log-loss)
  * Popularity: p proportional to visits + 1
  * Popular within 25 km: the same, restricted to places within 25 km of the user's recent history
  * Uniform: p = 1/100
Reported with the same split as 02_evaluate.py (history = train + val for the test users).

  CUDA_VISIBLE_DEVICES=0 python training/13_mae_benchmark.py --data data_v4 --run p1_split
Outputs: runs/<run>/mae_results.json
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NIL, GeoCtx, score_all, build_csr, RunData

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_v4")
ap.add_argument("--run", required=True)
ap.add_argument("--n-neg", type=int, default=99)
ap.add_argument("--val-users", type=int, default=3000)
ap.add_argument("--bs", type=int, default=32)
a = ap.parse_args()
TEMPS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

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
h_u = np.concatenate([tr_u, va_u]); h_p = np.concatenate([tr_p, va_p])
o = np.argsort(h_u, kind="stable"); h_u, h_p = h_u[o], h_p[o]
model = NIL(user_dim=U.shape[1], poi_dim=P.shape[1], h=cfg["hidden"], drop=(cfg["drop"], cfg["drop"] + 0.1, cfg["drop"]), geo_dim=cfg.get("geo_dim", 5)).to(dev)
model.load_state_dict(torch.load(os.path.join(run_dir, "best.pth"), map_location=dev))
model.eval()
with torch.no_grad():
    PT = model.poi_tower(P)
v2 = cfg.get("feat_v", 1) == 2
valid_ids = torch.as_tensor(np.nonzero(~no_clean_np)[0], device=dev)


@torch.no_grad()
def collect(users, targets, geo, csr, pop, seed):
    """-> model logits, popularity counts, distance (km) for [target, negatives...]: three (n, 1 + n_neg) arrays"""
    g = torch.Generator(device=dev); g.manual_seed(seed)
    pop_t = torch.as_tensor(pop, dtype=torch.float32, device=dev)
    L, C, D = [], [], []
    for s in range(0, len(users), a.bs):
        ub, tb = users[s:s + a.bs], targets[s:s + a.bs]
        ut = torch.as_tensor(ub, device=dev)
        seen = torch.zeros(len(ub), n_pois, dtype=torch.bool, device=dev)
        for r, u in enumerate(ub):
            seen[r, torch.as_tensor(csr[1][csr[0][u]:csr[0][u + 1]], device=dev)] = True
        seen[torch.arange(len(ub), device=dev), torch.as_tensor(tb, device=dev)] = True
        w = (~seen[:, valid_ids]).float()
        neg = valid_ids[torch.multinomial(w, a.n_neg, replacement=False, generator=g)]
        cand = torch.cat([torch.as_tensor(tb, device=dev)[:, None], neg], 1)
        sc = score_all(model, model.user_tower(U[ut]), PT, geo, ut)
        L.append(sc.gather(1, cand).cpu().numpy()); C.append(pop_t[cand].cpu().numpy())
        D.append(geo.dist(ut, cand).min(-1).values.cpu().numpy())
    return np.concatenate(L), np.concatenate(C), np.concatenate(D)


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def mae(p):
    y = np.zeros_like(p); y[:, 0] = 1.0
    return float(np.abs(y - p).mean())


def norm(w):
    return w / w.sum(1, keepdims=True)


# ---- temperature on validation users (history = train) ----
ok = ~no_clean_np[va_p]
sel = np.random.RandomState(0).permutation(int(ok.sum()))[:a.val_users]
geo_tr = GeoCtx(d["poi_lat"], d["poi_lon"], tr_u, tr_p, n_users, dev, M=cfg["geo_m"], pop=np.bincount(tr_p, minlength=n_pois), v2=v2)
Lv, _, _ = collect(va_u[ok][sel], va_p[ok][sel], geo_tr, build_csr(tr_u, tr_p, n_users), np.bincount(tr_p, minlength=n_pois), seed=1)
nll = {T: float(-np.log(softmax(Lv / T)[:, 0] + 1e-12).mean()) for T in TEMPS}
T = min(nll, key=nll.get)
print(f"temperature tuned on validation (log-loss): {T}   {nll}", flush=True)

# ---- test ----
ok = ~no_clean_np[te_p]
pop_h = np.bincount(h_p, minlength=n_pois)
geo_h = GeoCtx(d["poi_lat"], d["poi_lon"], h_u, h_p, n_users, dev, M=cfg["geo_m"], pop=pop_h, v2=v2)
Lt, Ct, Dt = collect(te_u[ok], te_p[ok], geo_h, build_csr(h_u, h_p, n_users), pop_h, seed=2)
k = Lt.shape[1]
probs = {"AuraPath": softmax(Lt / T), "Popularity": norm(Ct + 1.0),
         "Popular within 25 km": norm(np.where(Dt <= 25.0, Ct + 1.0, 1e-3)), "Uniform": np.full_like(Lt, 1.0 / k)}
res = {"n_users": int(len(Lt)), "candidates_per_user": int(k), "temperature": T, "methods": {}}
print(f"\n{len(Lt):,} test users, {k} candidates each")
print(f"{'method':24s}{'MAE':>9s}{'p(target)':>11s}{'top-1 acc':>11s}")
for n, p in probs.items():
    res["methods"][n] = {"mae": mae(p), "mean_p_target": float(p[:, 0].mean()), "top1": float((p[:, 0] > p[:, 1:].max(1)).mean())}
    print(f"{n:24s}{mae(p):9.5f}{p[:, 0].mean():11.3f}{(p[:, 0] > p[:, 1:].max(1)).mean():11.3f}")
for base in ("Popularity", "Popular within 25 km", "Uniform"):
    red = 1.0 - res["methods"]["AuraPath"]["mae"] / res["methods"][base]["mae"]
    res[f"mae_reduction_vs_{base.lower().replace(' ', '_')}"] = red
    print(f"MAE reduction against {base}: {red:.1%}")
json.dump(res, open(os.path.join(run_dir, "mae_results.json"), "w"), indent=1)
print("saved", os.path.join(run_dir, "mae_results.json"))
