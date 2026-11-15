"""
03_destination_eval.py -- "destination mode" for the new-place cold start.

Setting (what the app does): the user says WHERE they are going (a city / ~R km area). The app ranks the
places inside that area. Score = z(popularity) + lam * z(taste), z-scored inside the area, where taste is
cosine(user visual profile, POI visual embedding) [or the trained v7 model score].
  * lam is tuned on VAL visits that are > travel-km away from the user's train history, then applied ONCE to TEST.
  * Honest disclosure: the area is centred on the true destination, i.e. we assume the user told the app where they go.
  * Only trips > travel-km away from ALL history are evaluated (the case where v7 is weak: NDCG@20 0.02).

  CUDA_VISIBLE_DEVICES=7 python training/03_destination_eval.py --data data_v3 --run v7 --radius 30
Outputs: runs/<run>/destination_results.json
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NIL, GeoCtx, score_all, build_csr, summarize

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_v3")
ap.add_argument("--run", default="v7")
ap.add_argument("--radius", type=float, default=30.0, help="km around the destination")
ap.add_argument("--travel-km", type=float, default=50.0)
ap.add_argument("--bs", type=int, default=16)
a = ap.parse_args()
LAMS = [0.25, 0.5, 1.0, 2.0, 4.0]

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_dir = os.path.join("runs", a.run)
cfg = json.load(open(os.path.join(run_dir, "config.json")))
d = np.load(os.path.join(a.data, "prepared.npz"), allow_pickle=True)
n_users, n_pois = len(d["user_ids"]), len(d["poi_ids"])
U = torch.as_tensor(d["user_emb"], device=dev)
P = torch.as_tensor(d["poi_emb"], device=dev)
VD = P.shape[1] - 6
tr_u, tr_p = d["train_u"].astype(np.int64), d["train_p"].astype(np.int64)
va_u, va_p = d["val_u"].astype(np.int64), d["val_p"].astype(np.int64)
te_u, te_p = d["test_u"].astype(np.int64), d["test_p"].astype(np.int64)
no_clean_np = d["poi_no_clean"].astype(bool)
no_clean = torch.as_tensor(no_clean_np, device=dev)
lat = torch.as_tensor(np.radians(d["poi_lat"].astype(np.float64)), dtype=torch.float32, device=dev)
lon = torch.as_tensor(np.radians(d["poi_lon"].astype(np.float64)), dtype=torch.float32, device=dev)

h_u = np.concatenate([tr_u, va_u]); h_p = np.concatenate([tr_p, va_p])
o = np.argsort(h_u, kind="stable"); h_u, h_p = h_u[o], h_p[o]
v2 = cfg.get("feat_v", 1) == 2
geo_tr = GeoCtx(d["poi_lat"], d["poi_lon"], tr_u, tr_p, n_users, dev, M=cfg["geo_m"], pop=np.bincount(tr_p, minlength=n_pois), v2=v2)
geo_h = GeoCtx(d["poi_lat"], d["poi_lon"], h_u, h_p, n_users, dev, M=cfg["geo_m"], pop=np.bincount(h_p, minlength=n_pois), v2=v2)
csr_tr, csr_h = build_csr(tr_u, tr_p, n_users), build_csr(h_u, h_p, n_users)

model = NIL(h=cfg["hidden"], drop=(cfg["drop"], cfg["drop"] + 0.1, cfg["drop"]), geo_dim=cfg.get("geo_dim", 5)).to(dev)
model.load_state_dict(torch.load(os.path.join(run_dir, "best.pth"), map_location=dev))
model.eval()
with torch.no_grad():
    PT = model.poi_tower(P)
Un = torch.nn.functional.normalize(U[:, :VD], dim=1)
Pn = torch.nn.functional.normalize(P[:, :VD], dim=1)


def hav(i, j_lat, j_lon, j_cos):
    """km from POI index i (B,) to all POIs -> (B,P)"""
    pl, pn = lat[i][:, None], lon[i][:, None]
    x = torch.sin((lat[None] - pl) / 2) ** 2 + torch.cos(lat)[None] * torch.cos(pl) * torch.sin((lon[None] - pn) / 2) ** 2
    return 2 * 6371.0 * torch.asin(torch.sqrt(x.clamp(0, 1)))


def z(x, m):
    n = m.sum(1, keepdim=True).clamp(min=1).float()
    mu = (x * m).sum(1, keepdim=True) / n
    sd = (((x - mu) ** 2) * m).sum(1, keepdim=True).div(n).sqrt().clamp(min=1e-6)
    return (x - mu) / sd


@torch.no_grad()
def far_mask(geo, users, targets):
    out = []
    for s in range(0, len(users), 1024):
        u = torch.as_tensor(users[s:s + 1024], device=dev)
        p = torch.as_tensor(targets[s:s + 1024], device=dev)[:, None]
        out.append(geo.dist(u, p).min(-1).values[:, 0].cpu().numpy())
    return np.concatenate(out) > a.travel_km


@torch.no_grad()
def run(users, targets, geo, csr, pop_cnt):
    logpop = torch.log1p(torch.as_tensor(pop_cnt, dtype=torch.float32, device=dev))[None]
    ranks, sizes = {}, []
    names = ["popularity (area)", "taste only (area)", "model only (area)", "random (area)"]
    for lam in LAMS:
        names += [f"pop+{lam}*taste", f"pop+{lam}*model"]
    for n in names:
        ranks[n] = []
    g = torch.Generator(device=dev); g.manual_seed(0)
    for s in range(0, len(users), a.bs):
        ub, tb = users[s:s + a.bs], targets[s:s + a.bs]
        u = torch.as_tensor(ub, device=dev); t = torch.as_tensor(tb, device=dev).long()
        m = hav(t, None, None, None) <= a.radius
        m &= ~no_clean[None]
        ex = torch.zeros_like(m)
        for r, uu in enumerate(ub):
            seg = csr[1][csr[0][uu]:csr[0][uu + 1]]
            ex[r, torch.as_tensor(seg, device=dev)] = True
        m &= ~ex
        m[torch.arange(len(ub), device=dev), t] = True
        mf = m.float()
        sizes += m.sum(1).cpu().tolist()
        taste = (Un[u] @ Pn.T)
        mod = score_all(model, model.user_tower(U[u]), PT, geo, u)
        zp, zt, zm = z(logpop.expand_as(taste), mf), z(taste, mf), z(mod, mf)
        cands = {"popularity (area)": zp, "taste only (area)": zt, "model only (area)": zm,
                 "random (area)": torch.rand(taste.shape, device=dev, generator=g)}
        for lam in LAMS:
            cands[f"pop+{lam}*taste"] = zp + lam * zt
            cands[f"pop+{lam}*model"] = zp + lam * zm
        for n, sc in cands.items():
            sc = sc.masked_fill(~m, float("-inf"))
            tg = sc[torch.arange(len(ub), device=dev), t]
            gt = (sc > tg[:, None]).sum(1).float()
            eq = (sc == tg[:, None]).sum(1).float() - 1.0
            ranks[n].append((gt + 1 + eq / 2).cpu().numpy())
    return {n: np.concatenate(v) for n, v in ranks.items()}, np.array(sizes)


# ----- tune on VAL, evaluate once on TEST -----
ok = ~no_clean_np[va_p]
vu, vp = va_u[ok], va_p[ok]
fm = far_mask(geo_tr, vu, vp)
vu, vp = vu[fm], vp[fm]
print(f"VAL trips > {a.travel_km:.0f} km: {len(vu):,}", flush=True)
vr, vs = run(vu, vp, geo_tr, csr_tr, np.bincount(tr_p, minlength=n_pois))
nd = lambda r: float(np.where(r <= 20, 1 / np.log2(r + 1), 0).mean())
best = {}
for fam in ["taste", "model"]:
    c = [(nd(vr[f"pop+{l}*{fam}"]), l) for l in LAMS]
    best[fam] = max(c)[1]
    print(f"VAL best lam for pop+lam*{fam}: {best[fam]}  (NDCG@20 {max(c)[0]:.4f})", flush=True)

ok = ~no_clean_np[te_p]
tu_, tp_ = te_u[ok], te_p[ok]
fm = far_mask(geo_h, tu_, tp_)
tu_, tp_ = tu_[fm], tp_[fm]
print(f"TEST trips > {a.travel_km:.0f} km: {len(tu_):,}", flush=True)
tr_, ts_ = run(tu_, tp_, geo_h, csr_h, np.bincount(h_p, minlength=n_pois))
rep = ["random (area)", "popularity (area)", "taste only (area)", "model only (area)",
       f"pop+{best['taste']}*taste", f"pop+{best['model']}*model"]
res = {"radius_km": a.radius, "n_test_trips": int(len(tu_)), "median_area_size": float(np.median(ts_)), "best_lam": best, "methods": {}}
print(f"\nmedian candidates in the destination area: {np.median(ts_):.0f}")
print(f"{'method':24s}{'HR@5':>8s}{'HR@10':>8s}{'HR@20':>8s}{'NDCG@20':>9s}{'MRR':>8s}")
for n in rep:
    m = summarize(tr_[n])
    res["methods"][n] = m
    print(f"{n:24s}{m['HR@5(=Recall@5)']:8.3f}{m['HR@10(=Recall@10)']:8.3f}{m['HR@20(=Recall@20)']:8.3f}{m['NDCG@20']:9.4f}{m['MRR(=MAP)']:8.4f}")
json.dump(res, open(os.path.join(run_dir, "destination_results.json"), "w"), indent=1)
print("saved", os.path.join(run_dir, "destination_results.json"))
