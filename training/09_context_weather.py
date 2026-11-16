"""
09_context_weather.py -- do weather, temperature and crowd level help ranking? (ROADMAP Phase 3, steps 2-3)

Query context = the date/time of the held-out visit. Weather comes from brain/weather/grid.npz
(weather_noaa.py: NOAA station observations averaged on a 0.5 degree grid, daily).
Signals, all built from OTHER visits only (train for validation; train + val for test):
  * rain affinity: is the place visited relatively more on rainy (>= 1 mm) or dry days
  * temperature affinity: ... on cold (< 5 C), cool (5-15), mild (15-25) or hot (> 25) days
    (a place with few visits falls back to its aesthetic cluster's profile)
  * crowd level (the proxy for "traffic"): visits to the place in the query's time-of-day x weekend slot
At ranking time each candidate place is looked up with the weather AT THAT PLACE on the query date.
Weights are tuned on VALIDATION and applied once to TEST.

  CUDA_VISIBLE_DEVICES=0 python training/09_context_weather.py --data data_v4 --run p1_split
Outputs: runs/<run>/context_weather_results.json
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NIL, GeoCtx, score_all, build_csr, summarize, RunData

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_v4")
ap.add_argument("--run", required=True)
ap.add_argument("--weather", default="brain/weather/grid.npz")
ap.add_argument("--smooth", type=float, default=10.0)
ap.add_argument("--val-users", type=int, default=3000)
ap.add_argument("--bs", type=int, default=32)
a = ap.parse_args()
LAMS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
DAY = 86400

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_dir = os.path.join("runs", a.run)
cfg = json.load(open(os.path.join(run_dir, "config.json")))
d = RunData(a.data, run_dir)
n_users, n_pois = len(d["user_ids"]), len(d["poi_ids"])
U = torch.as_tensor(d["user_emb"], device=dev)
P = torch.as_tensor(d["poi_emb"], device=dev)
tr_u, tr_p, tr_t = d["train_u"].astype(np.int64), d["train_p"].astype(np.int64), d["train_ts"].astype(np.int64)
va_u, va_p, va_t = d["val_u"].astype(np.int64), d["val_p"].astype(np.int64), d["val_ts"].astype(np.int64)
te_u, te_p, te_t = d["test_u"].astype(np.int64), d["test_p"].astype(np.int64), d["test_ts"].astype(np.int64)
clu = d["poi_cluster"].astype(np.int64)
no_clean_np = d["poi_no_clean"].astype(bool)
no_clean = torch.as_tensor(no_clean_np, device=dev)
h_u = np.concatenate([tr_u, va_u]); h_p = np.concatenate([tr_p, va_p]); h_t = np.concatenate([tr_t, va_t])
o = np.argsort(h_u, kind="stable"); h_u, h_p, h_t = h_u[o], h_p[o], h_t[o]

# ---------- weather grid -> bucket id per (cell, day); -1 = unknown ----------
w = np.load(a.weather)
cell, lat0, lon0 = float(w["cell"]), float(w["lat0"]), float(w["lon0"])
n_lat, n_lon, n_days = w["tmean"].shape
day0 = int(np.datetime64(str(w["day0"])).astype("datetime64[D]").astype(np.int64))
tm = w["tmean"].reshape(n_lat * n_lon, n_days).astype(np.float32)
pr = w["prcp"].reshape(n_lat * n_lon, n_days).astype(np.float32)
RAIN = np.where(np.isfinite(pr), (pr >= 1.0).astype(np.int8), -1).astype(np.int8)
TEMP = np.where(np.isfinite(tm), np.digitize(tm, [5.0, 15.0, 25.0]).astype(np.int8), -1).astype(np.int8)
ci = np.clip(np.floor((d["poi_lat"].astype(np.float64) - lat0) / cell).astype(np.int64), 0, n_lat - 1)
cj = np.clip(np.floor((d["poi_lon"].astype(np.float64) - lon0) / cell).astype(np.int64), 0, n_lon - 1)
poi_cell = ci * n_lon + cj
used, poi_c = np.unique(poi_cell, return_inverse=True)            # keep only cells that contain places
RAIN_t = torch.as_tensor(RAIN[used], device=dev)                  # int8, (cells with places, days)
TEMP_t = torch.as_tensor(TEMP[used], device=dev)
poi_c_t = torch.as_tensor(poi_c, device=dev)
GROUPS = [("rain", 2, RAIN[used], RAIN_t), ("temperature", 4, TEMP[used], TEMP_t)]


def day_idx(ts):
    return np.asarray(ts) // DAY - day0


def slot(ts):                                                      # time-of-day (4) x weekend (2)
    ts = np.asarray(ts)
    return ((ts % DAY) // (6 * 3600)).astype(np.int64) * 2 + (((ts // DAY + 3) % 7) >= 5).astype(np.int64)


class Ctx:
    def __init__(self, p, t):
        dd = day_idx(t)
        inr = (dd >= 0) & (dd < n_days)
        self.cover = float(inr.mean())
        self.tabs, self.known = [], []
        for name, nb, grid, _ in GROUPS:
            b = np.full(len(p), -1, dtype=np.int64)
            b[inr] = grid[poi_c[p[inr]], dd[inr]]
            k = b >= 0
            cnt = np.zeros((n_pois, nb)); np.add.at(cnt, (p[k], b[k]), 1.0)
            glob = cnt.sum(0) / cnt.sum()
            cc = np.zeros((int(clu.max()) + 1, nb)); np.add.at(cc, clu, cnt)
            prior = ((cc + 50.0 * glob) / (cc.sum(1, keepdims=True) + 50.0))[clu]
            aff = np.log((cnt + a.smooth * prior) / (cnt.sum(1, keepdims=True) + a.smooth)) - np.log(glob)
            self.tabs.append(torch.as_tensor(aff, dtype=torch.float32, device=dev))
            self.known.append(float(k.mean()))
        crowd = np.zeros((n_pois, 8)); np.add.at(crowd, (p, slot(t)), 1.0)
        self.crowd = torch.as_tensor(np.log1p(crowd), dtype=torch.float32, device=dev)

    def affinity(self, qt, g):
        """(B, P) log-ratio of group g for the weather at each place on the query date; 0 where unknown."""
        dd = day_idx(qt)
        ok = torch.as_tensor((dd >= 0) & (dd < n_days), device=dev)
        dq = torch.as_tensor(np.clip(dd, 0, n_days - 1), device=dev)
        b = GROUPS[g][3][poi_c_t[:, None], dq[None, :]].long()     # (P, B) bucket at each place that day
        val = self.tabs[g].gather(1, b.clamp(min=0))
        return (val * (b >= 0).float() * ok[None].float()).T

    def crowd_level(self, qt):
        return self.crowd[:, torch.as_tensor(slot(qt), device=dev)].T


def z(x):
    return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp(min=1e-6)


@torch.no_grad()
def run(users, targets, qtimes, geo, ctx, csr, combos):
    """combos: {name: (lam_rain, lam_temp, lam_crowd)} -> {name: ranks}"""
    ranks = {n: [] for n in combos}
    for s in range(0, len(users), a.bs):
        ub, tb, qb = users[s:s + a.bs], targets[s:s + a.bs], qtimes[s:s + a.bs]
        ut = torch.as_tensor(ub, device=dev)
        m = z(score_all(model, model.user_tower(U[ut]), PT, geo, ut))
        parts = [ctx.affinity(qb, 0), ctx.affinity(qb, 1), z(ctx.crowd_level(qb))]
        ex_r = torch.as_tensor(np.concatenate([np.full(csr[0][u + 1] - csr[0][u], r) for r, u in enumerate(ub)]), device=dev)
        ex_c = torch.as_tensor(np.concatenate([csr[1][csr[0][u]:csr[0][u + 1]] for u in ub]), device=dev)
        tt = torch.as_tensor(tb, device=dev)
        ar = torch.arange(len(ub), device=dev)
        for name, lam in combos.items():
            sc = m.clone()
            for l, x in zip(lam, parts):
                if l:
                    sc = sc + l * x
            sc[ex_r, ex_c] = float("-inf")
            sc[:, no_clean] = float("-inf")
            tg = sc[ar, tt]
            gt = (sc > tg[:, None]).sum(1).float()
            eq = (sc == tg[:, None]).sum(1).float() - 1.0
            ranks[name].append((gt + 1.0 + eq / 2.0).cpu().numpy())
    return {n: np.concatenate(v) for n, v in ranks.items()}


model = NIL(user_dim=U.shape[1], poi_dim=P.shape[1], h=cfg["hidden"], drop=(cfg["drop"], cfg["drop"] + 0.1, cfg["drop"]), geo_dim=cfg.get("geo_dim", 5)).to(dev)
model.load_state_dict(torch.load(os.path.join(run_dir, "best.pth"), map_location=dev))
model.eval()
with torch.no_grad():
    PT = model.poi_tower(P)
v2 = cfg.get("feat_v", 1) == 2
nd = lambda r: np.where(r <= 20, 1.0 / np.log2(r + 1.0), 0.0)

print("[1/2] VALIDATION (tables from train visits)", flush=True)
ctx_tr = Ctx(tr_p, tr_t)
print(f"    train visits inside the weather period: {ctx_tr.cover:.1%}; with known rain {ctx_tr.known[0]:.1%}, temperature {ctx_tr.known[1]:.1%}", flush=True)
geo_tr = GeoCtx(d["poi_lat"], d["poi_lon"], tr_u, tr_p, n_users, dev, M=cfg["geo_m"], pop=np.bincount(tr_p, minlength=n_pois), v2=v2)
ok = ~no_clean_np[va_p]
sel = np.random.RandomState(0).permutation(int(ok.sum()))[:a.val_users]
vu, vp, vt = va_u[ok][sel], va_p[ok][sel], va_t[ok][sel]
grid = {}
for g, key in enumerate(["rain", "temp", "crowd"]):
    for l in LAMS:
        lam = [0.0, 0.0, 0.0]; lam[g] = l
        grid[f"{key}{l}"] = tuple(lam)
vr = run(vu, vp, vt, geo_tr, ctx_tr, build_csr(tr_u, tr_p, n_users), grid)
best = []
for key in ["rain", "temp", "crowd"]:
    sc = {l: float(nd(vr[f"{key}{l}"]).mean()) for l in LAMS}
    best.append(max(LAMS, key=lambda l: sc[l]))
    print(f"    {key:6s} " + "  ".join(f"{l}:{sc[l]:.4f}" for l in LAMS) + f"   -> best {best[-1]}", flush=True)

print("[2/2] TEST (tables from train + val visits)", flush=True)
ctx_h = Ctx(h_p, h_t)
geo_h = GeoCtx(d["poi_lat"], d["poi_lon"], h_u, h_p, n_users, dev, M=cfg["geo_m"], pop=np.bincount(h_p, minlength=n_pois), v2=v2)
ok = ~no_clean_np[te_p]
tu, tp, tq = te_u[ok], te_p[ok], te_t[ok]
dq = day_idx(tq)
inside = (dq >= 0) & (dq < n_days)
known_q = np.zeros(len(tq), dtype=bool)
known_q[inside] = RAIN[used][poi_c[tp[inside]], dq[inside]] >= 0
combos = {"model alone": (0.0, 0.0, 0.0), "model + rain": (best[0], 0.0, 0.0), "model + temperature": (0.0, best[1], 0.0),
          "model + rain + temperature": (best[0], best[1], 0.0), "model + crowd level": (0.0, 0.0, best[2]),
          "model + all three": tuple(best)}
tr = run(tu, tp, tq, geo_h, ctx_h, build_csr(h_u, h_p, n_users), combos)
res = {"weights": {"rain": best[0], "temperature": best[1], "crowd": best[2]}, "n_test": int(len(tu)),
       "test_queries_with_known_weather": float(known_q.mean()), "methods": {}}
print(f"    test queries with known weather at the target: {known_q.mean():.1%}")
rng = np.random.RandomState(0)
base = nd(tr["model alone"])
print(f"{'method':30s}{'HR@10':>8s}{'HR@20':>8s}{'NDCG@20':>9s}{'MRR':>8s}   NDCG@20 gain over model [95% CI]")
for name, r in tr.items():
    m = summarize(r)
    diff = nd(r) - base
    bs = np.array([diff[rng.randint(0, len(diff), len(diff))].mean() for _ in range(1000)])
    m["ndcg20_gain_over_model"] = {"mean": float(diff.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    res["methods"][name] = m
    print(f"{name:30s}{m['HR@10(=Recall@10)']:8.4f}{m['HR@20(=Recall@20)']:8.4f}{m['NDCG@20']:9.4f}{m['MRR(=MAP)']:8.4f}"
          f"   {diff.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
json.dump(res, open(os.path.join(run_dir, "context_weather_results.json"), "w"), indent=1)
print("saved", os.path.join(run_dir, "context_weather_results.json"))
