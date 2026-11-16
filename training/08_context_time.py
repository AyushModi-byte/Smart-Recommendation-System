"""
08_context_time.py -- does knowing WHEN the user is asking help? (ROADMAP Phase 3, step 1)

Query context = the timestamp of the held-out visit (the app knows the current time). No external data.
Two signals, both built from OTHER visits only (train for validation; train + val for test):
  * time affinity: for each place, how over- or under-represented the query's time-of-day, weekday/weekend,
    season and daylight buckets are among its visits (smoothed log-ratios, summed)
  * recent popularity: visits to the place in the 365 days BEFORE the query time (strictly earlier)
Each is blended with the trained model (z-scored per user); weights are tuned on VALIDATION, applied once to TEST.

  CUDA_VISIBLE_DEVICES=0 python training/08_context_time.py --data data_v4 --run p1_split
Outputs: runs/<run>/context_time_results.json
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NIL, GeoCtx, score_all, build_csr, summarize, RunData

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_v4")
ap.add_argument("--run", required=True)
ap.add_argument("--smooth", type=float, default=10.0, help="prior strength for the affinity tables")
ap.add_argument("--val-users", type=int, default=3000)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--prior", choices=["global", "cluster"], default="global",
                help="what a place with few visits falls back to: the global time profile or its aesthetic cluster's")
a = ap.parse_args()
LAMS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
DAY, SHIFT = 86400, 1 << 20

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
poi_lat = d["poi_lat"].astype(np.float64)
clu = d["poi_cluster"].astype(np.int64)
no_clean_np = d["poi_no_clean"].astype(bool)
no_clean = torch.as_tensor(no_clean_np, device=dev)
h_u = np.concatenate([tr_u, va_u]); h_p = np.concatenate([tr_p, va_p]); h_t = np.concatenate([tr_t, va_t])
o = np.argsort(h_u, kind="stable"); h_u, h_p, h_t = h_u[o], h_p[o], h_t[o]
T0 = int(min(tr_t.min(), va_t.min(), te_t.min()))
yr = lambda t: 1970 + np.asarray(t) / (365.25 * DAY)
print(f"visit years: train {yr(np.percentile(tr_t, [1, 50, 99])).round(1)}  test {yr(np.percentile(te_t, [1, 50, 99])).round(1)}", flush=True)

model = NIL(user_dim=U.shape[1], poi_dim=P.shape[1], h=cfg["hidden"], drop=(cfg["drop"], cfg["drop"] + 0.1, cfg["drop"]), geo_dim=cfg.get("geo_dim", 5)).to(dev)
model.load_state_dict(torch.load(os.path.join(run_dir, "best.pth"), map_location=dev))
model.eval()
with torch.no_grad():
    PT = model.poi_tower(P)
v2 = cfg.get("feat_v", 1) == 2
GROUPS = [("time of day", 4), ("weekend", 2), ("season", 4), ("daylight", 2)]


def buckets(ts, lat):
    """ts: seconds (camera clock time), lat: degrees -> (n, 4) int bucket ids for GROUPS."""
    t = np.asarray(ts).astype("datetime64[s]")
    hour = (np.asarray(ts) % DAY) / 3600.0
    dow = (np.asarray(ts) // DAY + 3) % 7                               # 0 = Monday
    month = t.astype("datetime64[M]").astype(np.int64) % 12             # 0 = January
    doy = (t.astype("datetime64[D]") - t.astype("datetime64[Y]")).astype(np.int64)
    decl = np.radians(-23.44) * np.cos(2 * np.pi * (doy + 10) / 365.0)
    half = np.degrees(np.arccos(np.clip(-np.tan(np.radians(lat)) * np.tan(decl), -1, 1))) / 15.0   # half day length, hours
    return np.stack([(hour // 6).astype(np.int64), (dow >= 5).astype(np.int64), ((month + 1) % 12) // 3,
                     (np.abs(hour - 12.0) < half).astype(np.int64)], axis=1)


class Ctx:
    """Affinity tables and time-sorted visit keys from one set of visits."""

    def __init__(self, p, t):
        b = buckets(t, poi_lat[p])
        self.tabs = []
        for g, (_, nb) in enumerate(GROUPS):
            cnt = np.zeros((n_pois, nb), dtype=np.float64)
            np.add.at(cnt, (p, b[:, g]), 1.0)
            glob = cnt.sum(0) / cnt.sum()
            prior = glob[None]
            if a.prior == "cluster":                                    # per-cluster profile, itself smoothed to global
                cc = np.zeros((int(clu.max()) + 1, nb)); np.add.at(cc, clu, cnt)
                prior = ((cc + 50.0 * glob) / (cc.sum(1, keepdims=True) + 50.0))[clu]
            aff = np.log((cnt + a.smooth * prior) / (cnt.sum(1, keepdims=True) + a.smooth)) - np.log(glob)
            self.tabs.append(torch.as_tensor(aff, dtype=torch.float32, device=dev))
        day = np.clip((t - T0) // DAY, 0, SHIFT - 1)
        self.keys = torch.as_tensor(np.sort(p * SHIFT + day), device=dev)
        self.base = torch.arange(n_pois, device=dev) * SHIFT
        self.tanlat = torch.as_tensor(np.tan(np.radians(poi_lat)), dtype=torch.float32, device=dev)

    def affinity(self, qt, parts=None):
        """qt: (B,) query times -> (B, P) summed log-ratio. Daylight uses each place's own latitude."""
        B = len(qt)
        out = torch.zeros(B, n_pois, device=dev)
        fixed = buckets(qt, np.zeros(B))                                # groups 0-2 do not depend on the place
        for g in range(3):
            if parts is None or g in parts:
                out += self.tabs[g][:, torch.as_tensor(fixed[:, g], device=dev)].T
        if parts is None or 3 in parts:                                 # same rule as buckets(), vectorised over places
            ts = np.asarray(qt)
            t = ts.astype("datetime64[s]")
            doy = (t.astype("datetime64[D]") - t.astype("datetime64[Y]")).astype(np.int64)
            decl = torch.as_tensor(np.radians(-23.44) * np.cos(2 * np.pi * (doy + 10) / 365.0), dtype=torch.float32, device=dev)
            hour = torch.as_tensor((ts % DAY) / 3600.0, dtype=torch.float32, device=dev)
            half = torch.rad2deg(torch.acos((-self.tanlat[None] * torch.tan(decl)[:, None]).clamp(-1, 1))) / 15.0
            dl = ((hour[:, None] - 12.0).abs() < half).long()            # (B, P)
            out += self.tabs[3].gather(1, dl.T).T
        return out

    def recent(self, qt, days=365):
        """(B, P) number of visits to each place in the `days` before the query (strictly earlier)."""
        q = torch.as_tensor(np.clip((np.asarray(qt) - T0) // DAY, 0, SHIFT - 1), device=dev)
        hi = torch.searchsorted(self.keys, (self.base[None] + q[:, None]).contiguous())
        lo = torch.searchsorted(self.keys, (self.base[None] + (q - days).clamp(min=0)[:, None]).contiguous())
        return (hi - lo).float()


def z(x):
    return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp(min=1e-6)


@torch.no_grad()
def run(users, targets, qtimes, geo, ctx, csr, combos):
    """combos: {name: (lam_time, lam_recent)} -> {name: ranks}; also the ablation of single time groups."""
    ranks = {n: [] for n in combos}
    for s in range(0, len(users), a.bs):
        ub, tb, qb = users[s:s + a.bs], targets[s:s + a.bs], qtimes[s:s + a.bs]
        ut = torch.as_tensor(ub, device=dev)
        m = z(score_all(model, model.user_tower(U[ut]), PT, geo, ut))
        aff = ctx.affinity(qb)
        rec = z(torch.log1p(ctx.recent(qb)))
        ex_r = np.concatenate([np.full(csr[0][u + 1] - csr[0][u], r) for r, u in enumerate(ub)])
        ex_c = np.concatenate([csr[1][csr[0][u]:csr[0][u + 1]] for u in ub])
        tt = torch.as_tensor(tb, device=dev)
        ar = torch.arange(len(ub), device=dev)
        for name, (l1, l2) in combos.items():
            if name.startswith("only:"):
                sc = m + l1 * ctx.affinity(qb, parts=[int(name[5:])])
            elif name == "time affinity alone":
                sc = aff.clone()
            else:
                sc = m + l1 * aff + l2 * rec
            sc[torch.as_tensor(ex_r, device=dev), torch.as_tensor(ex_c, device=dev)] = float("-inf")
            sc[:, no_clean] = float("-inf")
            tg = sc[ar, tt]
            gt = (sc > tg[:, None]).sum(1).float()
            eq = (sc == tg[:, None]).sum(1).float() - 1.0
            ranks[name].append((gt + 1.0 + eq / 2.0).cpu().numpy())
    return {n: np.concatenate(v) for n, v in ranks.items()}


nd = lambda r: np.where(r <= 20, 1.0 / np.log2(r + 1.0), 0.0)

print("[1/2] VALIDATION (tables from train visits)", flush=True)
ctx_tr = Ctx(tr_p, tr_t)
geo_tr = GeoCtx(d["poi_lat"], d["poi_lon"], tr_u, tr_p, n_users, dev, M=cfg["geo_m"], pop=np.bincount(tr_p, minlength=n_pois), v2=v2)
ok = ~no_clean_np[va_p]
sel = np.random.RandomState(0).permutation(int(ok.sum()))[:a.val_users]
vu, vp, vt = va_u[ok][sel], va_p[ok][sel], va_t[ok][sel]
grid = {f"t{l}": (l, 0.0) for l in LAMS}
grid.update({f"r{l}": (0.0, l) for l in LAMS[1:]})
vr = run(vu, vp, vt, geo_tr, ctx_tr, build_csr(tr_u, tr_p, n_users), grid)
for k in grid:
    print(f"    {k:6s} val NDCG@20 {nd(vr[k]).mean():.4f}", flush=True)
bt = max(LAMS, key=lambda l: nd(vr[f"t{l}"]).mean())
br = max(LAMS, key=lambda l: nd(vr["t0.0" if l == 0 else f"r{l}"]).mean())
print(f"    best weights: time affinity {bt}, recent popularity {br}", flush=True)

print("[2/2] TEST (tables from train + val visits)", flush=True)
ctx_h = Ctx(h_p, h_t)
geo_h = GeoCtx(d["poi_lat"], d["poi_lon"], h_u, h_p, n_users, dev, M=cfg["geo_m"], pop=np.bincount(h_p, minlength=n_pois), v2=v2)
ok = ~no_clean_np[te_p]
tu, tp, tq = te_u[ok], te_p[ok], te_t[ok]
combos = {"model alone": (0.0, 0.0), "time affinity alone": (0.0, 0.0), "model + time affinity": (bt, 0.0),
          "model + recent popularity": (0.0, br), "model + both": (bt, br)}
for g, (gname, _) in enumerate(GROUPS):
    combos[f"only:{g}"] = (bt, 0.0)
tr = run(tu, tp, tq, geo_h, ctx_h, build_csr(h_u, h_p, n_users), combos)
res = {"weights": {"time_affinity": bt, "recent_popularity": br}, "n_test": int(len(tu)),
       "val_ndcg20": {k: float(nd(vr[k]).mean()) for k in grid}, "methods": {}}
rng = np.random.RandomState(0)
base = nd(tr["model alone"])
print(f"{'method':38s}{'HR@10':>8s}{'HR@20':>8s}{'NDCG@20':>9s}{'MRR':>8s}   NDCG@20 gain over model [95% CI]")
for name, r in tr.items():
    label = name if not name.startswith("only:") else f"model + {GROUPS[int(name[5:])][0]} only"
    m = summarize(r)
    diff = nd(r) - base
    bs = np.array([diff[rng.randint(0, len(diff), len(diff))].mean() for _ in range(1000)])
    m["ndcg20_gain_over_model"] = {"mean": float(diff.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    res["methods"][label] = m
    print(f"{label:38s}{m['HR@10(=Recall@10)']:8.4f}{m['HR@20(=Recall@20)']:8.4f}{m['NDCG@20']:9.4f}{m['MRR(=MAP)']:8.4f}"
          f"   {diff.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
out_name = "context_time_results.json" if a.prior == "global" else f"context_time_{a.prior}_results.json"
res["prior"] = a.prior
json.dump(res, open(os.path.join(run_dir, out_name), "w"), indent=1)
print("saved", os.path.join(run_dir, out_name))
