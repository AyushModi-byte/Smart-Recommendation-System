"""
01_train.py -- train the geo-aware Neural Interaction Layer with sampled-softmax ranking loss.
Uses ONLY train interactions; selects the best epoch on VAL users. The test set is never touched here.

  CUDA_VISIBLE_DEVICES=7 python training/01_train.py --data data_v2 --name smoke --smoke    # 1-2 min sanity check
  CUDA_VISIBLE_DEVICES=7 python training/01_train.py --data data_v2 --name base             # real run
Outputs: runs/<name>/{best.pth,config.json,history.json}
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NIL, GeoCtx, score_all, build_csr, evaluate, popularity_scorer, nearest_scorer, geopop_scorer

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_v2")
ap.add_argument("--name", required=True)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=1024, help="positives per step")
ap.add_argument("--neg", type=int, default=63, help="negatives per positive")
ap.add_argument("--frac-pop", type=float, default=0.15)
ap.add_argument("--frac-cluster", type=float, default=0.10, help="negatives from the positive's aesthetic cluster")
ap.add_argument("--frac-geo", type=float, default=0.50, help="negatives from the positive's ~55km geo cell")
ap.add_argument("--cell-deg", type=float, default=0.5)
ap.add_argument("--geo-m", type=int, default=30, help="recent train visits used for distance features")
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--drop", type=float, default=0.2)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--patience", type=int, default=6)
ap.add_argument("--val-users", type=int, default=2000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--feat-v", type=int, default=2, help="1 = v7 geo features, 2 = + travel habit, area popularity, co-visit")
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()

torch.manual_seed(a.seed)
np.random.seed(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_dir = os.path.join("runs", a.name)
os.makedirs(out_dir, exist_ok=True)
amp = dev.type == "cuda"

d = np.load(os.path.join(a.data, "prepared.npz"), allow_pickle=True)
n_users, n_pois = len(d["user_ids"]), len(d["poi_ids"])
U = torch.as_tensor(d["user_emb"], device=dev)
P = torch.as_tensor(d["poi_emb"], device=dev)
VD = P.shape[1] - 6                                            # visual dims (256); last 6 = fuzzy membership
poi_vis_sum = torch.as_tensor(d["poi_vis_sum"], device=dev)
poi_cnt = torch.as_tensor(d["poi_cnt"], device=dev).float()
train_vsum = torch.as_tensor(d["train_vsum"], device=dev)
train_vcnt = torch.as_tensor(d["train_vcnt"], device=dev).float()
g_content = torch.as_tensor(d["g_content"], device=dev)
tr_u, tr_p = d["train_u"].astype(np.int64), d["train_p"].astype(np.int64)
va_u, va_p = d["val_u"].astype(np.int64), d["val_p"].astype(np.int64)
poi_lat, poi_lon = d["poi_lat"], d["poi_lon"]
geo = GeoCtx(poi_lat, poi_lon, tr_u, tr_p, n_users, dev, M=a.geo_m, pop=np.bincount(tr_p, minlength=n_pois), v2=a.feat_v == 2)   # built from FULL train set (also in smoke)
no_clean_np = d["poi_no_clean"].astype(bool)
no_clean = torch.as_tensor(no_clean_np, device=dev)             # POIs that exist only because of a held-out visit
n_after = d["poi_cnt"][tr_p] - d["train_vcnt"]                  # clean photos left after removing the user's own
eligible = np.nonzero(geo.has_hist & (n_after > 0))[0]          # positives that other photos also know about
ok = ~no_clean_np[va_p]
va_u, va_p = va_u[ok], va_p[ok]                                 # validate only on targets that existed before
if a.smoke:
    a.epochs, a.val_users = 2, 200
    eligible = np.random.permutation(eligible)[:20000]
tr_p_fit = tr_p[eligible]
print(f"device={dev} users={n_users:,} pois={n_pois:,} train_positives={len(eligible):,} val={len(va_u):,}", flush=True)


# ---------- negative samplers (all on GPU) ----------
class GroupSampler:
    def __init__(self, keys_np):
        self.keys = torch.as_tensor(keys_np, device=dev).long()
        self.order = torch.argsort(self.keys)
        self.sizes = torch.bincount(self.keys)
        self.starts = torch.cumsum(self.sizes, 0) - self.sizes

    def sample(self, pos, n):
        c = self.keys[pos]
        off = (torch.rand(len(pos), n, device=dev) * self.sizes[c][:, None]).long()
        off = torch.min(off, (self.sizes[c] - 1)[:, None])
        return self.order[self.starts[c][:, None] + off]


cell_key = (np.floor((poi_lat + 90) / a.cell_deg).astype(np.int64) * 100000 + np.floor((poi_lon + 180) / a.cell_deg).astype(np.int64))
cell_key = np.unique(cell_key, return_inverse=True)[1]
geo_sampler = GroupSampler(cell_key)
clu_sampler = GroupSampler(d["poi_cluster"].astype(np.int64))
pop = np.bincount(tr_p_fit, minlength=n_pois).astype(np.float64)
pop_w = torch.as_tensor(pop ** 0.75 + 1e-9, device=dev, dtype=torch.float)
n_pop, n_clu, n_geo = int(a.neg * a.frac_pop), int(a.neg * a.frac_cluster), int(a.neg * a.frac_geo)
n_uni = a.neg - n_pop - n_clu - n_geo
assert n_uni >= 0
print(f"negatives/positive: uniform={n_uni} popular={n_pop} cluster={n_clu} geo-cell={n_geo}", flush=True)


def sample_negs(pos):
    B = pos.shape[0]
    parts = [torch.randint(0, n_pois, (B, n_uni), device=dev)]
    if n_pop:
        parts.append(torch.multinomial(pop_w, B * n_pop, replacement=True).view(B, n_pop))
    if n_clu:
        parts.append(clu_sampler.sample(pos, n_clu))
    if n_geo:
        parts.append(geo_sampler.sample(pos, n_geo))
    return torch.cat(parts, dim=1)


# ---------- model ----------
model = NIL(h=a.hidden, drop=(a.drop, a.drop + 0.1, a.drop), geo_dim=geo.n_feat).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)
scaler = torch.cuda.amp.GradScaler(enabled=amp)
print("params:", sum(p.numel() for p in model.parameters()), flush=True)

# ---------- validation ----------
rng = np.random.RandomState(0)
vsel = rng.permutation(len(va_u))[:a.val_users]
v_users, v_targets = va_u[vsel], va_p[vsel]
train_csr = build_csr(tr_u, tr_p, n_users)


def model_scorer(ub):
    model.eval()                       # fp32 on purpose: fp16 would create score ties and distort ranks
    return score_all(model, model.user_tower(U[ub]), model.poi_tower(P), geo, ub)


def validate():
    return evaluate(model_scorer, v_users, v_targets, [train_csr], dev, bs=8, invalid=no_clean)[0]


pop_full = np.bincount(tr_p, minlength=n_pois).astype(np.float64)
base = {}
for nm, sc in [("popularity", popularity_scorer(pop_full)), ("nearest", nearest_scorer(geo, n_pois)),
               ("geo_popularity_25km", geopop_scorer(geo, pop_full, n_pois))]:
    base[nm] = evaluate(sc, v_users, v_targets, [train_csr], dev, bs=8, invalid=no_clean)[0]
    print(f"VAL baseline {nm:22s} NDCG@20 {base[nm]['NDCG@20']:.4f} HR@20 {base[nm]['HR@20(=Recall@20)']:.4f} "
          f"HR@50 {base[nm]['HR@50(=Recall@50)']:.4f} MRR {base[nm]['MRR(=MAP)']:.4f}", flush=True)

# ---------- train ----------
hist, best, bad = [], -1.0, 0
row_t = torch.as_tensor(eligible, device=dev)
pos_u_all = torch.as_tensor(tr_u, device=dev)
pos_p_all = torch.as_tensor(tr_p, device=dev)
N = len(eligible)
for ep in range(1, a.epochs + 1):
    t0 = time.time()
    model.train()
    perm = torch.randperm(N, device=dev)
    tot, nb = 0.0, 0
    for s in range(0, N - a.batch + 1, a.batch):
        rows = row_t[perm[s:s + a.batch]]
        u, p = pos_u_all[rows], pos_p_all[rows]
        cand = torch.cat([p[:, None], sample_negs(p)], dim=1)       # (B, 1+K)
        g = geo.feats(u, cand, rows=rows, pos=p)                      # history strictly before the positive
        # positive POI embedding WITHOUT the user's own photos (no self-style leak); neutral if nothing is left
        S = poi_vis_sum[p] - train_vsum[rows]
        n = poi_cnt[p] - train_vcnt[rows]
        pos_emb = torch.cat([S / n.clamp(min=1)[:, None], P[p][:, VD:]], dim=1)
        pos_emb = torch.where((n > 0)[:, None], pos_emb, g_content[None].expand_as(pos_emb))
        with torch.cuda.amp.autocast(enabled=amp):
            ut = model.user_tower(U[u])
            neg_flat = cand[:, 1:].reshape(-1)
            pt_all = model.poi_tower(torch.cat([pos_emb, P[neg_flat]], dim=0))   # one BN pass for pos+neg
            pt = torch.cat([pt_all[:len(u)][:, None, :], pt_all[len(u):].view(len(u), -1, pt_all.shape[-1])], dim=1)
            logits = model.head(ut[:, None, :].expand_as(pt), pt, g).float()
        dup = (cand[:, 1:] == p[:, None]) | no_clean[cand[:, 1:]]      # the positive itself, or a POI that did not exist yet
        logits = logits.masked_fill(torch.cat([torch.zeros_like(dup[:, :1]), dup], dim=1), float("-inf"))
        loss = F.cross_entropy(logits, torch.zeros(len(u), dtype=torch.long, device=dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        tot += loss.item()
        nb += 1
    sched.step()
    m = validate()
    score = m["NDCG@20"]
    hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val": m, "sec": round(time.time() - t0, 1)})
    flag = ""
    if score > best:
        best, bad = score, 0
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, os.path.join(out_dir, "best.pth"))
        flag = " *best*"
    else:
        bad += 1
    print(f"ep {ep:02d} loss {tot / max(nb, 1):.4f} | val NDCG@20 {score:.4f} HR@20 {m['HR@20(=Recall@20)']:.4f} "
          f"HR@10 {m['HR@10(=Recall@10)']:.4f} MRR {m['MRR(=MAP)']:.4f} | {time.time() - t0:.0f}s{flag}", flush=True)
    json.dump(hist, open(os.path.join(out_dir, "history.json"), "w"), indent=1)
    if bad >= a.patience:
        print("early stop")
        break

json.dump({**vars(a), "geo_dim": geo.n_feat, "best_val_ndcg20": best, "val_baselines": base},
          open(os.path.join(out_dir, "config.json"), "w"), indent=1)
print("done. best val NDCG@20 =", round(best, 4))
