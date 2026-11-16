"""
01b_train_attn.py -- 01_train.py with TRAINED attention pooling in place of mean pooling (ROADMAP Phase 4).

A user and a place are each a SET of photo embeddings. A small scorer gives every photo a weight from the photo
itself, the set's mean, and the photo's offset from that mean (so typical and outlying photos can be told apart);
the embedding is the weighted sum. It is trained end to end with the ranking loss and starts as plain mean
pooling (the scorer's last layer is zero), so any gain over 01_train.py comes from the learned weights.
No photo is discarded and no keyword list is used (this replaces notebook 07).

Leakage rules are those of 01_train.py: the positive place is pooled WITHOUT the user's own photos, and the user
is pooled without the photos they took at the positive place. Held-out visits' photos are never in any set.

  CUDA_VISIBLE_DEVICES=0 python training/01b_train_attn.py --data data_v4 --name p4_attn --feat-v 1
Outputs: runs/<name>/{best.pth (ranking model), pool.pth, pooled.npz (final user/place embeddings), config.json, history.json}
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NIL, GeoCtx, score_all, build_csr, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_v4")
ap.add_argument("--emb", default="brain/quintuplet_embeddings.npy", help="photo embeddings the data was prepared from")
ap.add_argument("--name", required=True)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--neg", type=int, default=63)
ap.add_argument("--frac-pop", type=float, default=0.15)
ap.add_argument("--frac-cluster", type=float, default=0.10)
ap.add_argument("--frac-geo", type=float, default=0.50)
ap.add_argument("--cell-deg", type=float, default=0.5)
ap.add_argument("--geo-m", type=int, default=30)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--drop", type=float, default=0.2)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--patience", type=int, default=6)
ap.add_argument("--val-users", type=int, default=2000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--feat-v", type=int, default=1)
ap.add_argument("--set-user", type=int, default=16, help="photos sampled per user per step")
ap.add_argument("--set-pos", type=int, default=16, help="photos sampled for the positive place")
ap.add_argument("--set-neg", type=int, default=6, help="photos sampled per negative place")
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_dir = os.path.join("runs", a.name)
os.makedirs(out_dir, exist_ok=True)
amp = dev.type == "cuda"

d = np.load(os.path.join(a.data, "prepared.npz"), allow_pickle=True)
ph = np.load(os.path.join(a.data, "photos.npz"))
n_users, n_pois = len(d["user_ids"]), len(d["poi_ids"])
MD = int(d["mem_dim"]) if "mem_dim" in d.files else 6
U = torch.as_tensor(d["user_emb"], device=dev)
P = torch.as_tensor(d["poi_emb"], device=dev)
VD = P.shape[1] - MD
poi_vis_sum = torch.as_tensor(d["poi_vis_sum"], device=dev)
poi_cnt = torch.as_tensor(d["poi_cnt"], device=dev).float()
train_vsum = torch.as_tensor(d["train_vsum"], device=dev)
train_vcnt = torch.as_tensor(d["train_vcnt"], device=dev).float()
g_content = torch.as_tensor(d["g_content"], device=dev)
tr_u, tr_p = d["train_u"].astype(np.int64), d["train_p"].astype(np.int64)
va_u, va_p = d["val_u"].astype(np.int64), d["val_p"].astype(np.int64)
poi_lat, poi_lon = d["poi_lat"], d["poi_lon"]
geo = GeoCtx(poi_lat, poi_lon, tr_u, tr_p, n_users, dev, M=a.geo_m, pop=np.bincount(tr_p, minlength=n_pois), v2=a.feat_v == 2)
no_clean_np = d["poi_no_clean"].astype(bool)
no_clean = torch.as_tensor(no_clean_np, device=dev)
n_after = d["poi_cnt"][tr_p] - d["train_vcnt"]
eligible = np.nonzero(geo.has_hist & (n_after > 0))[0]
ok = ~no_clean_np[va_p]
va_u, va_p = va_u[ok], va_p[ok]
if a.smoke:
    a.epochs, a.val_users = 2, 200
    eligible = np.random.permutation(eligible)[:20000]
tr_p_fit = tr_p[eligible]

# ---------- photo sets ----------
print("loading photo embeddings ...", flush=True)
V = torch.as_tensor(np.load(a.emb, mmap_mode="r")[ph["row"]].astype(np.float16), device=dev)      # (N, VD) half
assert V.shape[1] == VD
ph_p = torch.as_tensor(ph["p"].astype(np.int64), device=dev)
ph_u = torch.as_tensor(ph["u"].astype(np.int64), device=dev)


def csr_of(keys, mask, n):
    idx = np.nonzero(mask)[0]
    o = np.argsort(keys[idx], kind="stable")
    cnt = np.bincount(keys[idx], minlength=n)
    return (torch.as_tensor(np.concatenate([[0], np.cumsum(cnt)])[:-1], device=dev),
            torch.as_tensor(cnt, device=dev), torch.as_tensor(idx[o], device=dev))


pl_ptr, pl_cnt, pl_idx = csr_of(ph["p"].astype(np.int64), ph["clean"], n_pois)          # place -> its clean photos
us_ptr, us_cnt, us_idx = csr_of(ph["u"].astype(np.int64), ph["train"], n_users)         # user  -> their train photos
print(f"device={dev} users={n_users:,} pois={n_pois:,} photos={len(ph_p):,} train_positives={len(eligible):,} val={len(va_u):,}", flush=True)


class SetPool(nn.Module):
    def __init__(self, dim, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * dim, h), nn.GELU(), nn.Linear(h, 1))
        nn.init.zeros_(self.net[2].weight); nn.init.zeros_(self.net[2].bias)          # start as mean pooling

    def logit(self, e, ctx):
        """e: (..., D) photos, ctx: (..., D) their set's mean -> (...,) bounded attention logit"""
        return self.net(torch.cat([e, ctx, e - ctx], dim=-1)).squeeze(-1).clamp(-10.0, 10.0)

    def forward(self, e, ctx, mask):
        """e: (B, S, D) sampled photos, ctx: (B, D) exact mean of the whole set, mask: (B, S) -> (B, D)"""
        e, ctx, mk = e.float(), ctx.float(), mask.float()
        w = torch.exp(self.logit(e, ctx[:, None, :].expand_as(e)).float()) * mk
        den, nv = w.sum(1, keepdim=True), mk.sum(1, keepdim=True)
        # the set is only a SAMPLE: move the exact mean (ctx) by how far the weights depart from uniform.
        # With uniform weights this is exactly ctx, i.e. plain mean pooling, and it matches pool_all() in expectation.
        dev_w = w / den.clamp(min=1e-9) - mk / nv.clamp(min=1.0)
        return ctx + (dev_w[..., None] * e).sum(1)


def sample(ptr, cnt, idx, keys, n):
    """n photo indices (with replacement) for each key -> (len(keys), n) photo ids, valid (len(keys), 1)"""
    c = cnt[keys]
    off = (torch.rand(len(keys), n, device=dev) * c[:, None].float()).long().clamp(max=10 ** 9)
    off = torch.min(off, (c - 1).clamp(min=0)[:, None])
    return idx[(ptr[keys][:, None] + off).clamp(max=len(idx) - 1)], (c > 0)[:, None]


@torch.no_grad()
def pool_all(pool, ptr, cnt, idx, ctx, chunk=400000):
    """exact pooling over ALL photos of every set (no sampling) -> (n_sets, VD)"""
    n = len(cnt)
    owner = torch.repeat_interleave(torch.arange(n, device=dev), cnt)
    num = torch.zeros(n, VD, device=dev); den = torch.zeros(n, device=dev)
    for s in range(0, len(idx), chunk):
        e = V[idx[s:s + chunk]].float()
        o = owner[s:s + chunk]
        w = torch.exp(pool.logit(e, ctx[o]))
        num.index_add_(0, o, w[:, None] * e); den.index_add_(0, o, w)
    return torch.where(den[:, None] > 0, num / den[:, None].clamp(min=1e-9), ctx)


# ---------- negative samplers (as in 01_train.py) ----------
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
pop_w = torch.as_tensor(np.bincount(tr_p_fit, minlength=n_pois).astype(np.float64) ** 0.75 + 1e-9, device=dev, dtype=torch.float)
n_pop, n_clu, n_geo = int(a.neg * a.frac_pop), int(a.neg * a.frac_cluster), int(a.neg * a.frac_geo)
n_uni = a.neg - n_pop - n_clu - n_geo


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


model = NIL(user_dim=VD + MD, poi_dim=VD + MD, h=a.hidden, drop=(a.drop, a.drop + 0.1, a.drop), geo_dim=geo.n_feat).to(dev)
pool_u, pool_p = SetPool(VD).to(dev), SetPool(VD).to(dev)
params = list(model.parameters()) + list(pool_u.parameters()) + list(pool_p.parameters())
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)
scaler = torch.cuda.amp.GradScaler(enabled=amp)
print("params: ranking", sum(p.numel() for p in model.parameters()), "pooling", sum(p.numel() for p in pool_u.parameters()) * 2, flush=True)

rng = np.random.RandomState(0)
vsel = rng.permutation(len(va_u))[:a.val_users]
v_users, v_targets = va_u[vsel], va_p[vsel]
train_csr = build_csr(tr_u, tr_p, n_users)
U_vis, P_vis, U_mem, P_mem = U[:, :VD], P[:, :VD], U[:, VD:], P[:, VD:]


@torch.no_grad()
def pooled_embeddings():
    pool_u.eval(); pool_p.eval()
    up = pool_all(pool_u, us_ptr, us_cnt, us_idx, U_vis)
    pp = pool_all(pool_p, pl_ptr, pl_cnt, pl_idx, P_vis)
    return torch.cat([up, U_mem], 1), torch.cat([pp, P_mem], 1)


def validate():
    model.eval()
    Ue, Pe = pooled_embeddings()
    PT = model.poi_tower(Pe)
    sc = lambda ub: score_all(model, model.user_tower(Ue[ub]), PT, geo, ub)
    return evaluate(sc, v_users, v_targets, [train_csr], dev, bs=8, invalid=no_clean)[0], Ue, Pe


m0 = validate()[0]
print(f"before training (random ranking model, mean pooling): val NDCG@20 {m0['NDCG@20']:.4f}", flush=True)

hist, best, bad = [], -1.0, 0
row_t = torch.as_tensor(eligible, device=dev)
pos_u_all = torch.as_tensor(tr_u, device=dev)
pos_p_all = torch.as_tensor(tr_p, device=dev)
N = len(eligible)
for ep in range(1, a.epochs + 1):
    t0 = time.time()
    model.train(); pool_u.train(); pool_p.train()
    perm = torch.randperm(N, device=dev)
    tot, nb = 0.0, 0
    for s in range(0, N - a.batch + 1, a.batch):
        rows = row_t[perm[s:s + a.batch]]
        u, p = pos_u_all[rows], pos_p_all[rows]
        B = len(u)
        cand = torch.cat([p[:, None], sample_negs(p)], dim=1)
        g = geo.feats(u, cand, rows=rows, pos=p)
        neg = cand[:, 1:].reshape(-1)
        # leave-one-user-out mean of the positive place (context of its photo set); neutral if nothing is left
        n = poi_cnt[p] - train_vcnt[rows]
        ctx_pos = torch.where((n > 0)[:, None], (poi_vis_sum[p] - train_vsum[rows]) / n.clamp(min=1)[:, None], g_content[None, :VD].expand(B, -1))
        iu, vu = sample(us_ptr, us_cnt, us_idx, u, a.set_user)
        ip, vp = sample(pl_ptr, pl_cnt, pl_idx, p, a.set_pos)
        ineg, vneg = sample(pl_ptr, pl_cnt, pl_idx, neg, a.set_neg)
        with torch.cuda.amp.autocast(enabled=amp):
            eu = pool_u(V[iu].float(), U_vis[u], vu & (ph_p[iu] != p[:, None]))              # not the photos taken at the positive
            epos = pool_p(V[ip].float(), ctx_pos, vp & (ph_u[ip] != u[:, None]) & (n > 0)[:, None])   # not the user's own photos
            eneg = pool_p(V[ineg].float(), P_vis[neg], vneg.expand(-1, a.set_neg))
            ut = model.user_tower(torch.cat([eu, U_mem[u]], 1))
            pt_all = model.poi_tower(torch.cat([torch.cat([epos, P_mem[p]], 1), torch.cat([eneg, P_mem[neg]], 1)], 0))
            pt = torch.cat([pt_all[:B][:, None, :], pt_all[B:].view(B, -1, pt_all.shape[-1])], dim=1)
            logits = model.head(ut[:, None, :].expand_as(pt), pt, g).float()
        dup = (cand[:, 1:] == p[:, None]) | no_clean[cand[:, 1:]]
        logits = logits.masked_fill(torch.cat([torch.zeros_like(dup[:, :1]), dup], dim=1), float("-inf"))
        loss = F.cross_entropy(logits, torch.zeros(B, dtype=torch.long, device=dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update()
        tot += loss.item(); nb += 1
    sched.step()
    m, Ue, Pe = validate()
    score = m["NDCG@20"]
    hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val": m, "sec": round(time.time() - t0, 1)})
    flag = ""
    if score > best:
        best, bad = score, 0
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, os.path.join(out_dir, "best.pth"))
        torch.save({"pool_u": pool_u.state_dict(), "pool_p": pool_p.state_dict()}, os.path.join(out_dir, "pool.pth"))
        np.savez_compressed(os.path.join(out_dir, "pooled.npz"), user_emb=Ue.cpu().numpy().astype(np.float16), poi_emb=Pe.cpu().numpy().astype(np.float16))
        flag = " *best*"
    else:
        bad += 1
    print(f"ep {ep:02d} loss {tot / max(nb, 1):.4f} | val NDCG@20 {score:.4f} HR@20 {m['HR@20(=Recall@20)']:.4f} "
          f"HR@10 {m['HR@10(=Recall@10)']:.4f} MRR {m['MRR(=MAP)']:.4f} | {time.time() - t0:.0f}s{flag}", flush=True)
    json.dump(hist, open(os.path.join(out_dir, "history.json"), "w"), indent=1)
    if bad >= a.patience:
        print("early stop")
        break

# how uneven are the learned weights? (1.0 = every photo counts equally)
pool = torch.load(os.path.join(out_dir, "pool.pth"), map_location=dev)
pool_u.load_state_dict(pool["pool_u"]); pool_p.load_state_dict(pool["pool_p"])
with torch.no_grad():
    big = torch.nonzero(pl_cnt >= 20)[:2000, 0]
    ratios = []
    for k in big.tolist():
        e = V[pl_idx[pl_ptr[k]:pl_ptr[k] + pl_cnt[k]]].float()
        w = torch.exp(pool_p.logit(e, P_vis[k][None].expand_as(e)))
        w = w / w.sum()
        ratios.append(float(1.0 / (w ** 2).sum() / len(w)))                 # effective share of photos used
eff = float(np.mean(ratios))
print(f"effective share of a place's photos carrying the weight (places with >= 20 photos): {eff:.2f}")
json.dump({**vars(a), "geo_dim": geo.n_feat, "best_val_ndcg20": best, "pooling": "attention", "effective_photo_share": eff},
          open(os.path.join(out_dir, "config.json"), "w"), indent=1)
print("done. best val NDCG@20 =", round(best, 4))
