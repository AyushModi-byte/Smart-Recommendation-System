"""
12_quintuplet_v2.py -- retrain the visual projector with the four similarity levels of the proposal
(ROADMAP Phase 5; KNOWN_ISSUES 10).

For an anchor photo (user u, place p) the embedding must order four kinds of photos:
    L1 same user, same place  <  L2 other user, same place  <  L3 same user, other place  <  L4 other user, other place
Loss = three margin terms for that order + one term against a semi-hard negative mined inside the batch
(notebook 06 defined a mining function but never called it, used one flat margin of 0.93, and only "same place" positives).
Validation is by PLACE: 5% of places are never seen in training, and the score is a retrieval rate on them,
compared with the raw ResNet features and with the old projector.

Needs brain/resnet_f16/ (scripts/pack_resnet_float16.py), photo_poi_mapping<places>.csv and the CSV for user ids.
  CUDA_VISIBLE_DEVICES=0 python training/12_quintuplet_v2.py --brain brain --places _v2
Outputs: brain/meal_projector_v3.pth, brain/quintuplet_embeddings_v3.npy (same row order as the v1 file), brain/quintuplet_v3.json
"""
import argparse, glob, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--brain", default="brain")
ap.add_argument("--places", default="_v2")
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--batch", type=int, default=2048)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--margins", type=float, nargs=3, default=[0.05, 0.10, 0.05])
ap.add_argument("--hard-margin", type=float, default=0.10)
ap.add_argument("--val-frac", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
B = a.brain
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t0 = time.time()

print("[1/5] loading ResNet features (float16) and labels", flush=True)
parts = sorted(glob.glob(f"{B}/resnet_f16/part_*.npy"))
X = np.concatenate([np.load(f) for f in parts])                        # (N, 2048) float16, in RAM
pid = np.load(f"{B}/resnet_f16/photo_ids.npy")
assert len(pid) == len(X), (len(pid), X.shape)
N = len(X)
pm = pd.read_csv(f"{B}/photo_poi_mapping{a.places}.csv", usecols=["photoid", "poi_id"]).drop_duplicates("photoid")
place = pd.Series(pm["poi_id"].values, index=pm["photoid"].values).reindex(pid).values
assert not np.isnan(place.astype(np.float64)).any()
place = np.unique(place.astype(np.int64), return_inverse=True)[1]
want, got = pd.Index(pid), []
for ch in pd.read_csv(f"{B}/usa_europe_geotagged.csv", usecols=["photoid", "uid"], chunksize=1000000, dtype={"photoid": "int64", "uid": str}):
    got.append(ch[ch["photoid"].isin(want)])
got = pd.concat(got, ignore_index=True).drop_duplicates("photoid")
uid = pd.Series(got["uid"].values, index=got["photoid"].values).reindex(pid)
user = pd.factorize(uid.fillna("?").values)[0].astype(np.int64)
n_place, n_user = int(place.max()) + 1, int(user.max()) + 1
print(f"    photos {N:,}  places {n_place:,}  users {n_user:,}  [{time.time() - t0:.0f}s]", flush=True)

# ---------- split by place ----------
rng = np.random.RandomState(a.seed)
users_per_place = pd.DataFrame({"p": place, "u": user}).drop_duplicates().groupby("p").size().reindex(range(n_place), fill_value=0).values
val_place = np.zeros(n_place, dtype=bool)
cand = np.nonzero(users_per_place >= 2)[0]
val_place[rng.choice(cand, int(a.val_frac * len(cand)), replace=False)] = True
is_val = val_place[place]
tr = np.nonzero(~is_val)[0]

# ---------- index structures over TRAIN photos (on the GPU) ----------
def csr(keys, n):
    o = np.argsort(keys, kind="stable")
    cnt = np.bincount(keys, minlength=n)
    return (torch.as_tensor(np.concatenate([[0], np.cumsum(cnt)])[:-1], device=dev), torch.as_tensor(cnt, device=dev), torch.as_tensor(tr[o], device=dev))


pl_ptr, pl_cnt, pl_idx = csr(place[tr], n_place)
us_ptr, us_cnt, us_idx = csr(user[tr], n_user)
pair = place[tr] * n_user + user[tr]
pk, pinv, pc = np.unique(pair, return_inverse=True, return_counts=True)
pr_ptr, pr_cnt, pr_idx = csr(pinv, len(pk))
place_t, user_t = torch.as_tensor(place, device=dev), torch.as_tensor(user, device=dev)
pair_of = torch.zeros(N, dtype=torch.long, device=dev)
pair_of[torch.as_tensor(tr, device=dev)] = torch.as_tensor(pinv, device=dev)
places_per_user = pd.DataFrame({"p": place[tr], "u": user[tr]}).drop_duplicates().groupby("u").size().reindex(range(n_user), fill_value=0).values
tr_users_per_place = pd.DataFrame({"p": place[tr], "u": user[tr]}).drop_duplicates().groupby("p").size().reindex(range(n_place), fill_value=0).values
elig = tr[(pc[pinv] >= 2) & (tr_users_per_place[place[tr]] >= 2) & (places_per_user[user[tr]] >= 2)]
elig_t = torch.as_tensor(elig, device=dev)
print(f"    train photos {len(tr):,}  held-out places {int(val_place.sum()):,}  anchors with all four levels {len(elig):,}", flush=True)


def draw(ptr, cnt, idx, keys, k):
    off = (torch.rand(len(keys), k, device=dev) * cnt[keys][:, None].float()).long()
    off = torch.min(off, (cnt[keys] - 1).clamp(min=0)[:, None])
    return idx[ptr[keys][:, None] + off]


def first_valid(c, ok):
    """c, ok: (B, k) -> (B,) the first candidate that satisfies ok, and whether one exists"""
    j = ok.float().argmax(1)
    return c.gather(1, j[:, None])[:, 0], ok.any(1)


def sample_batch(n):
    aidx = elig_t[torch.randint(len(elig_t), (n,), device=dev)]
    p, u = place_t[aidx], user_t[aidx]
    c1 = draw(pr_ptr, pr_cnt, pr_idx, pair_of[aidx], 6); l1, ok1 = first_valid(c1, c1 != aidx[:, None])
    c2 = draw(pl_ptr, pl_cnt, pl_idx, p, 8);             l2, ok2 = first_valid(c2, user_t[c2] != u[:, None])
    c3 = draw(us_ptr, us_cnt, us_idx, u, 8);             l3, ok3 = first_valid(c3, place_t[c3] != p[:, None])
    c4 = elig_t[torch.randint(len(elig_t), (n, 4), device=dev)]
    l4, ok4 = first_valid(c4, (place_t[c4] != p[:, None]) & (user_t[c4] != u[:, None]))
    keep = ok1 & ok2 & ok3 & ok4
    return [t[keep] for t in (aidx, l1, l2, l3, l4)]


def feats(idx):
    return torch.as_tensor(X[idx.cpu().numpy()].astype(np.float32), device=dev)


class MEALProjector(nn.Module):                                        # same architecture as notebook 06
    def __init__(self, input_dim=2048, proj_dim=256):
        super().__init__()
        self.block1 = nn.Sequential(nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.3))
        self.block2 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.2))
        self.block3 = nn.Sequential(nn.Linear(512, proj_dim), nn.BatchNorm1d(proj_dim))
        self.shortcut = nn.Linear(input_dim, proj_dim, bias=False)

    def forward(self, x):
        return F.normalize(self.block3(self.block2(self.block1(x))) + self.shortcut(x), p=2, dim=1)


# ---------- validation: retrieval on held-out places ----------
vrng = np.random.RandomState(1)
vidx = np.nonzero(is_val)[0]
vidx = np.sort(vrng.choice(vidx, min(20000, len(vidx)), replace=False))
vX = torch.as_tensor(X[vidx].astype(np.float32), device=dev)
vp, vu = torch.as_tensor(place[vidx], device=dev), torch.as_tensor(user[vidx], device=dev)
same_place = vp[:, None] == vp[None, :]
other_user = vu[:, None] != vu[None, :]
has_target = (same_place & other_user).any(1)                         # queries that have a same-place photo by someone else


@torch.no_grad()
def retrieval(E):
    """share of queries whose 10 nearest photos BY OTHER USERS include one from the same place"""
    E = F.normalize(E, dim=1)
    S = E @ E.T
    S[~other_user] = -2.0
    top = S.topk(10, dim=1).indices
    hit = same_place.gather(1, top).any(1)
    return float(hit[has_target].float().mean())


model = MEALProjector().to(dev)
print("[2/5] reference scores on held-out places (recall@10, other users only)", flush=True)
ref = {"raw_resnet_2048": retrieval(vX)}
old = MEALProjector().to(dev)
old.load_state_dict(torch.load(f"{B}/meal_projector.pth", map_location=dev)); old.eval()
with torch.no_grad():
    ref["old_projector_notebook06"] = retrieval(old(vX))
print(f"    queries {int(has_target.sum()):,}  raw ResNet {ref['raw_resnet_2048']:.4f}  old projector {ref['old_projector_notebook06']:.4f} "
      f"(the old projector was trained on these places, so its score is optimistic)", flush=True)

print("[3/5] training", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
scaler = torch.cuda.amp.GradScaler()
m1, m2, m3 = a.margins
best, best_state, hist = -1.0, None, []
run_loss, run_viol = 0.0, np.zeros(4)
for step in range(1, a.steps + 1):
    model.train()
    ia, i1, i2, i3, i4 = sample_batch(a.batch)
    n = len(ia)
    with torch.cuda.amp.autocast():
        E = model(feats(torch.cat([ia, i1, i2, i3, i4]))).float()
    ea, e1, e2, e3, e4 = E.split(n)
    d = lambda x, y: 1.0 - (x * y).sum(1)                              # cosine distance
    d1, d2, d3, d4 = d(ea, e1), d(ea, e2), d(ea, e3), d(ea, e4)
    # semi-hard negative from the batch: another anchor, other place and other user, just beyond the positive
    D = 1.0 - ea @ ea.T
    pa, ua = place_t[ia], user_t[ia]
    okn = (pa[:, None] != pa[None, :]) & (ua[:, None] != ua[None, :])
    semi = okn & (D > d2[:, None])
    dh = torch.where(semi.any(1), D.masked_fill(~semi, 9.0).min(1).values, D.masked_fill(~okn, 9.0).min(1).values)
    terms = [F.relu(d1 - d2 + m1), F.relu(d2 - d3 + m2), F.relu(d3 - d4 + m3), F.relu(d2 - dh + a.hard_margin)]
    loss = sum(t.mean() for t in terms)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); sched.step()
    run_loss += loss.item(); run_viol += np.array([float((t > 0).float().mean()) for t in terms])
    if step % 500 == 0 or step == a.steps:
        model.eval()
        with torch.no_grad():
            r = retrieval(model(vX))
        k = 500 if step % 500 == 0 else step % 500
        hist.append({"step": step, "loss": run_loss / k, "violations": (run_viol / k).round(3).tolist(), "val_recall10": r})
        flag = ""
        if r > best:
            best, best_state, flag = r, {k2: v.detach().cpu().clone() for k2, v in model.state_dict().items()}, " *best*"
        print(f"    step {step:5d} loss {run_loss / k:.4f} | violated L1<L2 {run_viol[0] / k:.2f}  L2<L3 {run_viol[1] / k:.2f}  "
              f"L3<L4 {run_viol[2] / k:.2f}  hard {run_viol[3] / k:.2f} | val recall@10 {r:.4f}{flag}  [{time.time() - t0:.0f}s]", flush=True)
        run_loss, run_viol = 0.0, np.zeros(4)

print("[4/5] projecting all photos", flush=True)
model.load_state_dict(best_state); model.eval()
torch.save(best_state, f"{B}/meal_projector_v3.pth")
out = np.lib.format.open_memmap(f"{B}/quintuplet_embeddings_v3.npy", mode="w+", dtype=np.float32, shape=(N, 256))
with torch.no_grad():
    for s in range(0, N, 16384):
        out[s:s + 16384] = model(torch.as_tensor(X[s:s + 16384].astype(np.float32), device=dev)).cpu().numpy()
out.flush(); del out

print("[5/5] summary", flush=True)
res = {"photos": int(N), "places": n_place, "users": n_user, "anchors": int(len(elig)), "held_out_places": int(val_place.sum()),
       "val_queries": int(has_target.sum()), "reference": ref, "best_val_recall10": best, "history": hist,
       "margins": a.margins, "hard_margin": a.hard_margin, "steps": a.steps, "batch": a.batch}
json.dump(res, open(f"{B}/quintuplet_v3.json", "w"), indent=1)
print(f"held-out places, recall@10: raw ResNet {ref['raw_resnet_2048']:.4f} | old projector {ref['old_projector_notebook06']:.4f} | new {best:.4f}")
print(f"saved {B}/quintuplet_embeddings_v3.npy  [{time.time() - t0:.0f}s]")
