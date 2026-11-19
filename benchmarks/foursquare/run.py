"""
Train and evaluate the AuraPath v2 ranker on Foursquare NYC (GETNext splits and protocol).

  python -m benchmarks.foursquare.run --seed 0                       # full model
  python -m benchmarks.foursquare.run --seed 0 --variant no_cat      # ablation: no category embedding
  python -m benchmarks.foursquare.run --seed 0 --variant no_dist     # ablation: no distance features
  python -m benchmarks.foursquare.run --seed 0 --cat_text            # + frozen text embedding of category names
  python -m benchmarks.foursquare.run --seed 0 --smoke               # 1-minute sanity run

Training uses the training file only; the epoch is chosen on validation (GETNext's 4*Acc@1 + Acc@20);
the test file is scored once, with the chosen weights. Outputs: runs/foursquare_nyc/<variant>_seed<N>.json and
runs/foursquare_nyc/baselines.json.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from . import baselines as B
from .data import HERE, Index, TrainStream, eval_refs, eval_samples, load_splits, user_cat_counts
from .metrics import monitor, ranks_of, summarize
from .model import CatRanker, GeoNYC, cat_text_embeddings

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
ap.add_argument("--variant", choices=["full", "no_cat", "no_dist"], default="full")
ap.add_argument("--cat_text", action="store_true", help="add a frozen sentence-transformer embedding of the category name")
ap.add_argument("--out", default=os.path.join("runs", "foursquare_nyc"))
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE", help="override config values (tuning)")
ap.add_argument("--val_only", action="store_true", help="tuning run: never score the test file")
ap.add_argument("--tag", default="", help="suffix for the output file name")
ap.add_argument("--baselines_only", action="store_true", help="score the non-learned baselines and exit")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
for kv in a.set:
    k, v = kv.split("=", 1)
    sec, key = k.split(".")
    assert key in cfg[sec], f"unknown config key {k}"
    cfg[sec][key] = yaml.safe_load(v)
mc, tc = cfg["model"], cfg["train"]

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
try:
    torch.use_deterministic_algorithms(True, warn_only=True)     # torch >= 1.11
except TypeError:
    pass
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(a.out, exist_ok=True)
name = a.variant + ("_cattext" if a.cat_text else "") + ("_smoke" if a.smoke else "") + f"_seed{a.seed}" + \
    (f"_{a.tag}" if a.tag else "")

# ---------------- data (features from the training file only) ----------------
D = load_splits()
ix = Index(D["train"])
st = TrainStream(D["train"], ix)
M = mc["geo_m"]
S = {s: eval_samples(D[s], ix, cfg["data"]["short_traj_thres"]) for s in ("val", "test")}
R = {s: eval_refs(S[s], st, ix.n_users, M) for s in S}
for s in S:                                                    # cheap leakage guards (full test: test_leakage.py)
    assert st.t.max() < D[s]["t"].min(), "training check-ins must all precede validation / test"
    ref, ref_t = R[s]
    assert ((ref_t < S[s]["target_t"][:, None]) | (ref < 0)).all(), "a reference check-in is not before its target"
geo = GeoNYC(ix, st, dev, M, mc["near_km"])
C = torch.as_tensor(user_cat_counts(st, ix), device=dev)
poi_cat = torch.as_tensor(ix.poi_cat, device=dev)
print(f"device={dev} pois={ix.n_pois:,} users={ix.n_users:,} cats={ix.n_cats} train positives={len(st.positives):,} "
      f"val={len(S['val']['u']):,} test={len(S['test']['u']):,}", flush=True)


def evaluate(scorer, split, bs=tc["eval_batch"]):
    ref = torch.as_tensor(R[split][0], device=dev)
    u = torch.as_tensor(S[split]["u"], device=dev)
    tg = torch.as_tensor(S[split]["target"], device=dev)
    n = len(u)
    ranks = [ranks_of(scorer(u[i:i + bs], ref[i:i + bs], (split, i, min(i + bs, n))).float(), tg[i:i + bs])
             for i in range(0, n, bs)]
    return summarize(np.concatenate(ranks))


# ---------------- baselines (deterministic; same samples) ----------------
splits = ("val",) if a.val_only else ("val", "test")
base = {}
for nm, f in [("popularity", B.popularity), ("nearest", B.nearest), ("markov", B.Markov(st.pairs, ix.n_pois))]:
    base[nm] = {s: evaluate(lambda u, r, ctx, f=f: f(geo, r, ix.n_pois), s) for s in splits}
uf = B.UserFrequency(st, ix.n_users, ix.n_pois, S)
base["user_frequency"] = {s: evaluate(lambda u, r, ctx: uf(geo, r, ix.n_pois, ctx), s) for s in splits}
best_r = max(cfg["baselines"]["radius_km"],
             key=lambda r: monitor(evaluate(lambda u, ref, ctx: B.popular_within(r)(geo, ref, ix.n_pois), "val")))
f = B.popular_within(best_r)
base["popular_within"] = {"radius_km": best_r, **{s: evaluate(lambda u, r, ctx: f(geo, r, ix.n_pois), s) for s in splits}}
for nm, v in base.items():
    print(f"baseline {nm:15s} " + " | ".join(f"{s} Acc@1 {v[s]['Acc@1']:.4f} Acc@10 {v[s]['Acc@10']:.4f} MRR {v[s]['MRR']:.4f}"
                                             for s in splits), flush=True)
if not (a.smoke or a.val_only):
    json.dump(base, open(os.path.join(a.out, "baselines.json"), "w"), indent=1)
if a.baselines_only:
    raise SystemExit(0)

# ---------------- model ----------------
text = cat_text_embeddings(ix.cat_names, mc["cat_text_model"], os.path.join(HERE, "external")) if a.cat_text else None
model = CatRanker(ix.n_cats, mc["cat_dim"], mc["hidden"], mc["drop"], geo.n_feat, text=text,
                  use_cat=a.variant != "no_cat").to(dev)
no_dist = a.variant == "no_dist"
opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["wd"])
epochs = 2 if a.smoke else tc["epochs"]
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)


class GroupSampler:                                            # as in training/01_train.py
    def __init__(self, keys):
        self.keys = torch.as_tensor(keys, device=dev).long()
        self.order = torch.argsort(self.keys)
        self.sizes = torch.bincount(self.keys)
        self.starts = torch.cumsum(self.sizes, 0) - self.sizes

    def sample(self, pos, n):
        c = self.keys[pos]
        off = torch.min((torch.rand(len(pos), n, device=dev) * self.sizes[c][:, None]).long(), (self.sizes[c] - 1)[:, None])
        return self.order[self.starts[c][:, None] + off]


cell = np.floor((ix.lat + 90) / tc["cell_deg"]).astype(np.int64) * 100000 + np.floor((ix.lon + 180) / tc["cell_deg"]).astype(np.int64)
geo_s, cat_s = GroupSampler(np.unique(cell, return_inverse=True)[1]), GroupSampler(ix.poi_cat)
pop_w = geo.cnt ** 0.75 + 1e-9
n_pop, n_cat, n_geo = int(tc["neg"] * tc["frac_pop"]), int(tc["neg"] * tc["frac_cat"]), int(tc["neg"] * tc["frac_geo"])
n_uni = tc["neg"] - n_pop - n_cat - n_geo


def sample_negs(p):
    b = len(p)
    return torch.cat([torch.randint(0, ix.n_pois, (b, n_uni), device=dev),
                      torch.multinomial(pop_w, b * n_pop, replacement=True).view(b, n_pop),
                      cat_s.sample(p, n_cat), geo_s.sample(p, n_geo)], dim=1)


@torch.no_grad()
def model_scorer(u, ref, ctx=None):
    model.eval()
    cand = torch.arange(ix.n_pois, device=dev)[None].expand(len(u), -1)
    return model.score(C[u], poi_cat[cand], geo.feats_ref(ref, cand, no_dist=no_dist))


rows_all = torch.as_tensor(st.positives, device=dev)
if a.smoke:
    rows_all = rows_all[torch.randperm(len(rows_all), device=dev)[:5000]]
U_t, P_t = torch.as_tensor(st.u, device=dev), torch.as_tensor(st.p, device=dev)
hist, best, bad, best_state = [], -1.0, 0, None
for ep in range(1, epochs + 1):
    t0 = time.time()
    model.train()
    perm = rows_all[torch.randperm(len(rows_all), device=dev)]
    tot, nb = 0.0, 0
    for s in range(0, len(perm), tc["batch"]):
        rows = perm[s:s + tc["batch"]]
        u, p = U_t[rows], P_t[rows]
        cand = torch.cat([p[:, None], sample_negs(p)], dim=1)
        g = geo.feats_ref(geo.hist[rows], cand, pos=p, no_dist=no_dist)          # only check-ins before the positive
        counts = C[u] - F.one_hot(poi_cat[p], ix.n_cats).float()                 # user taste without this check-in
        logits = model.score(counts, poi_cat[cand], g)
        dup = torch.cat([torch.zeros_like(cand[:, :1], dtype=torch.bool), cand[:, 1:] == p[:, None]], dim=1)
        loss = F.cross_entropy(logits.masked_fill(dup, float("-inf")), torch.zeros(len(u), dtype=torch.long, device=dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item(); nb += 1
    sched.step()
    m = evaluate(model_scorer, "val")
    sc = monitor(m)
    hist.append({"epoch": ep, "loss": tot / nb, "val": m, "sec": round(time.time() - t0, 1)})
    flag = ""
    if sc > best:
        best, bad, flag = sc, 0, " *best*"
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
    print(f"ep {ep:02d} loss {tot / nb:.4f} | val Acc@1 {m['Acc@1']:.4f} Acc@5 {m['Acc@5']:.4f} Acc@10 {m['Acc@10']:.4f} "
          f"Acc@20 {m['Acc@20']:.4f} MRR {m['MRR']:.4f} | {time.time() - t0:.0f}s{flag}", flush=True)
    if bad >= tc["patience"]:
        print("early stop", flush=True)
        break

model.load_state_dict(best_state)
val_m = evaluate(model_scorer, "val")
test_m = None if a.val_only else evaluate(model_scorer, "test")
print("best val :", {k: round(v, 4) for k, v in val_m.items()}, f"monitor {monitor(val_m):.4f}")
if test_m:
    print("TEST     :", {k: round(v, 4) for k, v in test_m.items()}, flush=True)
json.dump({"variant": a.variant, "cat_text": a.cat_text, "seed": a.seed, "smoke": a.smoke, "config": cfg,
           "best_epoch": max(hist, key=lambda h: monitor(h["val"]))["epoch"], "val": val_m, "test": test_m,
           "history": hist}, open(os.path.join(a.out, name + ".json"), "w"), indent=1)
print("saved", os.path.join(a.out, name + ".json"))
