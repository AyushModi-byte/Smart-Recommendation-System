"""
06_export_bundle.py -- export the frozen model + data the demo app needs (numpy only, no torch at serving time).
Also VERIFIES that the numpy re-implementation ranks exactly like the trained torch model.

  CUDA_VISIBLE_DEVICES=7 python training/06_export_bundle.py --data data_v3 --run v7 --brain brain --out bundle
Outputs bundle/{weights.npz, pois.npz, users.npz, meta.json}
"""
import argparse, json, os, shutil, sys, time
from collections import Counter
import numpy as np
import pandas as pd
import torch
from urllib.parse import unquote_plus

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "app"))
from common import NIL, GeoCtx, score_all
from aurapath_np import AuraNP, geo_feats

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_v3"); ap.add_argument("--run", default="v7")
ap.add_argument("--brain", default="brain"); ap.add_argument("--out", default="bundle")
ap.add_argument("--n-users", type=int, default=300); ap.add_argument("--tag-photos", type=int, default=20)
a = ap.parse_args()
t0 = time.time()
os.makedirs(a.out, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_dir = os.path.join("runs", a.run)
cfg = json.load(open(os.path.join(run_dir, "config.json")))
assert cfg.get("feat_v", 1) == 1, "the app implements the 5-feature (v7) model"
d = np.load(os.path.join(a.data, "prepared.npz"), allow_pickle=True)
n_users, n_pois = len(d["user_ids"]), len(d["poi_ids"])
tr_u, tr_p = d["train_u"].astype(np.int64), d["train_p"].astype(np.int64)
va_u, va_p = d["val_u"].astype(np.int64), d["val_p"].astype(np.int64)
te_u, te_p = d["test_u"].astype(np.int64), d["test_p"].astype(np.int64)
h_u = np.concatenate([tr_u, va_u]); h_p = np.concatenate([tr_p, va_p])
o = np.argsort(h_u, kind="stable"); h_u, h_p = h_u[o], h_p[o]
cnt = np.bincount(h_p, minlength=n_pois).astype(np.float64)
lat, lon = d["poi_lat"].astype(np.float32), d["poi_lon"].astype(np.float32)

print("[1/4] weights", flush=True)
sd = torch.load(os.path.join(run_dir, "best.pth"), map_location="cpu")
sd = {k: v.numpy().astype(np.float32) for k, v in sd.items() if "num_batches_tracked" not in k}
np.savez_compressed(f"{a.out}/weights.npz", **sd)

print("[2/4] POI names + thumbnails from the CSV", flush=True)
pm = pd.read_csv(f"{a.brain}/photo_poi_mapping.csv", usecols=["photoid", "poi_id"]).drop_duplicates("photoid")
poi_idx = pd.Series(np.arange(n_pois), index=d["poi_ids"])
pm = pm[pm["poi_id"].isin(poi_idx.index)]
pm["p"] = poi_idx.loc[pm["poi_id"]].values
photo_p = pd.Series(pm["p"].values, index=pm["photoid"].values)
STOP = {"geotagged", "flickr", "iphone", "square", "squareformat", "instagramapp", "iphoneography", "uploaded", "photo", "photos",
        "canon", "nikon", "sony", "usa", "europe", "2010", "2011", "2012", "2013", "2009", "2008", "2007", "unknown"}
seen = np.zeros(n_pois, dtype=np.int64)
urls = np.array([""] * n_pois, dtype=object)
tags = [Counter() for _ in range(n_pois)]
for ch in pd.read_csv(f"{a.brain}/usa_europe_geotagged.csv", usecols=["photoid", "downloadurl", "usertags"], chunksize=1_000_000,
                      dtype={"photoid": "int64", "downloadurl": "string", "usertags": "string"}):
    ch = ch[ch["photoid"].isin(photo_p.index)]
    if not len(ch):
        continue
    ch = ch.assign(p=photo_p.loc[ch["photoid"]].values)
    ch["k"] = ch.groupby("p").cumcount().values + seen[ch["p"].values]
    np.add.at(seen, ch["p"].values, 1)
    for p, k, url, tg in zip(ch["p"].values, ch["k"].values, ch["downloadurl"].values, ch["usertags"].values):
        if k >= a.tag_photos:
            continue
        if k == 0 and isinstance(url, str):
            urls[p] = unquote_plus(url)
        if isinstance(tg, str):
            for t in unquote_plus(tg).lower().split(","):
                t = t.strip()
                if len(t) >= 3 and ":" not in t and "=" not in t and t not in STOP:
                    tags[p][t] += 1
names = np.array([" · ".join(w.title() for w, _ in tags[p].most_common(2)) or f"Place {int(d['poi_ids'][p])}" for p in range(n_pois)])
np.savez_compressed(f"{a.out}/pois.npz", emb=d["poi_emb"].astype(np.float16), lat=lat, lon=lon, cnt=cnt.astype(np.float32),
                    cluster=d["poi_cluster"].astype(np.int8), no_clean=d["poi_no_clean"].astype(bool),
                    poi_ids=d["poi_ids"], name=names, url=urls.astype(str))
print(f"    named {np.mean([len(c) > 0 for c in tags]):.0%} of POIs from tags, thumbnails for {np.mean(urls != ''):.0%}", flush=True)

print("[3/4] demo users", flush=True)
rng = np.random.RandomState(7)
ok_t = ~d["poi_no_clean"].astype(bool)[te_p]
hist_len = np.bincount(h_u, minlength=n_users)
pool = np.array([i for i in range(len(te_u)) if ok_t[i] and hist_len[te_u[i]] >= 8])
pick = rng.choice(pool, min(a.n_users, len(pool)), replace=False)
users = te_u[pick]
ptr, flat = [0], []
for u in users:
    h = h_p[h_u == u]
    flat += h.tolist(); ptr.append(len(flat))
np.savez_compressed(f"{a.out}/users.npz", user_emb=d["user_emb"][users].astype(np.float16), user_idx=users,
                    user_ids=d["user_ids"][users].astype(str), hist=np.array(flat, dtype=np.int32), ptr=np.array(ptr, dtype=np.int64),
                    test_poi=te_p[pick].astype(np.int32))

print("[4/4] verifying numpy == torch on demo users (full ranking over all POIs)", flush=True)
model = NIL(h=cfg["hidden"], drop=(cfg["drop"], cfg["drop"] + 0.1, cfg["drop"]), geo_dim=5).to(dev)
model.load_state_dict(torch.load(os.path.join(run_dir, "best.pth"), map_location=dev)); model.eval()
geo = GeoCtx(lat, lon, h_u, h_p, n_users, dev, M=cfg["geo_m"], pop=cnt)
U = torch.as_tensor(d["user_emb"], device=dev); P = torch.as_tensor(d["poi_emb"], device=dev)
net = AuraNP(sd)
pt_np = net.tower(d["poi_emb"].astype(np.float16).astype(np.float32), "poi_tower")     # exactly what the app computes
pop_norm = float(np.log1p(cnt.max()))
same, maxdiff = [], []
cand = np.arange(n_pois)
for u in users[:8]:
    with torch.no_grad():
        ref = score_all(model, model.user_tower(U[[int(u)]]), model.poi_tower(P), geo, torch.tensor([int(u)], device=dev))[0].cpu().numpy()
    hist = h_p[h_u == u][::-1][:cfg["geo_m"]]
    g = geo_feats(hist, cand, lat, lon, cnt, pop_norm)
    ue = net.tower(d["user_emb"][int(u)].astype(np.float16).astype(np.float32)[None], "user_tower")
    out = net.head(ue, pt_np, g)
    same.append(len(set(np.argsort(-out)[:20]) & set(np.argsort(-ref)[:20])) / 20)
    maxdiff.append(float(np.abs(out - ref).max()))
print(f"    top-20 overlap numpy vs torch: {np.mean(same):.3f} (1.000 = identical); max |logit diff| {max(maxdiff):.4f}")
assert np.mean(same) >= 0.95, "numpy export does not reproduce the trained model"

meta = {"run": a.run, "config": {k: cfg[k] for k in ["hidden", "drop", "geo_m", "feat_v"] if k in cfg}, "pop_norm": pop_norm, "n_pois": n_pois,
        "n_users_total": n_users, "n_demo_users": int(len(users)), "numpy_vs_torch_top20_overlap": float(np.mean(same)),
        "val_baselines": cfg.get("val_baselines"), "best_val_ndcg20": cfg.get("best_val_ndcg20")}
for f in ["test_results.json", "destination_results.json", "ablation_content.json"]:
    if os.path.exists(os.path.join(run_dir, f)):
        meta[f.replace(".json", "")] = json.load(open(os.path.join(run_dir, f)))
json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
print("sizes (MB):", {f: round(os.path.getsize(f"{a.out}/{f}") / 1e6, 1) for f in os.listdir(a.out)}, f"| {time.time() - t0:.0f}s")
