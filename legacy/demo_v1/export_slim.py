"""One-off: build a small deployable asset bundle (demo/assets) from brain/. Read-only on brain/.
Usage: python export_slim.py [--pois-per-cluster 4000] [--users 1500]"""
import argparse, json, os, numpy as np, pandas as pd
from urllib.parse import unquote_plus
from pth_reader import load_state_dict
from aurapath import fold_state_dict

ap = argparse.ArgumentParser(); ap.add_argument("--pois-per-cluster", type=int, default=4000)
ap.add_argument("--users", type=int, default=1500); ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
B = os.path.join(os.path.dirname(__file__), "..", "..", "brain"); O = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(O, exist_ok=True); rng = np.random.default_rng(a.seed)

np.savez_compressed(f"{O}/nil_weights.npz", **fold_state_dict(load_state_dict(f"{B}/neural_interaction_model.pth")))

poi_ids = np.load(f"{B}/poi_ids_262.npy"); poi_emb = np.load(f"{B}/poi_embeddings_262.npy", mmap_mode="r")
lab = pd.read_csv(f"{B}/poi_aesthetic_labels.csv").set_index("poi_id")
mem = lab[[f"cluster_{i}_prob" for i in range(6)]]
m = pd.read_csv(f"{B}/photo_poi_mapping.csv", usecols=["photoid", "longitude", "latitude", "poi_id"])
g = m.groupby("poi_id").agg(n=("photoid", "size"), lat=("latitude", "mean"), lon=("longitude", "mean"), photoid=("photoid", "first"))
g = g[g.index.isin(poi_ids)].join(lab["primary_cluster"])
g = g[g.n >= 5]
sel = pd.concat([x.nlargest(a.pois_per_cluster, "n") for _, x in g.groupby("primary_cluster")])
sel = sel.sort_index(); print("selected POIs", len(sel), sel.primary_cluster.value_counts().to_dict())
pos = pd.Series(np.arange(len(poi_ids)), index=poi_ids)[sel.index].values
np.save(f"{O}/poi_emb.npy", np.asarray(poi_emb[pos]).astype(np.float16))

# image url for each selected POI's representative photo (chunked scan of the big CSV)
want = set(sel.photoid.astype(int)); urls = {}
for ch in pd.read_csv(f"{B}/usa_europe_geotagged.csv", usecols=["photoid", "downloadurl"], chunksize=500000):
    ch = ch[ch.photoid.isin(want)]
    for pid, u in zip(ch.photoid, ch.downloadurl): urls[int(pid)] = unquote_plus(str(u))
meta = pd.DataFrame({"poi_id": sel.index, "lat": sel.lat.values.round(5), "lon": sel.lon.values.round(5), "n_photos": sel.n.values,
    "primary_cluster": sel.primary_cluster.values, "img": [urls.get(int(p), "") for p in sel.photoid]})
meta = meta.join(mem.loc[sel.index].reset_index(drop=True))
meta.to_csv(f"{O}/pois.csv", index=False); print("poi img urls found", (meta.img != "").mean())

uids = np.load(f"{B}/user_ids.npy", allow_pickle=True); uemb = np.load(f"{B}/user_embeddings.npy")
keep = set(rng.choice(len(uids), a.users, replace=False)); sample = np.where(uids == "90366094@N00")[0]
keep |= set(sample.tolist()); keep = np.array(sorted(keep))
np.save(f"{O}/user_emb.npy", uemb[keep].astype(np.float32)); np.save(f"{O}/user_ids.npy", uids[keep])
json.dump(json.load(open(f"{B}/cluster_label_map.json")), open(f"{O}/clusters.json", "w"))
json.dump(json.load(open(f"{B}/evaluation_results.json")), open(f"{O}/evaluation_results.json", "w"))
print("done", {f: os.path.getsize(f"{O}/{f}") // 1024 for f in os.listdir(O)}, "KB")
