"""
pack_resnet_float16.py -- build the compact copy of the ResNet features for the GPU server.

Reads brain/embeddings/embeddings_chunk_*.npy (float32, 53 GB) and writes only the photos that
belong to a place, as float16, in parts of ~1 GB. Row order is identical to
brain/quintuplet_embeddings.npy (chunk order, photos without a place skipped), so the existing
*_poi_ids.npy and the id replay in training/00b stay valid. The originals are only read.

  python scripts/pack_resnet_float16.py --brain brain --out staging/resnet_f16
Needs numpy only. Safe to re-run: finished parts are skipped.
"""
import argparse
import json
import os
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--brain", default="brain")
ap.add_argument("--out", default="staging/resnet_f16")
ap.add_argument("--rows-per-part", type=int, default=250000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
t0 = time.time()

m = np.loadtxt(os.path.join(a.brain, "photo_poi_mapping.csv"), delimiter=",", skiprows=1, usecols=(0,), dtype=np.int64)
mapped = np.unique(m)
saved_poi = np.load(os.path.join(a.brain, "quintuplet_embeddings_poi_ids.npy"))
n_chunks = len([f for f in os.listdir(os.path.join(a.brain, "ids")) if f.startswith("ids_chunk_")])
print(f"mapped photos: {len(mapped):,}  chunks: {n_chunks}  expected rows: {len(saved_poi):,}", flush=True)

buf, buf_rows, part, total, all_ids = [], 0, 0, 0, []


def flush():
    global buf, buf_rows, part
    if not buf_rows:
        return
    path = os.path.join(a.out, f"part_{part:03d}.npy")
    if not os.path.exists(path):
        np.save(path + ".tmp.npy", np.concatenate(buf))
        os.replace(path + ".tmp.npy", path)
    buf, buf_rows, part = [], 0, part + 1


for i in range(n_chunks):
    ids = np.load(os.path.join(a.brain, "ids", f"ids_chunk_{i}.npy"), allow_pickle=True).astype(np.int64)
    keep = np.isin(ids, mapped)
    all_ids.append(ids[keep])
    n = int(keep.sum())
    if n:
        done = os.path.exists(os.path.join(a.out, f"part_{part:03d}.npy")) and buf_rows == 0
        emb = None
        if not done:
            emb = np.load(os.path.join(a.brain, "embeddings", f"embeddings_chunk_{i}.npy"), mmap_mode="r")
            assert emb.shape[0] == len(ids), (i, emb.shape, len(ids))
        pos = np.nonzero(keep)[0]
        s = 0
        while s < n:                                   # fill parts exactly to rows-per-part
            take = min(a.rows_per_part - buf_rows, n - s)
            exists = os.path.exists(os.path.join(a.out, f"part_{part:03d}.npy"))
            if exists:
                buf.append(np.empty((0, 2048), np.float16))
            else:
                if emb is None:
                    emb = np.load(os.path.join(a.brain, "embeddings", f"embeddings_chunk_{i}.npy"), mmap_mode="r")
                buf.append(np.asarray(emb[pos[s:s + take]]).astype(np.float16))
            buf_rows += take
            s += take
            if buf_rows == a.rows_per_part:
                flush()
        total += n
    if i % 50 == 0:
        print(f"chunk {i}/{n_chunks}  rows {total:,}  parts {part}  {time.time() - t0:.0f}s", flush=True)
flush()

all_ids = np.concatenate(all_ids)
assert total == len(saved_poi) == len(all_ids), (total, len(saved_poi))
np.save(os.path.join(a.out, "photo_ids.npy"), all_ids)
json.dump({"rows": int(total), "dim": 2048, "dtype": "float16", "parts": part, "rows_per_part": a.rows_per_part,
           "order": "same as brain/quintuplet_embeddings.npy"}, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
size = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out)) / 1e9
print(f"done: {total:,} rows in {part} parts, {size:.1f} GB, {time.time() - t0:.0f}s")
