"""
00b_prepare_all_photos.py -- like 00_prepare_data.py, but uses ALL photos of the clustered POIs
(3.7M metric-learned embeddings) instead of the 767k that survive NB7's keyword/untrained-attention filter.
Writes the same prepared.npz format (so 01_train.py / 02_evaluate.py run unchanged) + the same leakage rules:
  * time-based leave-one-out per user (last visit = test, one before = val)
  * user embedding from TRAIN photos only; POI embedding from photos that are not held-out visits
  * POIs whose only photos are held-out get neutral content and are flagged poi_no_clean
  * leave-one-user-out sums for training positives

  python training/00b_prepare_all_photos.py --brain brain --out data_v3
Needs: brain/ids/ids_chunk_*.npy, brain/quintuplet_embeddings.npy (+_poi_ids.npy), photo_poi_mapping.csv,
       poi_fuzzy_memberships.npy, poi_cluster_ids.npy, usa_europe_geotagged.csv
"""
import argparse, glob, json, os, re, time
import numpy as np
import pandas as pd
import scipy.sparse as sp


def group_sum(vis, rows, keys, n_groups, chunk=400_000):
    """sum of vis[rows[i]] into group keys[i]; chunked, float64 accumulation. Returns (n_groups, D) float64."""
    out = np.zeros((n_groups, vis.shape[1]), dtype=np.float64)
    for s in range(0, len(rows), chunk):
        r, k = rows[s:s + chunk], keys[s:s + chunk]
        v = np.asarray(vis[r]).astype(np.float64)
        S = sp.csr_matrix((np.ones(len(k), dtype=np.float64), (k, np.arange(len(k)))), shape=(n_groups, len(k)))
        out += S @ v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-pois", type=int, default=5)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--places", default="", help='"" = original places, "_v2" = files written by 00a_split_places.py')
    a = ap.parse_args()
    t0 = time.time()
    B, O = os.path.expanduser(a.brain), os.path.expanduser(a.out)
    os.makedirs(O, exist_ok=True)
    csv_path = a.csv or os.path.join(B, "usa_europe_geotagged.csv")

    print("[1/7] replaying NB6 id order to align photo ids with embedding rows ...", flush=True)
    pm = pd.read_csv(f"{B}/photo_poi_mapping.csv", usecols=["photoid", "poi_id"])
    pm = pm.drop_duplicates("photoid")
    poi_of = pd.Series(pm["poi_id"].values, index=pm["photoid"].values)
    files = sorted(glob.glob(f"{B}/ids/ids_chunk_*.npy"), key=lambda f: int(re.findall(r"(\d+)\.npy$", f)[0]))
    all_pid, all_poi = [], []
    for f in files:
        ids = np.load(f, allow_pickle=True).astype(np.int64)
        pois = poi_of.reindex(ids).fillna(-1).astype(np.int64).values
        m = pois != -1
        all_pid.append(ids[m]); all_poi.append(pois[m])
    all_pid, all_poi = np.concatenate(all_pid), np.concatenate(all_poi)
    vis = np.load(f"{B}/quintuplet_embeddings.npy", mmap_mode="r")
    saved_poi = np.load(f"{B}/quintuplet_embeddings_poi_ids.npy")
    assert len(all_pid) == vis.shape[0] == len(saved_poi), (len(all_pid), vis.shape, len(saved_poi))
    assert (all_poi == saved_poi).all(), "poi id order differs from the saved embedding order -> alignment broken"
    print(f"    aligned rows={len(all_pid):,} (poi ids match the saved file exactly)")
    if a.places:                                   # same photos and row order, new place ids
        pm2 = pd.read_csv(f"{B}/photo_poi_mapping{a.places}.csv", usecols=["photoid", "poi_id"]).drop_duplicates("photoid")
        all_poi = pd.Series(pm2["poi_id"].values, index=pm2["photoid"].values).reindex(all_pid).values
        assert not np.isnan(all_poi.astype(np.float64)).any(), "a photo has no place in the new mapping"
        all_poi = all_poi.astype(np.int64)
        print(f"    using places{a.places}: {len(np.unique(all_poi)):,} places")

    mem_all = np.load(f"{B}/poi_fuzzy_memberships{a.places}.npy").astype(np.float32)
    mem_poi_ids = np.load(f"{B}/poi_cluster_ids{a.places}.npy").astype(np.int64)
    poi_row = pd.Series(np.arange(len(mem_poi_ids)), index=mem_poi_ids)
    keep = np.isin(all_poi, mem_poi_ids)
    rows_all = np.nonzero(keep)[0]
    print(f"    photos in POIs that have a fuzzy membership: {len(rows_all):,} of {len(all_pid):,}")

    print("[2/7] reading uid / time / coordinates from CSV ...", flush=True)
    wanted = pd.Index(all_pid[rows_all])
    parts = []
    for ch in pd.read_csv(csv_path, usecols=["photoid", "uid", "datetaken", "latitude", "longitude"], chunksize=1_000_000,
                          dtype={"photoid": "int64", "uid": "string", "datetaken": "string", "latitude": "float64", "longitude": "float64"}):
        ch = ch[ch["photoid"].isin(wanted)]
        if len(ch):
            parts.append(ch)
    meta = pd.concat(parts, ignore_index=True).drop_duplicates("photoid")
    meta["ts"] = pd.to_datetime(meta["datetaken"], errors="coerce")

    print("[3/7] building photo table ...", flush=True)
    df = pd.DataFrame({"photoid": all_pid[rows_all], "poi": all_poi[rows_all], "row": rows_all})
    df = df.merge(meta[["photoid", "uid", "ts", "latitude", "longitude"]], on="photoid", how="inner")
    df = df.dropna(subset=["uid", "ts", "latitude", "longitude"]).reset_index(drop=True)
    df["poi_row"] = poi_row.loc[df["poi"]].values
    print(f"    usable photos={len(df):,}")

    print("[4/7] time-based leave-one-out split ...", flush=True)
    visit = df.groupby(["uid", "poi"], sort=False)["ts"].min().reset_index()
    visit = visit.sort_values(["uid", "ts", "poi"]).reset_index(drop=True)
    visit["n"] = visit.groupby("uid")["poi"].transform("size")
    visit["from_end"] = visit.groupby("uid").cumcount(ascending=False)
    visit = visit[visit["n"] >= a.min_pois].copy()
    visit["role"] = np.where(visit["from_end"] == 0, "test", np.where(visit["from_end"] == 1, "val", "train"))
    users = np.array(sorted(visit["uid"].unique()))
    visit["u"] = pd.Series(np.arange(len(users)), index=users).loc[visit["uid"]].values
    print(f"    modelled users={len(users):,} visits={len(visit):,} roles={visit['role'].value_counts().to_dict()}")
    df = df.merge(visit[["uid", "poi", "role", "u"]], on=["uid", "poi"], how="left")
    held = df["role"].isin(["val", "test"]).values
    train_photo = (df["role"] == "train").values
    print(f"    held-out photos={held.sum():,} train photos of modelled users={train_photo.sum():,}")

    print("[5/7] POI embeddings (clean photos only) ...", flush=True)
    poi_ids = np.array(sorted(df["poi"].unique()))
    P = len(poi_ids)
    p_idx_map = pd.Series(np.arange(P), index=poi_ids)
    p_idx = p_idx_map.loc[df["poi"]].values
    rows = df["row"].values
    clean = np.nonzero(~held)[0]
    p_sum = group_sum(vis, rows[clean], p_idx[clean], P)
    p_cnt = np.bincount(p_idx[clean], minlength=P)
    no_clean = p_cnt == 0
    p_vis_k = np.zeros_like(p_sum, dtype=np.float32)
    p_vis_k[~no_clean] = (p_sum[~no_clean] / p_cnt[~no_clean, None]).astype(np.float32)
    poi_mem = mem_all[poi_row.loc[poi_ids].values]
    g_vis = p_vis_k[~no_clean].mean(0)
    g_mem = poi_mem.mean(0)
    p_vis = np.where(no_clean[:, None], g_vis, p_vis_k)
    poi_emb = np.concatenate([p_vis, np.where(no_clean[:, None], g_mem, poi_mem)], axis=1).astype(np.float32)
    g_content = np.concatenate([g_vis, g_mem]).astype(np.float32)
    poi_vis_sum = p_sum.astype(np.float32)
    ll_sum = np.zeros((P, 2)); np.add.at(ll_sum, p_idx, df[["latitude", "longitude"]].values)
    ll = ll_sum / np.bincount(p_idx, minlength=P)[:, None]
    poi_lat, poi_lon = ll[:, 0].astype(np.float32), ll[:, 1].astype(np.float32)

    print("[6/7] user embeddings (train photos only) + leave-one-user-out sums ...", flush=True)
    tp = np.nonzero(train_photo)[0]
    tu = df["u"].values[tp].astype(np.int64)
    u_sum = group_sum(vis, rows[tp], tu, len(users))
    u_cnt = np.bincount(tu, minlength=len(users))
    assert (u_cnt > 0).all()
    C = sp.csr_matrix((np.ones(len(tp)), (tu, p_idx[tp])), shape=(len(users), P))
    u_mem = (C @ poi_mem.astype(np.float64)) / u_cnt[:, None]
    user_emb = np.concatenate([(u_sum / u_cnt[:, None]), u_mem], axis=1).astype(np.float32)
    visit["p"] = p_idx_map.loc[visit["poi"]].values
    out = {}
    for role in ["train", "val", "test"]:
        v = visit[visit["role"] == role]
        out[f"{role}_u"] = v["u"].values.astype(np.int32)
        out[f"{role}_p"] = v["p"].values.astype(np.int32)
        out[f"{role}_ts"] = v["ts"].values.astype("datetime64[s]").astype(np.int64)
    vk_photo = tu * P + p_idx[tp]
    uk, inv = np.unique(vk_photo, return_inverse=True)
    vsum = group_sum(vis, rows[tp], inv, len(uk))
    vcnt = np.bincount(inv, minlength=len(uk))
    vk = out["train_u"].astype(np.int64) * P + out["train_p"].astype(np.int64)
    pos = np.searchsorted(uk, vk)
    assert (uk[pos] == vk).all()
    out["train_vsum"] = vsum[pos].astype(np.float32)
    out["train_vcnt"] = vcnt[pos].astype(np.int32)

    print("[7/7] writing outputs ...", flush=True)
    np.savez_compressed(f"{O}/prepared.npz", poi_vis_sum=poi_vis_sum, poi_cnt=p_cnt.astype(np.int32), g_content=g_content,
                        poi_no_clean=no_clean, user_ids=users.astype(str), poi_ids=poi_ids, user_emb=user_emb, poi_emb=poi_emb,
                        poi_lat=poi_lat, poi_lon=poi_lon, poi_cluster=mem_all[poi_row.loc[poi_ids].values].argmax(1).astype(np.int8), **out)
    stats = {"users": int(len(users)), "pois": int(P), "photos_used": int(len(df)), "train_visits": int(len(out["train_u"])),
             "val_visits": int(len(out["val_u"])), "test_visits": int(len(out["test_u"])),
             "pois_without_clean_photos": int(no_clean.sum()),
             "test_targets_without_clean_photos": float(no_clean[out["test_p"]].mean()),
             "test_pois_unseen_in_train": float(1 - np.isin(out["test_p"], out["train_p"]).mean()),
             "min_pois": a.min_pois, "seconds": round(time.time() - t0, 1)}
    json.dump(stats, open(f"{O}/prepare_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
