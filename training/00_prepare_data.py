"""
00_prepare_data.py  --  leakage-free data preparation for AuraPath v2.

Split protocol (per user, by time):
  - A user "visits" a POI at the earliest timestamp of any of their photos there.
  - Visits are ordered by that time (ties broken by poi id -> deterministic).
  - last visit  -> TEST,  second to last -> VAL,  everything before -> TRAIN.
  - Only users with >= MIN_POIS distinct POIs are modelled.

Leakage rules:
  - user embedding  = mean over the user's TRAIN photos only (visual + fuzzy membership)
  - poi  embedding  = mean over photos that are NOT a held-out (val/test) visit;
                      falls back to all photos for POIs that only occur in held-out visits.

Usage:
  python training/00_prepare_data.py --brain ~/Smart-Recommendation-System/brain --out ~/Smart-Recommendation-System/data_v2
Safe to re-run (overwrites --out/prepared.npz).
"""
import argparse, json, os, time
import numpy as np
import pandas as pd


def group_mean(values, keys, n_groups):
    """Mean of rows of `values` grouped by integer `keys` in [0, n_groups). Returns (means, counts)."""
    order = np.argsort(keys, kind="stable")
    k_sorted = keys[order]
    uniq, start = np.unique(k_sorted, return_index=True)
    sums = np.add.reduceat(values[order].astype(np.float64), start, axis=0)
    counts = np.diff(np.append(start, len(k_sorted)))
    out = np.zeros((n_groups, values.shape[1]), dtype=np.float64)
    cnt = np.zeros(n_groups, dtype=np.int64)
    out[uniq] = sums / counts[:, None]
    cnt[uniq] = counts
    return out.astype(np.float32), cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-pois", type=int, default=5)
    ap.add_argument("--csv", default=None, help="path to usa_europe_geotagged.csv (default: <brain>/usa_europe_geotagged.csv)")
    a = ap.parse_args()
    t0 = time.time()
    B = os.path.expanduser(a.brain)
    O = os.path.expanduser(a.out)
    os.makedirs(O, exist_ok=True)
    csv_path = a.csv or os.path.join(B, "usa_europe_geotagged.csv")

    print("[1/6] loading filtered embeddings / ids ...", flush=True)
    f_pid = np.load(f"{B}/filtered_photo_ids.npy").astype(np.int64)
    f_poi = np.load(f"{B}/filtered_poi_ids.npy").astype(np.int64)
    f_vis = np.load(f"{B}/filtered_visual_embeddings.npy", mmap_mode="r")
    mem_all = np.load(f"{B}/poi_fuzzy_memberships.npy").astype(np.float32)
    mem_poi_ids = np.load(f"{B}/poi_cluster_ids.npy").astype(np.int64)
    assert len(f_pid) == len(f_poi) == f_vis.shape[0], (len(f_pid), len(f_poi), f_vis.shape)
    assert mem_all.shape[0] == len(mem_poi_ids)
    vdim, mdim = f_vis.shape[1], mem_all.shape[1]
    print(f"    photos={len(f_pid):,} vis_dim={vdim} mem_dim={mdim} pois_with_membership={len(mem_poi_ids):,}")

    print("[2/6] reading uid + datetaken from CSV (chunked) ...", flush=True)
    wanted = pd.Index(f_pid)
    parts = []
    for ch in pd.read_csv(csv_path, usecols=["photoid", "uid", "datetaken", "latitude", "longitude"], chunksize=1_000_000,
                          dtype={"photoid": "int64", "uid": "string", "datetaken": "string",
                                 "latitude": "float64", "longitude": "float64"}):
        ch = ch[ch["photoid"].isin(wanted)]
        if len(ch):
            parts.append(ch)
    meta = pd.concat(parts, ignore_index=True).drop_duplicates("photoid")
    meta["ts"] = pd.to_datetime(meta["datetaken"], errors="coerce")
    print(f"    matched photos={len(meta):,}  bad timestamps={meta['ts'].isna().sum():,}")

    print("[3/6] building photo table ...", flush=True)
    df = pd.DataFrame({"photoid": f_pid, "poi": f_poi, "row": np.arange(len(f_pid))})
    df = df.merge(meta[["photoid", "uid", "ts", "latitude", "longitude"]], on="photoid", how="inner")
    df = df.dropna(subset=["uid", "ts", "latitude", "longitude"])
    poi_row = pd.Series(np.arange(len(mem_poi_ids)), index=mem_poi_ids)
    df = df[df["poi"].isin(poi_row.index)].reset_index(drop=True)
    df["poi_row"] = poi_row.loc[df["poi"]].values
    print(f"    usable photos={len(df):,}")

    print("[4/6] time-based leave-one-out split ...", flush=True)
    visit = df.groupby(["uid", "poi"], sort=False)["ts"].min().reset_index()
    visit = visit.sort_values(["uid", "ts", "poi"]).reset_index(drop=True)
    visit["n"] = visit.groupby("uid")["poi"].transform("size")
    visit["from_end"] = visit.groupby("uid").cumcount(ascending=False)
    visit = visit[visit["n"] >= a.min_pois].copy()
    visit["role"] = np.where(visit["from_end"] == 0, "test", np.where(visit["from_end"] == 1, "val", "train"))
    users = np.array(sorted(visit["uid"].unique()))
    uid_to_idx = pd.Series(np.arange(len(users)), index=users)
    visit["u"] = uid_to_idx.loc[visit["uid"]].values
    print(f"    modelled users={len(users):,}  visits={len(visit):,}  roles={visit['role'].value_counts().to_dict()}")

    # photo role: held-out only if its (uid, poi) visit is val/test
    df = df.merge(visit[["uid", "poi", "role", "u"]], on=["uid", "poi"], how="left")
    held = df["role"].isin(["val", "test"]).values
    train_photo = (df["role"] == "train").values
    print(f"    held-out photos={held.sum():,}  train photos of modelled users={train_photo.sum():,}")

    print("[5/6] building embeddings (train-only) ...", flush=True)
    vis = np.asarray(f_vis[df["row"].values])  # (n_usable, vdim)
    mem = mem_all[df["poi_row"].values]
    # POI embeddings: all photos except held-out visits; fallback to all photos
    poi_ids = np.array(sorted(df["poi"].unique()))
    p_idx_map = pd.Series(np.arange(len(poi_ids)), index=poi_ids)
    p_idx = p_idx_map.loc[df["poi"]].values
    keep = ~held
    p_vis_k, p_cnt_k = group_mean(vis[keep], p_idx[keep], len(poi_ids))
    p_vis_all, _ = group_mean(vis, p_idx, len(poi_ids))
    no_clean = p_cnt_k == 0
    p_vis = None  # set below (neutral content for POIs whose only photos are held-out visits)
    # POI location = mean photo coordinate (float64 accumulation, stored as float32 degrees)
    ll, _ = group_mean(df[["latitude", "longitude"]].values, p_idx, len(poi_ids))
    poi_lat, poi_lon = ll[:, 0].astype(np.float32), ll[:, 1].astype(np.float32)
    poi_mem = mem_all[poi_row.loc[poi_ids].values]
    # POIs whose only photos are held-out visits get NEUTRAL content (global mean), never their own held-out photo
    g_vis = p_vis_k[~no_clean].mean(0)
    g_mem = poi_mem.mean(0)
    p_vis = np.where(no_clean[:, None], g_vis, p_vis_k)
    poi_mem_eff = np.where(no_clean[:, None], g_mem, poi_mem)
    poi_emb = np.concatenate([p_vis, poi_mem_eff], axis=1).astype(np.float32)
    g_content = np.concatenate([g_vis, g_mem]).astype(np.float32)
    poi_vis_sum = (p_vis_k.astype(np.float64) * p_cnt_k[:, None]).astype(np.float32)
    # User embeddings: train photos only
    u_vis, u_cnt = group_mean(vis[train_photo], df.loc[train_photo, "u"].astype(np.int64).values, len(users))
    u_mem, _ = group_mean(mem[train_photo], df.loc[train_photo, "u"].astype(np.int64).values, len(users))
    user_emb = np.concatenate([u_vis, u_mem], axis=1).astype(np.float32)
    assert (u_cnt > 0).all(), "a modelled user has no train photos"

    print("[6/6] writing outputs ...", flush=True)
    visit["p"] = p_idx_map.loc[visit["poi"]].values
    out = {}
    for role in ["train", "val", "test"]:
        v = visit[visit["role"] == role]
        out[f"{role}_u"] = v["u"].values.astype(np.int32)
        out[f"{role}_p"] = v["p"].values.astype(np.int32)
        out[f"{role}_ts"] = v["ts"].values.astype("datetime64[s]").astype(np.int64)
    # leave-one-user-out pieces: photo-embedding sums of each train visit (to remove the positive's own photos)
    tp_u = df.loc[train_photo, "u"].astype(np.int64).values
    tp_p = p_idx[train_photo]
    key = tp_u * len(poi_ids) + tp_p
    o = np.argsort(key, kind="stable")
    uk, st = np.unique(key[o], return_index=True)
    vsums = np.add.reduceat(vis[train_photo][o].astype(np.float64), st, axis=0)
    vcnts = np.diff(np.append(st, len(key)))
    vk = out["train_u"].astype(np.int64) * len(poi_ids) + out["train_p"].astype(np.int64)
    pos = np.searchsorted(uk, vk)
    assert (uk[pos] == vk).all()
    out["train_vsum"] = vsums[pos].astype(np.float32)
    out["train_vcnt"] = vcnts[pos].astype(np.int32)
    np.savez_compressed(f"{O}/prepared.npz", poi_vis_sum=poi_vis_sum, poi_cnt=p_cnt_k.astype(np.int32),
                        g_content=g_content, poi_no_clean=no_clean, user_ids=users.astype(str), poi_ids=poi_ids, user_emb=user_emb,
                        poi_emb=poi_emb, poi_lat=poi_lat, poi_lon=poi_lon, poi_cluster=mem_all[poi_row.loc[poi_ids].values].argmax(1).astype(np.int8),
                        **out)
    # sanity checks that would catch leakage
    stats = {
        "users": int(len(users)), "pois": int(len(poi_ids)),
        "train_visits": int(len(out["train_u"])), "val_visits": int(len(out["val_u"])), "test_visits": int(len(out["test_u"])),
        "pois_without_clean_photos": int(no_clean.sum()),
        "test_targets_without_clean_photos": float(no_clean[out["test_p"]].mean()),
        "test_pois_unseen_in_train": float(1 - np.isin(out["test_p"], out["train_p"]).mean()),
        "min_pois": a.min_pois, "seconds": round(time.time() - t0, 1),
    }
    json.dump(stats, open(f"{O}/prepare_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
