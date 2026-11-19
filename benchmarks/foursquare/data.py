"""Foursquare NYC splits from GETNext, and evaluation samples built exactly like GETNext's TrajectoryDatasetVal.

Everything the model learns from (POI set, categories, popularity, user taste, distance history) comes from the
training file only. Validation / test files contribute nothing but their own trajectory prefixes and targets.
"""
import os
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "external", "GETNext")
SPLITS = ("train", "val", "test")


def csv_path(split):
    return os.path.join(REPO, "dataset", "NYC", f"NYC_{split}.csv")


def ensure_extracted():
    """GETNext ships the splits as dataset/NYC.zip; extract only the three split CSVs (not the 620 MB graph)."""
    if all(os.path.exists(csv_path(s)) for s in SPLITS):
        return
    zp = os.path.join(REPO, "dataset", "NYC.zip")
    if not os.path.exists(zp):
        raise SystemExit(f"{zp} not found. Clone https://github.com/songyangme/GETNext into {REPO}")
    with zipfile.ZipFile(zp) as z:
        for s in SPLITS:
            z.extract(f"NYC/NYC_{s}.csv", os.path.join(REPO, "dataset"))


def load_splits():
    ensure_extracted()
    out = {}
    for s in SPLITS:
        d = pd.read_csv(csv_path(s), dtype={"user_id": str, "POI_id": str, "POI_catid": str, "trajectory_id": str})
        d["t"] = pd.to_datetime(d["UTC_time"]).astype("int64") // 10**9          # seconds, for ordering checks
        d["row"] = np.arange(len(d))                                               # file order
        out[s] = d
    return out


class Index:
    """POIs, users and categories of the TRAINING file (GETNext's POI set = nodes of the train-only graph;
    category and coordinates from a POI's first training check-in, as in build_graph.py)."""

    def __init__(self, train):
        first = train.drop_duplicates("POI_id", keep="first")
        self.poi_ids = first["POI_id"].tolist()
        self.pid = {p: i for i, p in enumerate(self.poi_ids)}
        self.lat = first["latitude"].to_numpy(np.float64)
        self.lon = first["longitude"].to_numpy(np.float64)
        cats = first.drop_duplicates("POI_catid")
        self.cat_ids = cats["POI_catid"].tolist()
        self.cat_names = cats["POI_catname"].tolist()
        cid = {c: i for i, c in enumerate(self.cat_ids)}
        self.poi_cat = first["POI_catid"].map(cid).to_numpy(np.int64)
        self.user_ids = sorted(train["user_id"].unique(), key=int)
        self.uid = {u: i for i, u in enumerate(self.user_ids)}
        self.n_pois, self.n_users, self.n_cats = len(self.poi_ids), len(self.user_ids), len(self.cat_ids)


class TrainStream:
    """Training check-ins grouped by user, in time order (file order breaks ties), with trajectory positions."""

    def __init__(self, train, ix):
        d = train.assign(u=train["user_id"].map(ix.uid), p=train["POI_id"].map(ix.pid))
        d = d.sort_values(["u", "t", "row"], kind="stable").reset_index(drop=True)
        self.u = d["u"].to_numpy(np.int64)
        self.p = d["p"].to_numpy(np.int64)
        self.t = d["t"].to_numpy(np.int64)
        self.traj = d["trajectory_id"].to_numpy()
        self.key = list(zip(d["user_id"], d["POI_id"], d["UTC_time"]))        # for leakage checks
        same = np.r_[False, self.traj[1:] == self.traj[:-1]]
        self.prev_same_traj = same                                              # row has an earlier check-in in its trajectory
        # positives for training: check-ins with a previous check-in in the same trajectory, like GETNext's labels
        self.positives = np.nonzero(same)[0]
        # consecutive pairs inside a trajectory, for the Markov baseline
        self.pairs = np.stack([self.p[:-1][same[1:]], self.p[1:][same[1:]]], 1)


def eval_samples(df, ix, short_traj_thres=2):
    """GETNext TrajectoryDatasetVal: skip users not in train, drop POIs not in train, inputs = all but the last
    remaining check-in, skip if fewer than `short_traj_thres` inputs; score only the LAST step."""
    S = {"u": [], "prefix": [], "prefix_t": [], "target": [], "target_t": [], "traj": [], "key": []}
    for traj, g in df.groupby("trajectory_id", sort=False):           # file order inside a trajectory, as GETNext
        user = traj.split("_")[0]
        if user not in ix.uid:
            continue
        g = g[g["POI_id"].isin(ix.pid)]
        p = g["POI_id"].map(ix.pid).to_numpy(np.int64)
        if len(p) - 1 < short_traj_thres:
            continue
        S["u"].append(ix.uid[user]); S["prefix"].append(p[:-1]); S["prefix_t"].append(g["t"].to_numpy()[:-1])
        S["target"].append(int(p[-1])); S["target_t"].append(int(g["t"].iloc[-1])); S["traj"].append(traj)
        last = g.iloc[-1]
        S["key"].append((last["user_id"], last["POI_id"], last["UTC_time"]))
    S["u"] = np.array(S["u"], dtype=np.int64)
    S["target"] = np.array(S["target"], dtype=np.int64)
    S["target_t"] = np.array(S["target_t"], dtype=np.int64)
    return S


def eval_refs(S, stream, n_users, M):
    """Reference check-ins for the distance features, most recent first: the trajectory prefix (reversed),
    then the user's latest TRAINING check-ins. Returns (N, M) poi idx and (N, M) timestamps, -1 = padding."""
    counts = np.bincount(stream.u, minlength=n_users)
    end = np.cumsum(counts)
    N = len(S["u"])
    ref = -np.ones((N, M), dtype=np.int64)
    ref_t = -np.ones((N, M), dtype=np.int64)
    for i in range(N):
        u = S["u"][i]
        pre, pre_t = S["prefix"][i][::-1], S["prefix_t"][i][::-1]
        lo, hi = end[u] - counts[u], end[u]
        hist, hist_t = stream.p[lo:hi][::-1], stream.t[lo:hi][::-1]
        r = np.concatenate([pre, hist])[:M]
        rt = np.concatenate([pre_t, hist_t])[:M]
        ref[i, :len(r)] = r
        ref_t[i, :len(rt)] = rt
    return ref, ref_t


def user_cat_counts(stream, ix):
    """(n_users, n_cats) training check-ins per category: the user's taste, from training only."""
    C = np.zeros((ix.n_users, ix.n_cats), dtype=np.float32)
    np.add.at(C, (stream.u, ix.poi_cat[stream.p]), 1.0)
    return C
