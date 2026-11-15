"""
poi_names.py -- readable place names and thumbnails from Flickr tags.

A name is "<distinctive tag>, <locality tag>", for example "Eiffel Tower, Paris":
  * tags are counted by DISTINCT USERS at a place, so one person's private tags cannot name it
  * a tag must be used by at least --min-users different users overall (drops personal tags)
  * the locality tag is the most widespread tag of the place whose places all lie within
    --loc-km of each other (a city or district: "paris" qualifies, "nature" and "france" do not);
    a place without one borrows it from the nearest place that has one
  * the distinctive tag maximises (share of the place's users) x log(N / places that use the tag)
    and must cover a smaller area than the locality (so not "France, Paris" or "Rome, Roma")

Used by 06_export_bundle.py, and runnable on its own to refresh the names of an existing bundle:
  python training/poi_names.py --brain brain --bundle bundle
"""
import argparse
import os
import re
import time
from urllib.parse import unquote_plus

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

STOP = {
    "geotagged", "geotag", "flickr", "iphone", "square", "squareformat", "instagramapp", "iphoneography", "uploaded",
    "photo", "photos", "photography", "picture", "pictures", "image", "canon", "nikon", "sony", "olympus", "pentax",
    "panasonic", "fuji", "fujifilm", "leica", "kodak", "samsung", "nokia", "cameraphone", "moblog", "unknown",
    "usa", "europe", "america", "united states", "unitedstates", "travel", "trip", "vacation", "holiday", "holidays",
    "tourism", "tourist", "day", "night", "summer", "winter", "spring", "autumn", "fall", "color", "colour",
    "black and white", "blackandwhite", "bw", "hdr", "digital", "film", "lomo", "raw", "nef", "jpg", "jpeg",
    "panorama", "autostitch", "enblend", "hugin", "photoshop", "lightroom", "picasa", "explore", "explored",
    "interestingness", "me", "family", "friends", "people", "fun", "walk", "weekend", "birthday", "wedding", "party",
    "unfound", "foundphotos", "upcoming", "foursquare", "eyefi", "eye-fi", "uploaded by instagram", "instagram",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
    "december",
}
_JUNK = re.compile(r"^[^a-z]*$|^[a-z]{0,4}[-_ ]?\d+[a-z]{0,3}$|\d{4}|[:=]|^.{0,2}$|^.{41,}$")


def _clean(raw):
    """raw URL-encoded tag -> normalised tag, or '' if it should be ignored."""
    t = unquote_plus(raw).strip().lower()
    return "" if (t in STOP or _JUNK.search(t)) else t


def _km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _title(t):
    return " ".join(w if (w.isupper() and len(w) <= 4) else w.capitalize() for w in t.split())


def build_names(brain, poi_ids, lat, lon, min_users=3, loc_km=40.0, loc_min_pois=15, near_km=8.0, verbose=True):
    """Returns (names, urls): two str arrays aligned with poi_ids."""
    t0 = time.time()
    n_pois = len(poi_ids)
    pm = pd.read_csv(os.path.join(brain, "photo_poi_mapping.csv"), usecols=["photoid", "poi_id"]).drop_duplicates("photoid")
    poi_idx = pd.Series(np.arange(n_pois), index=poi_ids)
    pm = pm[pm["poi_id"].isin(poi_idx.index)]
    photo_p = pd.Series(poi_idx.loc[pm["poi_id"]].values, index=pm["photoid"].values)

    urls = np.array([""] * n_pois, dtype=object)
    parts, cache = [], {}
    reader = pd.read_csv(os.path.join(brain, "usa_europe_geotagged.csv"), usecols=["photoid", "uid", "downloadurl", "usertags"],
                         chunksize=1000000, dtype={"photoid": "int64", "uid": "str", "downloadurl": "str", "usertags": "str"})
    for ch in reader:
        ch = ch[ch["photoid"].isin(photo_p.index)]
        if not len(ch):
            continue
        ch = ch.assign(p=photo_p.loc[ch["photoid"]].values)
        first = ch.dropna(subset=["downloadurl"]).drop_duplicates("p")
        need = urls[first["p"].values] == ""
        urls[first["p"].values[need]] = [unquote_plus(u) for u in first["downloadurl"].values[need]]
        ch = ch.dropna(subset=["usertags", "uid"])[["p", "uid", "usertags"]]
        ch = ch.assign(tag=ch["usertags"].str.split(",")).drop(columns="usertags").explode("tag").dropna(subset=["tag"])
        for raw in ch["tag"].unique():
            if raw not in cache:
                cache[raw] = _clean(raw)
        ch["tag"] = ch["tag"].map(cache)
        parts.append(ch[ch["tag"] != ""].drop_duplicates())
    T = pd.concat(parts, ignore_index=True).drop_duplicates()            # one row per (place, user, tag)
    del parts
    if verbose:
        print(f"    (place, user, tag) rows: {len(T):,}  distinct tags: {T['tag'].nunique():,}  [{time.time() - t0:.0f}s]", flush=True)

    tag_users = T.drop_duplicates(["uid", "tag"]).groupby("tag").size()
    T = T[T["tag"].isin(tag_users.index[tag_users >= min_users])]
    n_users = T.drop_duplicates(["p", "uid"]).groupby("p").size()        # users with at least one usable tag
    G = T.groupby(["p", "tag"]).size().rename("sup").reset_index()       # sup = users of this tag at this place
    del T
    G["share"] = G["sup"].values / n_users.loc[G["p"]].values
    df = G.groupby("tag").size()
    G["df"] = df.loc[G["tag"]].values
    G["idf"] = np.log(n_pois / G["df"].values)

    # locality tags: all places that carry the tag lie close together
    G["lat"], G["lon"] = lat[G["p"].values], lon[G["p"].values]
    med = G.groupby("tag")[["lat", "lon"]].transform("median")
    G["d"] = _km(G["lat"].values, G["lon"].values, med["lat"].values, med["lon"].values)
    spread = G.groupby("tag")["d"].quantile(0.7)
    is_loc = (spread <= loc_km) & (df.loc[spread.index] >= loc_min_pois)
    G["loc"] = is_loc.loc[G["tag"]].values
    if verbose:
        top = df[is_loc[is_loc].index].sort_values(ascending=False).head(12)
        print(f"    usable tags: {len(df):,}  locality tags: {int(is_loc.sum()):,}  e.g. {', '.join(top.index)}", flush=True)

    G["spread"] = spread.loc[G["tag"]].values
    # a tag needs two users where the place has several; single-visitor places take what they have
    ok = G[(G["sup"] >= 2) | (n_users.loc[G["p"]].values < 3)]
    loc = ok[ok["loc"] & (ok["share"] >= 0.25) & (ok["d"] <= loc_km)]     # the place must lie inside the tag's area
    loc = loc.sort_values(["p", "df"], ascending=[True, False]).drop_duplicates("p")
    loc_tag = pd.Series(loc["tag"].values, index=loc["p"].values)
    # places without a locality tag of their own borrow it from the nearest place that has one (within near_km)
    has = loc_tag.index.values
    missing = np.setdiff1d(np.arange(n_pois), has)
    if len(has) and len(missing):
        def xyz(i):
            la, lo = np.radians(lat[i]), np.radians(lon[i])
            return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1) * 6371.0
        dist, j = cKDTree(xyz(has)).query(xyz(missing), k=1)
        near = dist <= near_km
        far_tag = pd.Series(loc_tag.values[j[~near & (dist <= 60.0)]], index=missing[~near & (dist <= 60.0)])
        loc_tag = pd.concat([loc_tag, pd.Series(loc_tag.values[j[near]], index=missing[near])])
    else:
        far_tag = pd.Series([], dtype=object)
    # the landmark must be more specific than the locality (rules out "France, Paris" and "Rome, Roma");
    # without any locality it must at least be tied to one area (rules out "Lane", "Will")
    b = loc_tag.reindex(ok["p"]).values
    has_b = pd.notnull(b)
    b_spread = spread.reindex(b).values
    b_df = df.reindex(b).values
    specific = has_b & (ok["tag"].values != b) & (ok["spread"].values <= 0.6 * b_spread) & (ok["df"].values <= 0.5 * b_df)
    local = ~has_b & (ok["spread"].values <= loc_km) & (ok["df"].values >= 2)
    cand = ok[specific | local].assign(score=lambda x: x["share"] * x["idf"])
    best = cand.sort_values(["p", "score"], ascending=[True, False]).drop_duplicates("p")
    best_tag = pd.Series(best["tag"].values, index=best["p"].values)

    names = np.array([""] * n_pois, dtype=object)
    for p in range(n_pois):
        a, b = best_tag.get(p), loc_tag.get(p)
        if a and b and b not in a:
            names[p] = f"{_title(a)}, {_title(b)}"
        elif a or b:
            names[p] = _title(a or b)
        elif far_tag.get(p):
            names[p] = f"Near {_title(far_tag.get(p))}"
        else:
            names[p] = "Unnamed place"
    if verbose:
        full = np.mean([", " in n for n in names])
        none = np.mean([n.startswith(("Near ", "Unnamed")) for n in names])
        print(f"    landmark + locality: {full:.0%}   one of the two: {1 - full - none:.0%}   'Near X' or unnamed: {none:.0%}   "
              f"thumbnails: {np.mean(urls != ''):.0%}   [{time.time() - t0:.0f}s]", flush=True)
    return names.astype(str), urls.astype(str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="brain")
    ap.add_argument("--bundle", default="bundle")
    ap.add_argument("--min-users", type=int, default=3)
    ap.add_argument("--loc-km", type=float, default=40.0)
    a = ap.parse_args()
    path = os.path.join(a.bundle, "pois.npz")
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    z.close()
    names, urls = build_names(a.brain, d["poi_ids"], d["lat"].astype(np.float64), d["lon"].astype(np.float64),
                              min_users=a.min_users, loc_km=a.loc_km)
    d["name"] = names
    np.savez_compressed(path, **d)
    order = np.argsort(-d["cnt"])
    print("most visited places:")
    for p in order[:40]:
        print(f"  {int(d['cnt'][p]):6d}  {names[p]}")
    rng = np.random.RandomState(0)
    print("random places:")
    for p in rng.choice(len(names), 25, replace=False):
        print(f"  {int(d['cnt'][p]):6d}  {names[p]}")
    print("saved", path)
