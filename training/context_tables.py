"""
context_tables.py -- per-place context profiles for the demo's real-time filter (numpy only).

For every place: how over- or under-represented each context bucket is among its visits (smoothed log-ratio,
falling back to the place's aesthetic cluster when it has few visits), plus a crowd level per time slot.
Same definitions as 08_context_time.py and 09_context_weather.py, where these signals were evaluated offline
(they did not improve ranking, so the demo uses them as an explicit, user-visible filter and nothing more).
"""
import numpy as np

DAY = 86400
GROUPS = {"tod": 4, "weekend": 2, "season": 4, "daylight": 2, "rain": 2, "temp": 4}
TEMP_EDGES = [5.0, 15.0, 25.0]


def time_buckets(ts, lat):
    """ts: seconds (camera clock time), lat: degrees -> dict of int arrays for tod / weekend / season / daylight"""
    ts = np.asarray(ts)
    t = ts.astype("datetime64[s]")
    hour = (ts % DAY) / 3600.0
    dow = (ts // DAY + 3) % 7
    month = t.astype("datetime64[M]").astype(np.int64) % 12
    doy = (t.astype("datetime64[D]") - t.astype("datetime64[Y]")).astype(np.int64)
    decl = np.radians(-23.44) * np.cos(2 * np.pi * (doy + 10) / 365.0)
    half = np.degrees(np.arccos(np.clip(-np.tan(np.radians(lat)) * np.tan(decl), -1, 1))) / 15.0
    return {"tod": (hour // 6).astype(np.int64), "weekend": (dow >= 5).astype(np.int64),
            "season": ((month + 1) % 12) // 3, "daylight": (np.abs(hour - 12.0) < half).astype(np.int64)}


def affinity(p, b, n_pois, nb, clu, smooth=10.0):
    """p: place index per visit, b: bucket per visit (-1 = unknown) -> (n_pois, nb) float32 log-ratio table"""
    k = b >= 0
    cnt = np.zeros((n_pois, nb)); np.add.at(cnt, (p[k], b[k]), 1.0)
    glob = cnt.sum(0) / max(cnt.sum(), 1.0)
    glob = np.clip(glob, 1e-6, None)
    cc = np.zeros((int(clu.max()) + 1, nb)); np.add.at(cc, clu, cnt)
    prior = ((cc + 50.0 * glob) / (cc.sum(1, keepdims=True) + 50.0))[clu]
    return (np.log((cnt + smooth * prior) / (cnt.sum(1, keepdims=True) + smooth)) - np.log(glob)).astype(np.float32)


def build(p, t, poi_lat, poi_lon, clu, weather=None):
    """p, t: visits (place index, timestamp). weather: path to weather_noaa.py's grid.npz or None.
    Returns {"aff_<group>": (P, nb) float32, "crowd": (P, 8) float32 log1p(visits per time-of-day x weekend slot)}"""
    n_pois = len(poi_lat)
    p, t = np.asarray(p, dtype=np.int64), np.asarray(t, dtype=np.int64)
    tb = time_buckets(t, np.asarray(poi_lat, dtype=np.float64)[p])
    out = {f"aff_{g}": affinity(p, tb[g], n_pois, GROUPS[g], clu) for g in ("tod", "weekend", "season", "daylight")}
    crowd = np.zeros((n_pois, 8)); np.add.at(crowd, (p, tb["tod"] * 2 + tb["weekend"]), 1.0)
    out["crowd"] = np.log1p(crowd).astype(np.float32)
    rain = np.full(len(p), -1, dtype=np.int64); temp = rain.copy()
    if weather is not None:
        w = np.load(weather)
        cell, lat0, lon0 = float(w["cell"]), float(w["lat0"]), float(w["lon0"])
        n_lat, n_lon, n_days = w["tmean"].shape
        day0 = int(np.datetime64(str(w["day0"])).astype("datetime64[D]").astype(np.int64))
        ci = np.clip(np.floor((np.asarray(poi_lat, dtype=np.float64) - lat0) / cell).astype(np.int64), 0, n_lat - 1)
        cj = np.clip(np.floor((np.asarray(poi_lon, dtype=np.float64) - lon0) / cell).astype(np.int64), 0, n_lon - 1)
        dd = t // DAY - day0
        ok = (dd >= 0) & (dd < n_days)
        pr = w["prcp"][ci[p[ok]], cj[p[ok]], dd[ok]].astype(np.float32)
        tm = w["tmean"][ci[p[ok]], cj[p[ok]], dd[ok]].astype(np.float32)
        rain[ok] = np.where(np.isfinite(pr), (pr >= 1.0).astype(np.int64), -1)
        temp[ok] = np.where(np.isfinite(tm), np.digitize(tm, TEMP_EDGES), -1)
    out["aff_rain"] = affinity(p, rain, n_pois, 2, clu)
    out["aff_temp"] = affinity(p, temp, n_pois, 4, clu)
    return out
