"""towns.py -- the town a place card names ("Santorini, Greece"), from GeoNames cities15000.

Plain nearest-town picks suburbs next to big cities (the Empire State Building came out as Hoboken). Instead, among
towns within 30 km, pick the one with the highest log(population) - 2.5 * log(distance_km + 2): a big city wins
unless a real town is much closer.
"""
import numpy as np
import pandas as pd

COLS = ["gid", "name", "ascii", "alt", "lat", "lon", "fc", "fcode", "cc", "cc2", "a1", "a2", "a3", "a4", "pop", "elev", "dem", "tz", "mod"]


def _xyz(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1).astype(np.float32)


def assign(lat, lon, geonames_dir, max_km=30.0, chunk=2048):
    """lat, lon: place coordinates (degrees) -> dict of arrays city, region, country, cc, town_km, town_names (comma-joined spellings)"""
    C = pd.read_csv(f"{geonames_dir}/cities15000.txt", sep="\t", header=None, names=COLS,
                    usecols=["name", "ascii", "alt", "lat", "lon", "fcode", "cc", "a1", "pop"], dtype={"a1": str, "cc": str},
                    keep_default_na=False, quoting=3)
    # city districts ("PPLX") and numbered arrondissements ("Paris 15 Vaugirard") -> use the city itself
    C = C[(C["fcode"] != "PPLX") & ~C["name"].str.contains(r"\d", regex=True)].reset_index(drop=True)
    a1 = pd.read_csv(f"{geonames_dir}/admin1CodesASCII.txt", sep="\t", header=None, names=["code", "name", "ascii", "gid"],
                     keep_default_na=False, quoting=3)
    a1_name = dict(zip(a1["code"], a1["name"]))
    cc_name = {}
    for line in open(f"{geonames_dir}/countryInfo.txt", encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if not line.startswith("#") and len(f) > 4:
            cc_name[f[0]] = f[4]
    cx, px = _xyz(C["lat"].values, C["lon"].values), _xyz(np.asarray(lat), np.asarray(lon))
    logpop = np.log(np.maximum(C["pop"].values.astype(np.float64), 1.0)).astype(np.float32)
    P = len(px)
    pick = np.empty(P, dtype=np.int64); km = np.empty(P, dtype=np.float32)
    for s in range(0, P, chunk):
        d = 6371.0 * np.arccos(np.clip(px[s:s + chunk] @ cx.T, -1, 1))        # (chunk, towns) km
        score = np.where(d <= max_km, logpop[None] - 2.5 * np.log(d + 2.0), -np.inf)
        j = np.where(np.isfinite(score.max(1)), score.argmax(1), d.argmin(1))   # nothing within 30 km -> nearest
        pick[s:s + chunk] = j
        km[s:s + chunk] = d[np.arange(len(j)), j]
    cc = C["cc"].values[pick].astype(str)
    # every spelling of the town ("Roma", "Rom", "Rome"): tags that only repeat it are dropped from cards
    alt = C["name"].str.lower() + "," + C["ascii"].str.lower() + "," + C["alt"].str.lower()
    return {"city": C["name"].values[pick].astype(str), "cc": cc, "town_km": km, "town_names": alt.values[pick],
            "region": np.array([a1_name.get(f"{c}.{r}", "") for c, r in zip(cc, C["a1"].values[pick])]),
            "country": np.array([cc_name.get(c, c) for c in cc])}
