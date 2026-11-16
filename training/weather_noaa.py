"""
weather_noaa.py -- daily observed weather on a coarse grid, from NOAA GHCN-Daily (public, no account).

Downloads one file per year (about 180 MB each, deleted after parsing), keeps stations in the USA / Europe
boxes used by this project, and averages TMAX, TMIN and PRCP over the stations of each --cell degree cell.

  python training/weather_noaa.py --out brain/weather --years 2004 2012
Output: <out>/grid.npz  with tmean (deg C) and prcp (mm) as float16 arrays [n_lat, n_lon, n_days], NaN = no station.
"""
import argparse, os, subprocess, time
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="brain/weather")
ap.add_argument("--years", type=int, nargs=2, default=[2004, 2012])
ap.add_argument("--cell", type=float, default=0.5)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
BASE = "https://noaa-ghcn-pds.s3.amazonaws.com"
LAT0, LAT1, LON0, LON1 = 24.0, 72.0, -125.0, 40.0
t0 = time.time()


def fetch(url, path):
    subprocess.check_call(["curl", "-sS", "-L", "--retry", "3", "-o", path, url])


st_path = os.path.join(a.out, "ghcnd-stations.txt")
if not os.path.exists(st_path):
    fetch(f"{BASE}/ghcnd-stations.txt", st_path)
st = pd.read_fwf(st_path, colspecs=[(0, 11), (12, 20), (21, 30)], names=["id", "lat", "lon"])
usa = st["lat"].between(24.0, 49.5) & st["lon"].between(-125.0, -66.5)
eur = st["lat"].between(35.0, 72.0) & st["lon"].between(-10.0, 40.0)
st = st[usa | eur].copy()
st["i"] = np.floor((st["lat"] - LAT0) / a.cell).astype(int)
st["j"] = np.floor((st["lon"] - LON0) / a.cell).astype(int)
n_lat, n_lon = int(np.ceil((LAT1 - LAT0) / a.cell)), int(np.ceil((LON1 - LON0) / a.cell))
st = st[(st["i"] >= 0) & (st["i"] < n_lat) & (st["j"] >= 0) & (st["j"] < n_lon)]
cell_of = pd.Series((st["i"] * n_lon + st["j"]).values, index=st["id"].values)
print(f"stations in the USA / Europe boxes: {len(st):,}  grid {n_lat} x {n_lon}", flush=True)

day0 = np.datetime64(f"{a.years[0]}-01-01")
n_days = int((np.datetime64(f"{a.years[1] + 1}-01-01") - day0).astype(int))
S = {e: np.zeros((n_lat * n_lon, n_days), dtype=np.float32) for e in ("TMAX", "TMIN", "PRCP")}
N = {e: np.zeros((n_lat * n_lon, n_days), dtype=np.int32) for e in S}
for year in range(a.years[0], a.years[1] + 1):
    gz = os.path.join(a.out, f"{year}.csv.gz")
    fetch(f"{BASE}/csv.gz/{year}.csv.gz", gz)
    rows = 0
    for ch in pd.read_csv(gz, header=None, usecols=[0, 1, 2, 3, 5], names=["id", "date", "el", "val", "q"], chunksize=3_000_000,
                          dtype={"id": str, "date": str, "el": str, "val": np.float32, "q": str}):
        ch = ch[ch["el"].isin(["TMAX", "TMIN", "PRCP"]) & ch["q"].isna() & ch["id"].isin(cell_of.index)]
        if not len(ch):
            continue
        c = cell_of.loc[ch["id"]].values
        dts = pd.to_datetime(ch["date"], format="%Y%m%d").values.astype("datetime64[D]")
        dd = (dts - day0).astype(int)
        ok = (dd >= 0) & (dd < n_days)
        for e in S:
            m = ok & (ch["el"].values == e)
            np.add.at(S[e], (c[m], dd[m]), ch["val"].values[m] / 10.0)          # tenths of deg C / tenths of mm
            np.add.at(N[e], (c[m], dd[m]), 1)
        rows += len(ch)
    os.remove(gz)
    print(f"  {year}: {rows:,} observations kept  [{time.time() - t0:.0f}s]", flush=True)

with np.errstate(invalid="ignore", divide="ignore"):
    tmax, tmin, prcp = (np.where(N[e] > 0, S[e] / N[e], np.nan) for e in ("TMAX", "TMIN", "PRCP"))
tmean = (tmax + tmin) / 2.0
shape = (n_lat, n_lon, n_days)
np.savez_compressed(os.path.join(a.out, "grid.npz"), tmean=tmean.reshape(shape).astype(np.float16), prcp=prcp.reshape(shape).astype(np.float16),
                    lat0=LAT0, lon0=LON0, cell=a.cell, day0=str(day0))
print(f"cell-days with temperature: {np.isfinite(tmean).mean():.1%}  with precipitation: {np.isfinite(prcp).mean():.1%}  "
      f"cells with any data: {int(np.isfinite(prcp).any(1).sum()):,}")
print(f"mean temperature {np.nanmean(tmean):.1f} C, rainy (>= 1 mm) share {np.nanmean(prcp >= 1.0):.1%}")
print("saved", os.path.join(a.out, "grid.npz"), f"[{time.time() - t0:.0f}s]")
