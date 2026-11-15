"""
05_poi_diag.py -- are the DBSCAN "POIs" (NB5: eps=100 m, min_samples=3) real places, or chained blobs?
DBSCAN joins photos that are < 100 m apart transitively, so a long street of photos becomes ONE giant POI.
Reads brain/photo_poi_mapping.csv (photoid, latitude, longitude, poi_id).

  python training/05_poi_diag.py --brain brain
"""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--brain", default="brain")
a = ap.parse_args()
df = pd.read_csv(f"{a.brain}/photo_poi_mapping.csv", usecols=["latitude", "longitude", "poi_id"])
g = df.groupby("poi_id")
lat0 = g["latitude"].transform("mean").values
lon0 = g["longitude"].transform("mean").values
dy = (df["latitude"].values - lat0) * 111.32
dx = (df["longitude"].values - lon0) * 111.32 * np.cos(np.radians(lat0))
df["r"] = np.sqrt(dx ** 2 + dy ** 2)                               # km from POI centroid
s = df.groupby("poi_id").agg(n=("r", "size"), rmax=("r", "max"), rmean=("r", "mean"))
print(f"POIs: {len(s):,}   photos: {s['n'].sum():,}")
print("POI radius (max distance of a photo from centroid), km:")
print(s["rmax"].describe(percentiles=[.5, .75, .9, .95, .99]).round(3).to_string())
for km in (0.3, 0.5, 1.0, 2.0, 5.0):
    big = s["rmax"] > km
    print(f"POIs with radius > {km:>3} km: {big.sum():>8,} ({big.mean():.1%} of POIs, {s.loc[big, 'n'].sum() / s['n'].sum():.1%} of photos)")
print("\n10 largest POIs by radius:")
print(s.sort_values("rmax", ascending=False).head(10).round(2).to_string())
