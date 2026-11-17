"""
16_export_swipe.py -- data bundle for the swipe app's backend (server/), sized for a 512 MB host.

Inputs : bundle/ (from 06_export_bundle.py), brain/poi_details<places>.npz (from 15_place_details.py)
Outputs: bundle_swipe/
  core.npz      numbers the ranking needs in memory: emb (f16), lat, lon, cnt, cluster, valid, mood (f16), crowd (f16)
  weights.npz   model weights (copied)
  meta.json     clusters, moods, pop_norm, geo_m, run name
  places.db     SQLite: card text per place (name, photos, town, tags, months, counts) + a towns table for search.
                Read per request, so the text never sits in memory.
  .venv\\Scripts\\python.exe training\\16_export_swipe.py
"""
import argparse, json, os, re, shutil, sqlite3, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from towns import assign  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", default="bundle")
ap.add_argument("--details", default="brain/poi_details_v2.npz")
ap.add_argument("--photos", default="brain/place_photos_v2.npz", help="18_place_photos.py output; skipped if missing")
ap.add_argument("--out", default="bundle_swipe")
ap.add_argument("--geonames", default="staging/geonames")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
B = np.load(f"{a.bundle}/pois.npz")
D = np.load(a.details)
assert (B["poi_ids"] == D["poi_ids"]).all(), "details and bundle are not aligned"
meta = json.load(open(f"{a.bundle}/meta.json", encoding="utf-8"))
P = len(B["poi_ids"])
# towns are recomputed here (towns.py) so a better town rule does not need the slow photo pass of 15_place_details.py again
_t = assign(B["lat"].astype(np.float64), B["lon"].astype(np.float64), a.geonames)
city, region, country, cc = _t["city"].tolist(), _t["region"].tolist(), _t["country"].tolist(), _t["cc"].tolist()
town_km = _t["town_km"]

# "generic" names: no name, or a name shared by many places (usually just the city: "Chicago", "Nyc").
# The app titles these from their tags instead and ranks them a little lower.
_, inv, cnt_name = np.unique(B["name"], return_inverse=True, return_counts=True)
generic = (B["name"] == "Unnamed place") | (cnt_name[inv] >= 20) | (np.char.lower(B["name"].astype(str)) == np.char.lower(np.asarray(city)))
np.savez(f"{a.out}/core.npz", emb=B["emb"].astype(np.float16), lat=B["lat"].astype(np.float32), lon=B["lon"].astype(np.float32),
         cnt=B["cnt"].astype(np.float32), cluster=B["cluster"].astype(np.int8), valid=~B["no_clean"],
         mood=B["mood"].astype(np.float16), crowd=B["crowd"].astype(np.float16), generic=generic)
print(f"generic names: {generic.sum():,} of {len(generic):,}")
shutil.copy(f"{a.bundle}/weights.npz", f"{a.out}/weights.npz")
json.dump({"run": meta["run"], "clusters": meta["clusters"], "moods": meta["moods"], "pop_norm": meta["pop_norm"],
           "geo_m": meta["config"].get("geo_m", 30), "mem_dim": meta.get("mem_dim", 6), "n_places": P},
          open(f"{a.out}/meta.json", "w", encoding="utf-8"), indent=1)

name, url = B["name"].tolist(), B["url"].tolist()
tags, best = D["tags"], D["best_months"].tolist()
n_photos, n_visitors = D["n_photos"], D["n_visitors"]
# up to 6 chosen photos per place ("server/id_secret" keys, best first); the old single link stays as a fallback
photos = [""] * P
if os.path.exists(a.photos):
    Ph = np.load(a.photos)
    o = np.lexsort((Ph["rank"], Ph["place"]))
    pl, keys = Ph["place"][o], Ph["key"][o]
    starts = np.r_[0, np.nonzero(np.diff(pl))[0] + 1]
    ends = np.r_[starts[1:], len(pl)]
    for s0, e0 in zip(starts, ends):
        photos[int(pl[s0])] = "|".join(keys[s0:e0].tolist())
    print(f"photos: {sum(1 for x in photos if x):,} places have chosen photos")
else:
    print("photos: none (run 17_photo_scores.py and 18_place_photos.py)")


# local spellings of countries that GeoNames' English list misses and that show up as tags
COUNTRY_LOCAL = {"IT": {"italia", "italy"}, "DE": {"deutschland", "germany"}, "ES": {"espana", "españa", "spain"},
                 "FR": {"france"}, "NL": {"nederland", "holland"}, "AT": {"österreich", "osterreich"}, "CH": {"schweiz", "suisse", "svizzera"},
                 "GB": {"uk", "england", "united kingdom", "great britain", "britain", "scotland", "wales"},
                 "US": {"usa", "united states", "america"}, "CZ": {"česko", "czech republic"}, "PT": {"portugal"},
                 "BE": {"belgique", "belgië", "belgium"}, "DK": {"danmark", "denmark"}, "SE": {"sverige", "sweden"},
                 "NO": {"norge", "norway"}, "GR": {"greece", "hellas"}, "IE": {"ireland", "éire"}, "CA": {"canada"}}


# camera, upload-tool and file tags say nothing about the place
JUNK = re.compile(r"canon|nikon|\beos\b|sony|olympus|fuji|panasonic|lumix|pentax|leica|iphone|camera|dslr|\bdsc|\bimg|geotag|"
                  r"flickr|upload|\bexif|\d+\s?mm\b|\d{3,}|^\d+$")


def clean_tags(i):
    """top tags minus ones that just repeat the place's town, region or country, or its name"""
    drop = {city[i].lower(), region[i].lower(), country[i].lower(), cc[i].lower(), name[i].lower(), *_t["town_names"][i].split(",")}
    drop |= COUNTRY_LOCAL.get(cc[i], set())
    out = []
    for t in tags[i]:
        t = str(t).strip()
        if t and t.lower() not in drop and not JUNK.search(t.lower()) and not any(t.lower() == w for w in name[i].lower().split(", ")):
            out.append(t)
    return "|".join(out[:4])


db = f"{a.out}/places.db"
if os.path.exists(db):
    os.remove(db)                     # bundle_swipe/ is a build output, regenerated from bundle/ and brain/
con = sqlite3.connect(db)
con.execute("""CREATE TABLE place (id INTEGER PRIMARY KEY, name TEXT, url TEXT, city TEXT, region TEXT, country TEXT, cc TEXT,
               town_km REAL, n_photos INTEGER, n_visitors INTEGER, best_months TEXT, tags TEXT, photos TEXT)""")
con.executemany("INSERT INTO place VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ((i, name[i], url[i].replace("http://", "https://", 1), city[i], region[i], country[i], cc[i], float(town_km[i]),
                  int(n_photos[i]), int(n_visitors[i]), best[i], clean_tags(i), photos[i]) for i in range(P)))
# towns that have places with real activity: what the search box offers
valid = ~B["no_clean"]
lat, lon = B["lat"].astype(np.float64), B["lon"].astype(np.float64)
towns = {}
for i in np.nonzero(valid & (town_km < 30))[0]:
    k = (city[i], region[i], cc[i])
    t = towns.setdefault(k, [0, 0.0, 0.0, 0])
    w = int(n_visitors[i]) + 1
    t[0] += 1; t[1] += lat[i] * w; t[2] += lon[i] * w; t[3] += w
con.execute("CREATE TABLE town (name TEXT, region TEXT, country TEXT, cc TEXT, lat REAL, lon REAL, n_places INTEGER, weight INTEGER)")
cc2country = dict(zip(cc, country))
con.executemany("INSERT INTO town VALUES (?,?,?,?,?,?,?,?)",
                ((k[0], k[1], cc2country.get(k[2], k[2]), k[2], v[1] / v[3], v[2] / v[3], v[0], v[3])
                 for k, v in towns.items() if v[0] >= 5))
con.execute("CREATE INDEX town_name ON town (name COLLATE NOCASE)")
con.commit()
n_towns = con.execute("SELECT COUNT(*) FROM town").fetchone()[0]
con.execute("VACUUM")
con.close()
for f in sorted(os.listdir(a.out)):
    print(f"{f:14s}{os.path.getsize(os.path.join(a.out, f)) / 1e6:8.1f} MB")
print(f"{P:,} places, {n_towns:,} searchable towns")
