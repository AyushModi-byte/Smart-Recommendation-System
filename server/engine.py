"""
Ranking engine for the swipe app (numpy only, no torch).

A user's taste is the mean embedding of the places they accepted, and their accepted places are the location history
the model's geo features use. training/14_swipe_check.py tested exactly this substitution offline: taste from visited
places scores as well as taste from the user's own photos, and with 1 / 3 accepted places the model beats
popular-within-25-km (HR@10 0.145 vs 0.117, 0.156 vs 0.097).

Final score over places in the chosen area = z(log popularity) + 2 * z(model), the blend tuned for destination mode
(03_destination_eval.py), plus explicit user choices (mood, scenery, hidden gems) and a penalty for looking like a
rejected place.

Live conditions (Open-Meteo, no key) are an explicit, visible filter, like mood: in offline tests weather, time and
crowd did not improve ranking (09_context_weather.py), so they only reorder when the user would expect it: rain,
cold (< 5 °C) or heat (> 25 °C) favour places people photograph in those conditions, and "quieter" demotes places
busy in the current time slot (photographers per time-of-day x weekend slot, the stand-in for traffic).
"""
import datetime as dt, difflib, json, math, os, re, sqlite3, sys, threading, time, urllib.parse, urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
from aurapath_np import AuraNP, geo_feats, haversine_km  # noqa: E402

CAT_SHORT = ["Events & Fun", "Cityscapes", "Heritage", "Scenic & Water", "Nightlife", "Nature & Wildlife"]
CAT_PHRASE = ["spot for events and days out", "slice of city architecture", "heritage spot", "scenic waterside spot",
              "spot for nights out", "green escape"]
MOOD_ADJ = {"Relaxed": "A relaxed", "Lively": "A lively", "Romantic": "A romantic", "Adventurous": "An adventurous",
            "Curious": "An intriguing", "Playful": "A playful"}
MONTH_NAME = {"JAN": "Jan", "FEB": "Feb", "MAR": "Mar", "APR": "Apr", "MAY": "May", "JUN": "Jun", "JUL": "Jul",
              "AUG": "Aug", "SEP": "Sep", "OCT": "Oct", "NOV": "Nov", "DEC": "Dec"}
MODEL_W, CHOICE_W, REJECT_W, GENERIC_PEN = 2.0, 1.0, 0.75, 0.25
WEATHER_W, QUIET_W = 0.75, 0.75
TOD = ["night", "morning", "afternoon", "evening"]                   # hour // 6, as in training/context_tables.py
TEMP_EDGES = [5.0, 15.0, 25.0]
RAIN_CODES = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}   # WMO codes: drizzle, rain, showers, storms
COND_TTL = 900                                                        # seconds a weather lookup is reused
MAX_CAND = 20000


def _key(title):
    """spelling-insensitive title key: "Empire State Building" == "Empirestatebuilding" """
    return re.sub(r"[^a-z]", "", title.lower())


def _z(x):
    x = x.astype(np.float32)
    return (x - x.mean()) / max(float(x.std()), 1e-6)


def bundle_dir():
    for d in (os.environ.get("SWIPE_BUNDLE"), os.path.join(HERE, "bundle_swipe"), os.path.join(HERE, "..", "bundle_swipe")):
        if d and os.path.exists(os.path.join(d, "core.npz")):
            return d
    raise FileNotFoundError("bundle_swipe not found: set SWIPE_BUNDLE or run training/16_export_swipe.py")


class Engine:
    def __init__(self, path=None):
        path = path or bundle_dir()
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.net = AuraNP(np.load(os.path.join(path, "weights.npz")))
        z = np.load(os.path.join(path, "core.npz"))
        self.emb = z["emb"]                                    # float16 (P, 262): 256 visual + 6 cluster memberships
        self.lat, self.lon = z["lat"].astype(np.float64), z["lon"].astype(np.float64)
        self.cnt, self.cluster, self.valid = z["cnt"], z["cluster"].astype(np.int64), z["valid"]
        self.mood, self.generic = z["mood"], z["generic"]
        self.crowd = z["crowd"]                                # (P, 8) log1p photographers per time-of-day x weekend slot
        self.logpop = np.log1p(self.cnt).astype(np.float32)
        self.moods = self.meta["moods"]
        self.n_mem = int(self.meta.get("mem_dim", 6))
        self.pop_norm, self.geo_m = float(self.meta["pop_norm"]), int(self.meta["geo_m"])
        self.n = len(self.lat)
        self.db_path = os.path.join(path, "places.db")
        self._local = threading.local()
        ctx = next((c for c in (os.path.join(path, "context.npz"), os.path.join(HERE, "context.npz")) if os.path.exists(c)), None)
        self.ctx = None
        if ctx:                                                 # weather affinity per place (training/context_tables.py)
            z = np.load(ctx)
            if len(z["aff_rain"]) == self.n:
                self.ctx = {k: z[k].astype(np.float32) for k in ("aff_rain", "aff_temp")}
        self._cond_cache, self._cond_lock = {}, threading.Lock()

    # ---------- storage ----------
    def _db(self):
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)
            con.row_factory = sqlite3.Row
            self._local.con = con
        return con

    def rows(self, ids):
        ids = [int(i) for i in ids]
        if not ids:
            return {}
        q = f"SELECT * FROM place WHERE id IN ({','.join('?' * len(ids))})"
        return {r["id"]: r for r in self._db().execute(q, ids)}

    def towns(self, q, limit=8):
        q = (q or "").strip()
        if len(q) < 2:
            return []
        cur = self._db().execute("SELECT name, region, country, cc, lat, lon, n_places FROM town WHERE name LIKE ? "
                                 "ORDER BY (name LIKE ?) DESC, weight DESC LIMIT ?", (f"%{q}%", f"{q}%", limit))
        return [dict(r) for r in cur]

    def top_towns(self, limit=12):
        cur = self._db().execute("SELECT name, region, country, cc, lat, lon, n_places FROM town ORDER BY weight DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur]

    # ---------- text ----------
    def title(self, i, r):
        tags = [t for t in (r["tags"] or "").split("|") if t]
        if not self.generic[i]:
            return r["name"].split(", ")[0]           # "Millennium Park, Chicago" -> the town has its own line
        if tags:
            return tags[0].title()
        return f"{CAT_SHORT[self.cluster[i]]} spot near {r['city']}"

    def describe(self, i, r):
        mood = self.moods[int(np.argmax(self.mood[i]))]
        km = float(r["town_km"])
        where = f"in {r['city']}" if km < 8 else f"{km:.0f} km from {r['city']}"      # town points are city centres
        s = f"{MOOD_ADJ.get(mood, 'A')} {CAT_PHRASE[self.cluster[i]]} {where}."
        n = int(r["n_visitors"])
        best = (r["best_months"] or "") if n >= 40 else ""          # month patterns of small places are noise
        if best and best != "ALL YEAR":
            months = best
            for k, v in MONTH_NAME.items():
                months = months.replace(k, v)
            s += f" Photographed by {n:,} people, busiest in {months.replace('-', '–').replace(' & ', ' and ')}."
        elif best == "ALL YEAR":
            s += f" Photographed by {n:,} people, all year round."
        elif n >= 2:
            s += f" Photographed by {n:,} people."
        else:
            s += " A rarely photographed spot."
        return s

    @staticmethod
    def photos(r):
        """chosen photos first (live.staticflickr.com), then the old single link as a fallback; [{src, page}]"""
        out, seen = [], set()
        keys = [k for k in ((r["photos"] if "photos" in r.keys() else "") or "").split("|") if k]
        srcs = [f"https://live.staticflickr.com/{k}.jpg" for k in keys] + ([r["url"]] if r["url"] else [])
        for src in srcs:
            try:
                pid = src.rsplit("/", 1)[1].split("_")[0]
            except Exception:
                continue
            if pid in seen:
                continue
            seen.add(pid)
            out.append({"src": src, "page": "https://www.flickr.com/photo.gne?id=" + pid})
        return out

    def card(self, i, r, extra=None):
        photos = self.photos(r)
        url = photos[0]["src"] if photos else ""
        page = photos[0]["page"] if photos else ""
        c = {"id": int(i), "name": self.title(i, r), "lat": float(self.lat[i]), "lon": float(self.lon[i]), "photo": url,
             "photo_page": page, "photos": photos, "town": r["city"], "region": r["region"], "country": r["country"], "cc": r["cc"],
             "tags": [t for t in (r["tags"] or "").split("|") if t][:3 if self.generic[i] else 4],
             "best_months": (r["best_months"] or "") if int(r["n_visitors"]) >= 40 else "", "n_photos": int(r["n_photos"]), "n_visitors": int(r["n_visitors"]),
             "category": CAT_SHORT[self.cluster[i]], "category_id": int(self.cluster[i]),
             "mood": self.moods[int(np.argmax(self.mood[i]))], "description": self.describe(i, r)}
        if self.generic[i] and c["tags"]:
            c["tags"] = c["tags"][1:]                       # the first tag is already the title
        if extra:
            c.update(extra)
        return c

    # ---------- live conditions ----------
    def conditions(self, lat, lon):
        """Current weather and local time at a point (Open-Meteo, no key), cached per ~10 km cell. None if offline."""
        key = (round(float(lat), 1), round(float(lon), 1))
        with self._cond_lock:
            hit = self._cond_cache.get(key)
        if hit and time.time() - hit[0] < COND_TTL:
            return hit[1]
        url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(
            {"latitude": key[0], "longitude": key[1], "current": "temperature_2m,precipitation,weather_code,is_day",
             "timezone": "auto"})
        try:
            with urllib.request.urlopen(url, timeout=4) as r:
                cur = json.loads(r.read().decode("utf-8"))["current"]
            t = dt.datetime.fromisoformat(cur["time"])
            temp, rain = float(cur["temperature_2m"]), float(cur["precipitation"]) > 0.0 or int(cur.get("weather_code", 0)) in RAIN_CODES
            out = {"temp_c": round(temp), "rain": bool(rain), "is_day": bool(cur["is_day"]), "local_time": t.strftime("%H:%M"),
                   "tod": t.hour // 6, "weekend": t.weekday() >= 5, "temp_band": int(np.digitize(temp, TEMP_EDGES))}
        except Exception:
            out = None
        with self._cond_lock:
            if len(self._cond_cache) > 5000:
                self._cond_cache.clear()
            self._cond_cache[key] = (time.time(), out)
        return out

    def _context(self, cand, weather, quiet, cond):
        """-> (score adjustment, notes shown to the user)"""
        adj, notes = np.zeros(len(cand), dtype=np.float32), []
        if cond is None:
            if weather or quiet:
                notes.append("Live weather is unavailable right now.")
            return adj, notes
        if weather and self.ctx is not None:
            if cond["rain"]:
                adj += WEATHER_W * _z(self.ctx["aff_rain"][cand, 1])
                notes.append(f"Raining now ({cond['temp_c']} °C): places people photograph in the rain come first.")
            elif cond["temp_band"] in (0, 3):
                adj += WEATHER_W * _z(self.ctx["aff_temp"][cand, cond["temp_band"]])
                notes.append(f"{'Cold' if cond['temp_band'] == 0 else 'Hot'} out ({cond['temp_c']} °C): places people "
                             f"visit in this weather come first.")
        if quiet:
            slot = cond["tod"] * 2 + int(cond["weekend"])
            adj -= QUIET_W * _z(self.crowd[cand, slot].astype(np.float32))
            notes.append(f"Quieter places for a {'weekend' if cond['weekend'] else 'weekday'} {TOD[cond['tod']]}.")
        return adj, notes

    # ---------- ranking ----------
    def _model(self, accepted, cand):
        taste = self.emb[accepted].astype(np.float32).mean(0, keepdims=True)
        ut = self.net.tower(taste, "user_tower")
        ref = np.asarray(accepted[::-1][:self.geo_m], dtype=np.int64)          # most recent first
        out = []
        for s in range(0, len(cand), 8192):
            c = cand[s:s + 8192]
            pt = self.net.tower(self.emb[c].astype(np.float32), "poi_tower")
            out.append(self.net.head(ut, pt, geo_feats(ref, c, self.lat, self.lon, self.cnt, self.pop_norm)))
        return np.concatenate(out)

    def _vis(self, idx):
        v = self.emb[idx, :-self.n_mem].astype(np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)

    def recommend(self, lat, lon, radius_km=25.0, accepted=(), rejected=(), seen=(), mood=None, category=None,
                  hidden=False, n=6, explore=0.2, seed=None, weather=True, quiet=False):
        accepted = [int(i) for i in accepted if 0 <= int(i) < self.n]
        rejected = [int(i) for i in rejected if 0 <= int(i) < self.n]
        seen = [int(i) for i in seen if 0 <= int(i) < self.n]
        rng = np.random.default_rng(seed)
        d = haversine_km(self.lat, self.lon, float(lat), float(lon))
        keep = self.valid & (d <= float(radius_km)) & (self.cnt >= 2)
        keep[accepted + rejected + seen] = False
        cand = np.nonzero(keep)[0]
        cond = self.conditions(lat, lon) if (weather or quiet) else None
        live = {"conditions": cond, "notes": []}
        if len(cand) == 0:
            return {"cards": [], "n_candidates": 0, **live}
        if len(cand) > MAX_CAND:
            cand = cand[np.argsort(-self.logpop[cand])[:MAX_CAND]]
        if (~self.generic[cand]).sum() >= 100:            # enough properly named places: skip tag-titled ones
            cand = cand[~self.generic[cand]]
        touched = np.asarray((accepted + rejected + seen)[-200:], dtype=np.int64)
        if len(touched):                                   # same landmark under another spelling
            dd = haversine_km(self.lat[cand][:, None], self.lon[cand][:, None], self.lat[touched][None], self.lon[touched][None])
            cand = cand[dd.min(1) >= 0.1]
            if len(cand) == 0:
                return {"cards": [], "n_candidates": 0, **live}
        score = (-0.5 if hidden else 1.0) * _z(self.logpop[cand])
        if accepted:
            score = score + MODEL_W * _z(self._model(accepted, cand))
        if mood in self.moods:
            score = score + CHOICE_W * _z(self.mood[cand, self.moods.index(mood)])
        if category is not None and 0 <= int(category) < self.n_mem:
            mem = self.emb[cand, -self.n_mem:].astype(np.float32)
            score = score + CHOICE_W * _z(mem[:, int(category)])
        sim_rej = None
        if rejected:
            sim_rej = (self._vis(cand) @ self._vis(rejected[-30:]).T).max(1)
            score = score - REJECT_W * _z(sim_rej)
        score = score - GENERIC_PEN * self.generic[cand]
        adj, live["notes"] = self._context(cand, weather, quiet, cond)
        score = score + adj

        order = np.argsort(-score)
        rank_of = np.empty(len(cand), dtype=np.int64); rank_of[order] = np.arange(len(cand))
        # pick: best-ranked, one card per name, with ~explore share drawn from further down the list
        pool = order[: min(len(order), 80)]
        far = order[30:230] if len(order) > 60 else np.array([], dtype=np.int64)
        names_taken = {_key(self.title(k, r)) for k, r in self.rows(accepted[-50:] + rejected[-50:] + seen[-100:]).items()}
        info = self.rows(cand[np.concatenate([pool, far])])
        picks, it, tries = [], iter(pool), 0
        while len(picks) < n and tries < 400:
            tries += 1
            if picks and len(far) and rng.random() < explore:      # the first card is always the best match
                j, why = int(rng.choice(far)), "explore"
            else:
                j = next(it, None)
                if j is None:
                    break
                why = "taste" if accepted else "popular"
            i = int(cand[j])
            r = info.get(i)
            if r is None:
                continue
            k = _key(self.title(i, r))
            if k in names_taken or any(len(k) >= 8 and difflib.SequenceMatcher(None, k, t).ratio() >= 0.9 for t in names_taken):
                continue
            # one landmark is often several places under different spellings ("Time Square", "Timessquare")
            if picks and haversine_km(self.lat[cand[[p for p, _ in picks]]], self.lon[cand[[p for p, _ in picks]]],
                                      self.lat[i], self.lon[i]).min() < 0.3:
                continue
            names_taken.add(k); picks.append((j, why))
        accepted_vis = self._vis(accepted[-30:]) if accepted else None
        acc_rows = self.rows(accepted[-30:]) if accepted else {}
        cards = []
        for j, why in picks:
            i = int(cand[j])
            reason = {"popular": "Popular around here", "explore": "Something different"}.get(why)
            if why == "taste":
                k = int(np.argmax(accepted_vis @ self._vis([i])[0]))
                src = accepted[-30:][k]
                reason = f"Because you liked {self.title(src, acc_rows[src])}" if src in acc_rows else "Matches your taste"
            top = max(1, math.ceil(100.0 * (rank_of[j] + 1) / len(cand)))
            cards.append(self.card(i, info[i], {"reason": reason, "top_pct": int(top), "distance_km": round(float(d[i]), 1)}))
        return {"cards": cards, "n_candidates": int(len(cand)), **live}
