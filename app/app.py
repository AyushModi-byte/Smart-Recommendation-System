"""AuraPath demo: place recommendations from geotagged photos.

Numpy-only inference (no torch) so it runs on a free CPU host.
Run locally from the repo root:   python app/app.py
The bundle (weights.npz, pois.npz, users.npz, meta.json) is read from
$AURAPATH_BUNDLE, ./bundle next to this file, or ../bundle.
"""
import datetime as dt
import html
import json
import os
import urllib.parse
import urllib.request

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from aurapath_np import AuraNP, geo_feats, haversine_km

HERE = os.path.dirname(os.path.abspath(__file__))
CITIES = {
    "London": (51.5074, -0.1278), "Paris": (48.8566, 2.3522), "Rome": (41.9028, 12.4964),
    "Barcelona": (41.3851, 2.1734), "Amsterdam": (52.3676, 4.9041), "Berlin": (52.52, 13.405),
    "Prague": (50.0755, 14.4378), "Vienna": (48.2082, 16.3738), "Venice": (45.4408, 12.3155),
    "Florence": (43.7696, 11.2558), "Madrid": (40.4168, -3.7038), "Lisbon": (38.7223, -9.1393),
    "Dublin": (53.3498, -6.2603), "Edinburgh": (55.9533, -3.1883), "Copenhagen": (55.6761, 12.5683),
    "Stockholm": (59.3293, 18.0686), "Budapest": (47.4979, 19.0402), "Istanbul": (41.0082, 28.9784),
    "New York": (40.7128, -74.006), "San Francisco": (37.7749, -122.4194), "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298), "Washington DC": (38.9072, -77.0369), "Boston": (42.3601, -71.0589),
    "Seattle": (47.6062, -122.3321), "Portland": (45.5152, -122.6784), "New Orleans": (29.9511, -90.0715),
    "Las Vegas": (36.1699, -115.1398), "Miami": (25.7617, -80.1918), "Austin": (30.2672, -97.7431),
    "Denver": (39.7392, -104.9903), "Philadelphia": (39.9526, -75.1652),
}
CUSTOM = "Custom coordinates"
MODE_MOOD, MODE_TRAV, MODE_HOME = "Pick a mood", "Sample traveller, at this destination", "Sample traveller, near places they know"
ANY, LIVE = "Any", "Live"
TOD = ["Night", "Morning", "Afternoon", "Evening"]
WEATHER = ["Dry", "Rain"]
TEMP = ["Cold (under 5 °C)", "Cool (5–15 °C)", "Mild (15–25 °C)", "Hot (over 25 °C)"]


def _bundle_dir():
    for d in (os.environ.get("AURAPATH_BUNDLE"), os.path.join(HERE, "bundle"), os.path.join(HERE, "..", "bundle")):
        if d and os.path.exists(os.path.join(d, "pois.npz")):
            return d
    raise FileNotFoundError("bundle not found: set AURAPATH_BUNDLE or put the bundle folder next to app.py")


B = _bundle_dir()
with open(os.path.join(B, "meta.json"), encoding="utf-8") as _f:
    META = json.load(_f)
NET = AuraNP(np.load(os.path.join(B, "weights.npz")))
_p = np.load(os.path.join(B, "pois.npz"))
EMB = _p["emb"].astype(np.float32)
LAT, LON, CNT = _p["lat"].astype(np.float64), _p["lon"].astype(np.float64), _p["cnt"].astype(np.float32)
CLUSTER, VALID = _p["cluster"].astype(np.int64), ~_p["no_clean"]
NAMES = _p["name"].tolist()
URLS = [u.replace("http://", "https://", 1) for u in _p["url"].tolist()]
CLUSTERS = META.get("clusters") or ["Live Events & Recreation", "Urban & Street Architecture", "Historic & Rural Heritage",
                                    "Scenic Nature & Waterscapes", "Nightlife & Social Scenes", "Green Nature & Wildlife"]
MEM = EMB[:, -len(CLUSTERS):]                      # fuzzy cluster memberships
MOODS = META.get("moods") or []
MOOD = _p["mood"].astype(np.float32) if "mood" in _p.files and MOODS else None
CTX = {k: _p[k].astype(np.float32) for k in _p.files if k.startswith("aff_") or k == "crowd"}
del _p
PT = NET.tower(EMB, "poi_tower")                   # (P, h) once at start-up
LOGPOP = np.log1p(CNT)
POP_NORM = float(META["pop_norm"])
GEO_M = int(META["config"].get("geo_m", 30))
_u = np.load(os.path.join(B, "users.npz"))
U_EMB, U_HIST, U_PTR = _u["user_emb"].astype(np.float32), _u["hist"].astype(np.int64), _u["ptr"]
del _u
TRAVELLERS = [f"Traveller {i + 1:03d}" for i in range(len(U_EMB))]


def _z(x):
    return (x - x.mean()) / max(float(x.std()), 1e-6)


def _history(ui):
    return U_HIST[U_PTR[ui]:U_PTR[ui + 1]]


def _model_scores(ui, cand, chunk=32768):
    ref = _history(ui)[::-1][:GEO_M]               # most recent first
    ut = NET.tower(U_EMB[ui][None], "user_tower")
    return np.concatenate([NET.head(ut, PT[cand[s:s + chunk]], geo_feats(ref, cand[s:s + chunk], LAT, LON, CNT, POP_NORM))
                           for s in range(0, len(cand), chunk)])


def _taste_summary(ui):
    hist = _history(ui)
    share = np.bincount(CLUSTER[hist], minlength=len(CLUSTERS)) / max(len(hist), 1)
    parts = ", ".join(f"{CLUSTERS[i]} ({share[i]:.0%})" for i in np.argsort(-share)[:3] if share[i] > 0)
    return f"**{TRAVELLERS[ui]}** has photographed {len(hist)} places. Mostly: {parts}."


def live_conditions(lat, lon):
    """Current weather and local time at a location (Open-Meteo, no key). Returns a dict or None."""
    url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(
        {"latitude": round(float(lat), 3), "longitude": round(float(lon), 3),
         "current": "temperature_2m,precipitation,is_day", "timezone": "auto"})
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            cur = json.loads(r.read().decode("utf-8"))["current"]
        t = dt.datetime.fromisoformat(cur["time"])
        return {"tod": t.hour // 6, "weekend": int(t.weekday() >= 5), "season": ((t.month % 12) // 3), "daylight": int(cur["is_day"]),
                "rain": int(float(cur["precipitation"]) > 0.0), "temp": int(np.digitize(float(cur["temperature_2m"]), [5.0, 15.0, 25.0])),
                "text": f"{cur['temperature_2m']:.0f} °C, {'raining' if float(cur['precipitation']) > 0 else 'dry'}, "
                        f"{TOD[t.hour // 6].lower()} ({t:%H:%M} local time)"}
    except Exception:
        return None


def _context(centre, use_live, tod, weather, temp):
    """-> (bucket dict, description). Manual choices override the live values."""
    cond, text = {}, []
    live = live_conditions(*centre) if use_live else None
    if live:
        cond = {k: live[k] for k in ("tod", "weekend", "season", "daylight", "rain", "temp")}
        text.append("Live conditions: " + live["text"])
    elif use_live:
        text.append("Live conditions unavailable right now")
    if tod in TOD:
        cond["tod"] = TOD.index(tod); cond.pop("daylight", None)
    if weather in WEATHER:
        cond["rain"] = WEATHER.index(weather)
    if temp in TEMP:
        cond["temp"] = TEMP.index(temp)
    manual = [x for x in (tod, weather, temp) if x != LIVE]
    if manual:
        text.append("Set by you: " + ", ".join(manual))
    return cond, ". ".join(text)


def _context_score(cand, cond):
    sc = np.zeros(len(cand), dtype=np.float32)
    for k, b in cond.items():
        tab = CTX.get(f"aff_{k}")
        if tab is not None:
            sc += tab[cand, int(b)]
    return sc


def recommend(mode, city, lat, lon, radius, mood, scenery, traveller, weight, use_live, tod, weather, temp, ctx_weight, avoid_crowds, top_k):
    top_k = int(top_k)
    ui = TRAVELLERS.index(traveller) if traveller in TRAVELLERS else 0
    notes = []
    if mode == MODE_HOME:
        hist = _history(ui)
        keep = VALID.copy()
        keep[hist] = False
        # the model is strongly local: skip places > 500 km from everything in the history (keeps CPU time low)
        ref = hist[::-1][:GEO_M]
        near = haversine_km(LAT[:, None], LON[:, None], LAT[ref][None, :], LON[ref][None, :]).min(1) <= 500.0
        if (keep & near).sum() >= 1000:
            keep &= near
        cand = np.nonzero(keep)[0]
        score = _model_scores(ui, cand)
        centre = (float(LAT[hist[-1]]), float(LON[hist[-1]]))
        notes.append(_taste_summary(ui) + " Ranked by the model over all places, excluding ones already visited.")
    else:
        if city != CUSTOM:
            lat, lon = CITIES[city]
        if lat is None or lon is None:
            return "<p>Enter a latitude and longitude.</p>", None, ""
        centre = (float(lat), float(lon))
        d = haversine_km(LAT, LON, centre[0], centre[1])
        keep = VALID & (d <= float(radius))
        if mode == MODE_TRAV:
            keep[_history(ui)] = False
        else:
            keep &= CNT >= 2                       # mood mode: skip places only one person has been to
        cand = np.nonzero(keep)[0]
        if len(cand) < 3:
            return f"<p>Only {len(cand)} known places within {radius:.0f} km. Try a larger radius.</p>", None, ""
        score = _z(LOGPOP[cand])
        if mode == MODE_TRAV:
            score = score + float(weight) * _z(_model_scores(ui, cand))
            notes.append(_taste_summary(ui) + f" Ranking {len(cand):,} places within {radius:.0f} km.")
        else:
            what = []
            if MOOD is not None and mood in MOODS:
                score = score + float(weight) * _z(MOOD[cand, MOODS.index(mood)]); what.append(f"the mood **{mood}**")
            if scenery in CLUSTERS:
                score = score + float(weight) * _z(MEM[cand, CLUSTERS.index(scenery)]); what.append(f"**{scenery}**")
            notes.append(f"Ranking {len(cand):,} places within {radius:.0f} km by popularity" + (" and " + " and ".join(what) if what else "") + ".")
        cond, text = _context(centre, use_live, tod, weather, temp)
        if cond and float(ctx_weight) > 0:
            score = score + float(ctx_weight) * _z(_context_score(cand, cond))
        if text:
            notes.append(text + ".")
        if avoid_crowds and "crowd" in CTX:
            slot = cond.get("tod", 2) * 2 + cond.get("weekend", 0)
            score = score - 0.75 * _z(CTX["crowd"][cand, slot])
            notes.append("Favouring quieter places for this time slot.")
    picks, seen = [], set()                        # a big landmark is several adjacent places: show each name once
    for i in np.argsort(-score):
        if NAMES[cand[i]] not in seen:
            seen.add(NAMES[cand[i]]); picks.append(cand[i])
            if len(picks) == top_k:
                break
    picks = np.array(picks, dtype=np.int64)
    dist = haversine_km(LAT[picks], LON[picks], centre[0], centre[1])
    return _cards(picks, dist, mode), _map(picks, centre, mode), " ".join(notes)


def _photo_page(url):
    try:
        return "https://www.flickr.com/photo.gne?id=" + url.rsplit("/", 1)[1].split("_")[0]
    except Exception:
        return url


def _cards(picks, dist, mode):
    cells = []
    for rank, (p, dk) in enumerate(zip(picks, dist), 1):
        name = html.escape(NAMES[p])
        where = "from your last place" if mode == MODE_HOME else "from centre"
        tag = CLUSTERS[CLUSTER[p]]
        if MOOD is not None:
            tag += " · " + MOODS[int(MOOD[p].argmax())]
        cells.append(
            f'<div class="ap-card"><a href="{html.escape(_photo_page(URLS[p]))}" target="_blank" rel="noopener">'
            f'<img src="{html.escape(URLS[p])}" loading="lazy" alt="{name}" '
            f'onerror="this.style.visibility=\'hidden\'"></a>'
            f'<div class="ap-body"><div class="ap-name">{rank}. {name}</div>'
            f'<div class="ap-meta">{html.escape(tag)}</div>'
            f'<div class="ap-meta">{dk:.1f} km {where} · {int(CNT[p])} visitors</div></div></div>')
    return '<div class="ap-grid">' + "".join(cells) + "</div>"


def _map(picks, centre, mode):
    fig = go.Figure(go.Scattermap(
        lat=LAT[picks], lon=LON[picks], mode="markers+text", marker=dict(size=13, color="#d1495b"),
        text=[str(i) for i in range(1, len(picks) + 1)], textposition="top center",
        hovertext=[f"{i}. {NAMES[p]}" for i, p in enumerate(picks, 1)], hoverinfo="text"))
    c_lat, c_lon = (float(LAT[picks].mean()), float(LON[picks].mean())) if mode == MODE_HOME else centre
    span = max(float(np.ptp(LAT[picks])), float(np.ptp(LON[picks])), 0.02)
    zoom = float(np.clip(8.5 - np.log2(span / 0.35), 2, 14))
    fig.update_layout(map=dict(style="open-street-map", center=dict(lat=c_lat, lon=c_lon), zoom=zoom),
                      margin=dict(l=0, r=0, t=0, b=0), height=430, showlegend=False)
    return fig


def _about():
    t = META.get("test_results", {})
    k = next((x for x in t if x.startswith("all test")), None)
    if not k:
        return "Model details unavailable."
    rows = "".join(f"| {name} | {m['HR@10(=Recall@10)']:.3f} | {m['NDCG@20']:.3f} |\n" for name, m in t[k].items() if isinstance(m, dict) and m)
    return (
        "AuraPath learns from 3.7 million geotagged Flickr photos (YFCC100M, USA and Europe) which places "
        "people photograph, and ranks places a person has not been to yet.\n\n"
        f"**Offline test** ({t[k]['n']:,} users, last visit held out, ranked against all {META['n_pois']:,} places):\n\n"
        "| Method | HR@10 | NDCG@20 |\n|---|---|---|\n" + rows +
        "\n**What is and is not evaluated.** The sample-traveller modes use the trained model above. "
        "Mood, scenery, live weather, time of day and crowd level are explicit filters: in offline tests the time, weather "
        "and crowd signals did not improve ranking accuracy, and mood has no ground truth to test against. "
        "Mood comes from each place's photo tags; \"crowds\" is the number of photographers at that time slot, a stand-in for traffic. "
        "Place names are generated from photo tags and can be rough. "
        "Photos are Creative Commons images hosted by Flickr; click a photo for its source page and licence. "
        "Live weather is from Open-Meteo."
    )


CSS = """
.ap-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:14px}
.ap-card{border:1px solid var(--border-color-primary);border-radius:10px;overflow:hidden;background:var(--background-fill-secondary)}
.ap-card img{width:100%;height:150px;object-fit:cover;display:block;background:var(--neutral-200)}
.ap-body{padding:8px 10px}
.ap-name{font-weight:600;font-size:14px;line-height:1.25;overflow-wrap:anywhere}
.ap-meta{font-size:12px;opacity:.75;margin-top:2px}
"""

with gr.Blocks(title="AuraPath") as demo:
    gr.Markdown("# AuraPath\nPlace recommendations learned from geotagged photos. Choose where you are going and what you are in the mood for.")
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            mode = gr.Radio([MODE_MOOD, MODE_TRAV, MODE_HOME], value=MODE_MOOD, label="How should we personalise?")
            city = gr.Dropdown(list(CITIES) + [CUSTOM], value="Paris", label="Destination")
            with gr.Row(visible=False) as custom_row:
                lat = gr.Number(label="Latitude", value=48.8566)
                lon = gr.Number(label="Longitude", value=2.3522)
            radius = gr.Slider(2, 100, value=30, step=1, label="Search radius (km)")
            mood = gr.Dropdown(([ANY] + MOODS) if MOODS else [ANY], value=MOODS[0] if MOODS else ANY, label="Mood")
            scenery = gr.Dropdown([ANY] + CLUSTERS, value=ANY, label="Scenery")
            traveller = gr.Dropdown(TRAVELLERS, value=TRAVELLERS[0], label="Sample traveller", visible=False)
            weight = gr.Slider(0, 4, value=1.5, step=0.25, label="Personalisation strength (0 = popularity only)")
            with gr.Accordion("Right now at the destination", open=True) as ctx_box:
                use_live = gr.Checkbox(value=True, label="Use live weather and local time")
                tod = gr.Dropdown([LIVE] + TOD, value=LIVE, label="Time of day")
                weather = gr.Dropdown([LIVE] + WEATHER, value=LIVE, label="Weather")
                temp = gr.Dropdown([LIVE] + TEMP, value=LIVE, label="Temperature")
                ctx_weight = gr.Slider(0, 2, value=0.75, step=0.25, label="How much conditions matter")
                avoid = gr.Checkbox(value=False, label="Avoid crowds")
            top_k = gr.Slider(4, 24, value=12, step=1, label="How many places")
            go_btn = gr.Button("Recommend", variant="primary")
        with gr.Column(scale=2):
            note = gr.Markdown()
            cards = gr.HTML()
            fmap = gr.Plot(label="Map")
    with gr.Accordion("About this demo", open=False):
        gr.Markdown(_about())

    def _toggle(m, c):
        dest = m != MODE_HOME
        return (gr.update(visible=dest), gr.update(visible=dest and c == CUSTOM), gr.update(visible=dest),
                gr.update(visible=m == MODE_MOOD), gr.update(visible=m == MODE_MOOD), gr.update(visible=m != MODE_MOOD),
                gr.update(visible=dest), gr.update(visible=dest))

    for comp in (mode, city):
        comp.change(_toggle, [mode, city], [city, custom_row, radius, mood, scenery, traveller, weight, ctx_box])
    inputs = [mode, city, lat, lon, radius, mood, scenery, traveller, weight, use_live, tod, weather, temp, ctx_weight, avoid, top_k]
    go_btn.click(recommend, inputs, [cards, fmap, note])
    demo.load(recommend, inputs, [cards, fmap, note])

if __name__ == "__main__":
    demo.launch(css=CSS)
