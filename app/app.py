"""AuraPath demo: place recommendations from geotagged photos.

Numpy-only inference (no torch) so it runs on a free CPU host.
Run locally from the repo root:   python app/app.py
The bundle (weights.npz, pois.npz, users.npz, meta.json) is read from
$AURAPATH_BUNDLE, ./bundle next to this file, or ../bundle.
"""
import html
import json
import os

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from aurapath_np import AuraNP, geo_feats, haversine_km

HERE = os.path.dirname(os.path.abspath(__file__))
CLUSTERS = ["Live Events & Recreation", "Urban & Street Architecture", "Historic & Rural Heritage",
            "Scenic Nature & Waterscapes", "Nightlife & Social Scenes", "Green Nature & Wildlife"]
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
del _p
MEM = EMB[:, -6:]                                  # fuzzy cluster memberships
PT = NET.tower(EMB, "poi_tower")                   # (P, h) once at start-up
LOGPOP = np.log1p(CNT)
POP_NORM = float(META["pop_norm"])
GEO_M = int(META["config"].get("geo_m", 30))
_u = np.load(os.path.join(B, "users.npz"))
U_EMB, U_HIST, U_PTR = _u["user_emb"].astype(np.float32), _u["hist"].astype(np.int64), _u["ptr"]
del _u
N_USERS = len(U_EMB)
TRAVELLERS = [f"Traveller {i + 1:03d}" for i in range(N_USERS)]


def _z(x):
    return (x - x.mean()) / max(float(x.std()), 1e-6)


def _history(ui):
    return U_HIST[U_PTR[ui]:U_PTR[ui + 1]]


def _model_scores(ui, cand, chunk=32768):
    hist = _history(ui)
    ref = hist[::-1][:GEO_M]                       # most recent first
    ut = NET.tower(U_EMB[ui][None], "user_tower")
    out = []
    for s in range(0, len(cand), chunk):
        c = cand[s:s + chunk]
        out.append(NET.head(ut, PT[c], geo_feats(ref, c, LAT, LON, CNT, POP_NORM)))
    return np.concatenate(out)


def _taste_summary(ui):
    hist = _history(ui)
    share = np.bincount(CLUSTER[hist], minlength=6) / max(len(hist), 1)
    top = np.argsort(-share)[:3]
    parts = ", ".join(f"{CLUSTERS[i]} ({share[i]:.0%})" for i in top if share[i] > 0)
    return f"**{TRAVELLERS[ui]}** has photographed {len(hist)} places. Mostly: {parts}."


def recommend(mode, city, lat, lon, radius, mood, traveller, weight, top_k):
    top_k = int(top_k)
    ui = TRAVELLERS.index(traveller) if traveller in TRAVELLERS else 0
    note = ""
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
        note = _taste_summary(ui) + " Ranked by the model over all places, excluding ones already visited."
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
        zp = _z(LOGPOP[cand])
        if mode == MODE_TRAV:
            score = zp + float(weight) * _z(_model_scores(ui, cand))
            note = _taste_summary(ui) + f" Ranking {len(cand):,} places within {radius:.0f} km."
        else:
            ci = CLUSTERS.index(mood)
            score = zp + float(weight) * _z(MEM[cand, ci])
            note = f"Ranking {len(cand):,} places within {radius:.0f} km by popularity and how strongly they match **{mood}**."
    order = np.argsort(-score)[:top_k]
    picks = cand[order]
    dist = haversine_km(LAT[picks], LON[picks], centre[0], centre[1])
    return _cards(picks, dist, mode), _map(picks, centre, mode), note


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
        cells.append(
            f'<div class="ap-card"><a href="{html.escape(_photo_page(URLS[p]))}" target="_blank" rel="noopener">'
            f'<img src="{html.escape(URLS[p])}" loading="lazy" alt="{name}" '
            f'onerror="this.style.visibility=\'hidden\'"></a>'
            f'<div class="ap-body"><div class="ap-name">{rank}. {name}</div>'
            f'<div class="ap-meta">{CLUSTERS[CLUSTER[p]]}</div>'
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
    rows = ""
    if k:
        for name, m in t[k].items():
            if isinstance(m, dict) and m:
                rows += f"| {name} | {m['HR@10(=Recall@10)']:.3f} | {m['NDCG@20']:.3f} |\n"
    return (
        "AuraPath learns from 3.7 million geotagged Flickr photos (YFCC100M, USA and Europe) which places "
        "people photograph, and ranks places a person has not been to yet.\n\n"
        f"**Offline test** ({t[k]['n']:,} users, last visit held out, ranked against all {META['n_pois']:,} places):\n\n"
        "| Method | HR@10 | NDCG@20 |\n|---|---|---|\n" + rows +
        "\nThe mood mode is a simple popularity-plus-cluster ranking and has not been evaluated offline. "
        "Place names are generated from photo tags and can be rough. "
        "Photos are Creative Commons images hosted by Flickr; click a photo for its source page and licence."
    ) if k else "Model details unavailable."


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
            mood = gr.Dropdown(CLUSTERS, value=CLUSTERS[3], label="Mood")
            traveller = gr.Dropdown(TRAVELLERS, value=TRAVELLERS[0], label="Sample traveller", visible=False)
            weight = gr.Slider(0, 4, value=2.0, step=0.25, label="Personalisation strength (0 = popularity only)")
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
                gr.update(visible=m == MODE_MOOD), gr.update(visible=m != MODE_MOOD), gr.update(visible=dest))

    for comp in (mode, city):
        comp.change(_toggle, [mode, city], [city, custom_row, radius, mood, traveller, weight])
    inputs = [mode, city, lat, lon, radius, mood, traveller, weight, top_k]
    go_btn.click(recommend, inputs, [cards, fmap, note])
    demo.load(recommend, inputs, [cards, fmap, note])

if __name__ == "__main__":
    demo.launch(css=CSS)
