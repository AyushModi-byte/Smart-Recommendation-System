# Swipe app

A map on the left, one recommended place at a time on the right. Like a place and the next cards adapt to your
taste (and you can post about it); skip it and similar places drop down the list. Save puts it aside without
teaching the model anything.

```
web/      React + MapLibre front end (Vercel)          -> calls /api/*
server/   FastAPI + numpy ranking engine (Render)      -> reads bundle_swipe/
training/14_swipe_check.py   offline test of the idea
training/15_place_details.py photographers, busiest months, tags per place (laptop, reads the raw CSV)
training/17_photo_scores.py  per-photo typicality / people / junk scores from the stored ResNet50 features (DGX)
training/18_place_photos.py  picks up to 6 good photos per place, one per photographer (laptop)
training/16_export_swipe.py  builds bundle_swipe/ (core.npz, places.db, weights.npz, meta.json)
training/towns.py            the "Rome, Lazio, Italy" line (GeoNames)
```

## Does "learn from likes" work? (offline check)

In the app a new user has no photos, so their taste is the mean embedding of the places they liked, and their
liked places are the location history. `14_swipe_check.py` tests exactly that on the usual test users, no
retraining (6,000 users, final model `p1_split`, HR@10 / NDCG@20):

| User's input | AuraPath | Popular within 25 km | Nearest place |
|---|---|---|---|
| Own photos (the normal model) | 0.175 / 0.118 | 0.066 / 0.044 | 0.059 / 0.039 |
| All visited places instead of photos | 0.177 / 0.121 | | |
| Only the last 1 place | 0.145 / 0.099 | 0.117 / 0.075 | 0.112 / 0.071 |
| Only the last 3 places | 0.156 / 0.108 | 0.097 / 0.063 | 0.091 / 0.057 |
| Only the last 5 places | 0.164 / 0.114 | 0.085 / 0.055 | 0.075 / 0.050 |
| Only the last 10 places | 0.171 / 0.117 | 0.073 / 0.047 | 0.065 / 0.043 |

Places work as well as photos, and three likes already put the model 60% ahead of the best baseline. (Baselines
get *better* with fewer places because the most recent place is the best guess of where someone is.)

## How a card is chosen

Places within the chosen area (5–60 km), at least two visits, not already shown:

`score = z(log popularity) + 2 · z(model)` once the user has liked something (the blend tuned for destination
mode), plus the chosen mood or scenery, minus similarity to skipped places. The first card is always the best
match; later cards are drawn from further down the list about one time in five ("Something different").
Duplicates of one landmark ("Time Square" / "Timessquare", places 300 m apart) are shown once. Places whose only
name is their city are titled from their tags and only used where named places run out.

"Top 3%" is the card's rank among places in the area, not a probability. Descriptions are written from the
place's scenery cluster, strongest mood, town and photo counts; nothing is generated or invented.

## Run locally (CMD)

```
.venv\Scripts\python.exe training\16_export_swipe.py
.venv\Scripts\python.exe -m uvicorn server.main:app --port 8000
cd web
npm install
npm run dev
```
Open http://localhost:5173. The API uses about 215 MB of memory at peak and answers in under 0.2 s.

## Deploy (free, no card)

1. **Bundle.** `.venv\Scripts\python.exe scripts\fetch_swipe_bundle.py --pack` writes `bundle_swipe.zip` (98 MB).
   Upload it as an asset of a GitHub release on the repo and copy the asset's download URL.
2. **API on Render.** New > Blueprint, pick the repo; `render.yaml` sets everything. Fill in `SWIPE_BUNDLE_URL`
   (step 1) and, after step 3, `ALLOWED_ORIGINS` (the Vercel address). The free plan sleeps after 15 minutes idle;
   the first request then takes up to a minute and the app says so.
3. **Web app on Vercel.** New project from the repo, root directory `web`, framework Vite, environment variable
   `VITE_API_URL=https://<your-render-service>.onrender.com`.
4. **Presentation backup on the laptop.** `cd web`, `set VITE_API_URL=same-origin`, `npm run build`, then
   `scripts\run_local_tunnel.cmd`. It starts the API (which then also serves the page) and prints a public
   `https://….trycloudflare.com` address that works while the window is open. `cloudflared.exe` is in `D:\dev\tools`.

## Limits

- Likes, saves and posts live in the browser (localStorage) until accounts are added; they do not sync between
  devices and other people cannot see posts yet.
- Feedback events go to the API's `FEEDBACK_LOG` file when that is set; Render's disk is wiped on every deploy, so
  measuring like rates on Render needs a database (planned: Supabase).
- Photos are linked from Flickr, not copied. Each card has up to 6; a deleted one is skipped, and the placeholder
  shows only when none load. Photos where people or screens fill the frame are ranked last.
- Liked and saved places are refreshed from the API on every load, so they pick up data updates.
- The data is 2004–2014 Flickr photos: some places have closed or changed.
