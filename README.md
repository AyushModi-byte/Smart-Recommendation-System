# AuraPath

**Aesthetic-driven point-of-interest recommendation from geotagged photos.**

AuraPath learns what kind of places a person likes to photograph and recommends
places they have not visited yet. It combines visual embeddings of Flickr photos,
fuzzy "aesthetic" clusters of places, and (in v2) geography and popularity.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Live app:** https://recommendation-system-gray-seven.vercel.app (the free API sleeps when idle; the first
load can take up to a minute)

## Status

| Part | State |
|---|---|
| v1 pipeline (notebooks 01–12) | Complete. Kept as the historical record of the original experiments. |
| v2 pipeline (`training/`) | Complete. Leakage-free protocol, results below. |
| Experiments from the proposal | All built and tested: quintuplet loss, attention pooling, ICA + fuzzy C-means, weather, time, crowd, mood, collaborative filtering. See [What the proposal asked for](#what-the-proposal-asked-for). |
| Swipe app (`server/`, `web/`) | **Live.** Map plus one recommended place at a time; learns from likes. API on Render, web app on Vercel. |
| Gradio demo (`app/`) | Earlier demo with mood, live weather and crowd filters; runs locally ([docs/DEPLOY.md](docs/DEPLOY.md)). |
| Still to do | Live weather and crowd in the swipe app, user accounts, report and slides update. See [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md). |

## Results

Test set: 14,955 users, one held-out last visit each, ranked against all 177,564 places.

| Method | HR@10 | HR@20 | NDCG@20 | MRR |
|---|---|---|---|---|
| **AuraPath v2** | **0.175** | **0.232** | **0.118** | **0.092** |
| Popular within 25 km | 0.068 | 0.109 | 0.046 | 0.033 |
| Nearest place | 0.058 | 0.100 | 0.039 | 0.027 |
| Popularity | 0.010 | 0.018 | 0.007 | 0.005 |

- 2.6 times the strongest baseline. Precision@10 and MAP are far above the proposal's target of
  23% over popularity.
- With a stated destination (new-area trips, 30 km radius): HR@10 0.456 against 0.424 for popularity.
- Without one, trips to a new area are not predictable (HR@10 0.005).
- MAE of visit probability is 52% below popularity (the proposal aimed for 60%).
- About 9% of the score comes from the visual embeddings and aesthetic clusters; geography and
  popularity carry the rest.

Also tested and **not** adopted, because none improved ranking: collaborative filtering, time and
weather context, crowd level, trained attention pooling, retrained quintuplet embeddings, and a
re-derived clustering. Every experiment, with numbers, is in [docs/RESULTS.md](docs/RESULTS.md).

### What the proposal asked for

| Proposal item | Outcome |
|---|---|
| Visual encoder: ResNet50, quintuplet loss, self-attention over photos | Built. Retrained quintuplet embeddings retrieve unseen places better (Recall@10 0.219 vs 0.106 for v1) but do not improve ranking; learned attention stays close to a plain mean. |
| Clustering: ICA picks the number of clusters, fuzzy C-means | Built. Four clusters are best supported; ICA reaches the same objective as k-means++ initialisation. |
| Fusion of visual, latent and context features (weather, temperature, traffic, mood) | Built. Weather, time and crowd add nothing measurable (places average under three visits). Mood has no ground truth, so it is a user-chosen filter. |
| Precision and MAP more than 23% above popularity | **Met:** +155% Precision@10 and +178% MAP even over the strongest baseline. |
| 60% lower MAE | **Not met:** 52%. |
| Recommendations in a new city | **Only with a stated destination** (HR@10 0.456 vs 0.424). The app asks for the area. |
| Real-time mood, weather and traffic filtering | Mood in the swipe app; weather and crowd in the Gradio demo, not yet in the swipe app. |
| "Social influence clusters" of users | Not built. |

> **About the v1 numbers.** The metrics originally reported for v1 (AUC-ROC 0.998,
> Hit Rate@20 37.6%) came from an evaluation that leaked test information and are
> not a valid measure of the model. See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).

## How it works

1. **Visual encoder.** A pre-trained ResNet50 gives a 2048-d embedding per photo.
   A small projector trained with a quintuplet loss maps it to 256-d.
2. **Places.** Photos are grouped into places by DBSCAN on coordinates, and oversized
   groups are split by density peaks (209,342 places from 3.74M photos, USA and Europe, YFCC subset).
3. **Aesthetic clusters.** Fuzzy C-Means over POI embeddings gives each place a
   soft membership in six clusters (for example 60% historic, 40% scenic).
4. **Ranking model.** A two-tower network with an interaction layer scores
   (user, place) pairs. v2 adds distance, density and popularity features and is
   trained with a sampled-softmax ranking loss.

| Cluster | Label | POIs |
|---|---|---|
| 0 | Live Events & Recreation | 21,983 |
| 1 | Urban & Street Architecture | 29,928 |
| 2 | Historic & Rural Heritage | 22,883 |
| 3 | Scenic Nature & Waterscapes | 28,920 |
| 4 | Nightlife & Social Scenes | 33,361 |
| 5 | Green Nature & Wildlife | 32,770 |

## Repository layout

```
notebooks/   01–12: the original v1 experiments, with their outputs
training/    v2 pipeline as runnable scripts (prepare, train, evaluate, export)
app/         the Gradio demo and the numpy-only inference both apps use
server/      API for the swipe app (FastAPI, numpy)
web/         the swipe app: map plus one recommended place at a time (React, MapLibre)
scripts/     server sync, bundle pack/download, laptop tunnel
render.yaml  Render blueprint for the swipe app's API
docs/        data manifest, known issues, server workflow
legacy/      the first demo attempt (v1 model), kept for reference
brain/       data and model artifacts (not in git, see docs/DATA.md)
runs/        one folder per experiment: config, history, test results
```

## Quick start (v2 pipeline)

```bash
git clone https://github.com/AyushModi-byte/Smart-Recommendation-System.git
cd Smart-Recommendation-System
pip install -r requirements.txt

# 1. split oversized places, then build the time-based split with train-only embeddings
python training/00a_split_places.py --brain brain
python training/00b_prepare_all_photos.py --brain brain --out data_v4 --places _v2

# 2. train (a 2-minute sanity run first, then the real one)
python training/01_train.py --data data_v4 --name smoke --smoke
python training/01_train.py --data data_v4 --name p1_split --feat-v 1

# 3. final test-set evaluation against popularity / nearest / geo-popularity baselines
python training/02_evaluate.py --data data_v4 --run p1_split

# 4. optional: destination mode, content ablation, MAE benchmark, demo bundle
python training/03_destination_eval.py --data data_v4 --run p1_split --radius 30
python training/04_ablate_content.py   --data data_v4 --run p1_split
python training/13_mae_benchmark.py    --data data_v4 --run p1_split
python training/06_export_bundle.py    --data data_v4 --run p1_split --brain brain --out bundle --places _v2
```

Scripts `07`-`12` and `01b` are the experiments reported in [docs/RESULTS.md](docs/RESULTS.md)
(collaborative filtering, context, mood, attention pooling, clustering, quintuplet retraining).

The data in `brain/` is not distributed with the repository. See
[docs/DATA.md](docs/DATA.md) for what each file is and which ones each step needs.

## Swipe app

**Live:** https://recommendation-system-gray-seven.vercel.app

A map with one recommended place at a time: like it (and post about it), skip it or save it, and the next
cards adapt. Pick any of 3,464 towns, an area of 5–60 km, a scenery type, a mood or "hidden gems".

- **Learns from likes.** A new user has no photos, so taste is the mean embedding of the liked places. Tested
  offline on 6,000 users: liked places work as well as the user's own photos (HR@10 0.177 vs 0.175), and
  three likes already put the model 60% ahead of the best baseline.
- **Cards.** Name, town, tags, best months, visitor count and up to six photos per place, one per photographer.
  Photos are ranked from the stored ResNet50 features: how typical of the place they look, minus how much
  people or screens fill the frame (`training/17`–`18`). Dead Flickr links are skipped.
- **Stack.** FastAPI + numpy on Render (free, about 215 MB of memory, under 0.2 s per request), React +
  MapLibre on Vercel. The data bundle (103 MB) is a GitHub release asset. A laptop + Cloudflare Tunnel
  backup serves the whole app from one command.
- **Not yet.** Likes, saves and posts stay in the browser; accounts are next ([docs/NEXT_STEPS.md](docs/NEXT_STEPS.md)).

Setup, results and deployment: [docs/SWIPE_APP.md](docs/SWIPE_APP.md).

## Evaluation protocol (v2)

- Each user's visits are ordered by time. The last visit is the test target, the
  one before it is validation, everything earlier is training.
- User and place embeddings are built from training photos only.
- The model ranks the target against **all** places, with no candidate pool.
- Every result is reported next to three baselines: popularity, nearest place,
  and most popular within 25 km.

## Documentation

- [docs/RESULTS.md](docs/RESULTS.md): every result and experiment, with numbers
- [docs/DATA.md](docs/DATA.md): data manifest and what depends on what
- [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md): problems found in v1 and their status
- [docs/DGX_WORKFLOW.md](docs/DGX_WORKFLOW.md): working between the laptop and the GPU server
- [docs/DEPLOY.md](docs/DEPLOY.md): running the Gradio demo locally and deploying it
- [docs/SWIPE_APP.md](docs/SWIPE_APP.md): the swipe app, its offline check and deployment
- [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md): what is left and future directions

## Authors

**Ayush Modi** and **Anurag Yadav**, B.Tech Computer Science.

```
@misc{aurapath2026,
  title  = {AuraPath: Aesthetic-Driven POI Recommendation from Geotagged Photos},
  author = {Modi, Ayush and Yadav, Anurag},
  year   = {2026}
}
```

Photos come from the YFCC100M dataset (Flickr, Creative Commons licences).
Released under the [MIT License](LICENSE).
