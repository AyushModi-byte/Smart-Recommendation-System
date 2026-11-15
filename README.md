# AuraPath

**Aesthetic-driven point-of-interest recommendation from geotagged photos.**

AuraPath learns what kind of places a person likes to photograph and recommends
places they have not visited yet. It combines visual embeddings of Flickr photos,
fuzzy "aesthetic" clusters of places, and (in v2) geography and popularity.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Status

| Part | State |
|---|---|
| v1 pipeline (notebooks 01–12) | Complete. Kept as the historical record of the original experiments. |
| v2 pipeline (`training/`) | Complete. Leakage-free protocol, results below. |
| Web demo (`app/`) | Gradio app, runs locally. Deployment steps in [docs/DEPLOY.md](docs/DEPLOY.md). |

## Results (v2, run `v7`)

Test set: 13,236 users, one held-out last visit each, ranked against all 169,845 places.

| Method | HR@10 | HR@20 | NDCG@10 | NDCG@20 | MRR |
|---|---|---|---|---|---|
| **AuraPath v2** | **0.225** | **0.284** | **0.141** | **0.156** | **0.125** |
| Popular within 25 km | 0.129 | 0.165 | 0.083 | 0.092 | 0.075 |
| Popularity | 0.074 | 0.098 | 0.043 | 0.049 | 0.038 |
| Nearest place | 0.058 | 0.093 | 0.029 | 0.038 | 0.027 |

- HR@10 is 74% above the strongest baseline and three times global popularity.
- NDCG@20 gain over the nearest-place baseline: 0.119, 95% CI [0.113, 0.123].
- **Local visits** (within 50 km of the user's history, 69% of the test set): HR@10 0.313 against 0.188 for the best baseline.
- **Trips to a new area** (more than 50 km away, 31%): HR@10 0.032, below global popularity at 0.148. This is the main open weakness.
- 21.5% of test visits go to places with no earlier photos; these cannot be recommended and are excluded.

### When the destination is known

The 4,156 trips to a new area, ranking only places within 30 km of the destination
(about 400 candidates per trip). The blend weight was tuned on validation trips.

| Method | HR@5 | HR@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|
| **Popularity + AuraPath** | **0.594** | **0.692** | **0.777** | **0.549** |
| Popularity in area | 0.572 | 0.663 | 0.751 | 0.533 |
| Visual taste only | 0.087 | 0.142 | 0.213 | 0.089 |
| Random in area | 0.054 | 0.095 | 0.160 | 0.063 |

This assumes the user states where they are going. Popularity carries most of the
result; personalisation adds about 4% on HR@10.

### What the visual signal contributes

Same model and test users (a 4,000-user sample), with the content inputs deliberately damaged:

| Variant | HR@10 | NDCG@20 |
|---|---|---|
| Normal | 0.227 | 0.157 |
| User taste shuffled between users | 0.203 | 0.144 |
| Place content replaced by the global mean | 0.211 | 0.147 |
| Both | 0.207 | 0.145 |

The visual embeddings and aesthetic clusters account for roughly 8% of NDCG@20.
The rest comes from geography and popularity.

Per-run configs, training histories and full result tables are in `runs/`.

> **About the v1 numbers.** The metrics originally reported for v1 (AUC-ROC 0.998,
> Hit Rate@20 37.6%) came from an evaluation that leaked test information and are
> not a valid measure of the model. See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).

## How it works

1. **Visual encoder.** A pre-trained ResNet50 gives a 2048-d embedding per photo.
   A small projector trained with a quintuplet loss maps it to 256-d.
2. **Places.** Photos are grouped into POIs by DBSCAN on coordinates
   (201,623 POIs from 3.74M photos, USA and Europe, YFCC subset).
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
app/         the web demo (Gradio) and its numpy-only inference
scripts/     helpers for syncing with the GPU server
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

# 1. time-based split, train-only embeddings (uses all 3.7M photos)
python training/00b_prepare_all_photos.py --brain brain --out data_v3

# 2. train (a 2-minute sanity run first, then the real one)
python training/01_train.py --data data_v3 --name smoke --smoke
python training/01_train.py --data data_v3 --name v7 --feat-v 1

# 3. final test-set evaluation against popularity / nearest / geo-popularity baselines
python training/02_evaluate.py --data data_v3 --run v7

# 4. optional: destination-mode evaluation, content ablation, demo bundle
python training/03_destination_eval.py --data data_v3 --run v7 --radius 30
python training/04_ablate_content.py   --data data_v3 --run v7
python training/06_export_bundle.py    --data data_v3 --run v7 --brain brain --out bundle
```

The data in `brain/` is not distributed with the repository. See
[docs/DATA.md](docs/DATA.md) for what each file is and which ones each step needs.

## Evaluation protocol (v2)

- Each user's visits are ordered by time. The last visit is the test target, the
  one before it is validation, everything earlier is training.
- User and place embeddings are built from training photos only.
- The model ranks the target against **all** places, with no candidate pool.
- Every result is reported next to three baselines: popularity, nearest place,
  and most popular within 25 km.

## Documentation

- [docs/DATA.md](docs/DATA.md): data manifest and what depends on what
- [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md): problems found in v1 and their status
- [docs/DGX_WORKFLOW.md](docs/DGX_WORKFLOW.md): working between the laptop and the GPU server
- [docs/DEPLOY.md](docs/DEPLOY.md): running the demo locally and deploying it

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
