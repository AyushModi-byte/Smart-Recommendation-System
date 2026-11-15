# Data manifest

Everything lives in `brain/` (about 72 GB, not in git). The old server is gone, so
**the copy on the laptop is the only copy. Do not delete anything here** and keep a
second copy on an external drive.

## Files

| File | Size | Made by | What it is |
|---|---|---|---|
| `usa_europe_geotagged.csv` | 5.1 GB | NB 02 | Photo metadata, 7.67M rows (uid, time, tags, lat/lon, URL) |
| `embeddings/embeddings_chunk_*.npy` | 53 GB | NB 03 | ResNet50 2048-d features, 768 chunks. **Cannot be rebuilt cheaply** (needs the images) |
| `ids/ids_chunk_*.npy` (+ `ids.tar.gz`) | 55 MB | NB 03 | Photo ids aligned with the chunks above |
| `faiss_id_map.npy` | 53 MB | NB 03 | Id map for a FAISS index that is no longer present |
| `context_embeddings_v2/` (+ `context_photo_ids.npy`) | 7.3 GB | NB 04 | 391-d context vectors (time, location, tag text) and their photo ids |
| `photo_poi_mapping.csv` | 133 MB | NB 05 | photoid, lon, lat, poi_id for 3.74M photos |
| `meal_projector.pth` | 13 MB | NB 06 | Trained 2048-to-256 projector |
| `quintuplet_embeddings.npy` (+ `_poi_ids.npy`) | 3.6 GB | NB 06 | 256-d embedding for each of the 3.74M photos |
| `filtered_visual_embeddings.npy`, `filtered_photo_ids.npy`, `filtered_poi_ids.npy` | 760 MB | NB 07 | The 767k photos that passed the v1 filter |
| `filtered_visual_embeddings_combined.npy` | 1.9 GB | NB 07 | Same photos, visual + context (647-d) |
| `poi_fuzzy_memberships.npy/.csv`, `poi_cluster_ids.npy`, `poi_hard_labels.npy`, `cluster_centers.npy`, `ica_fcm_k_search.png` | 30 MB | NB 08 | Six-cluster fuzzy memberships for 169,845 POIs |
| `poi_aesthetic_labels.csv`, `cluster_label_map.json` | 20 MB | NB 09 | Cluster names and per-POI labels |
| `neural_interaction_model.pth`, `user_embeddings.npy`, `user_ids.npy`, `poi_embeddings_262.npy`, `poi_ids_262.npy` | 210 MB | NB 10 | v1 model and its inputs |
| `evaluation_results.json/.png` | small | NB 11 | v1 metrics (invalid, see KNOWN_ISSUES.md) |

## What each v2 step needs

| Step | Needs from `brain/` |
|---|---|
| `training/00b_prepare_all_photos.py` | `ids/`, `quintuplet_embeddings.npy`, `quintuplet_embeddings_poi_ids.npy`, `photo_poi_mapping.csv`, `poi_fuzzy_memberships.npy`, `poi_cluster_ids.npy`, `usa_europe_geotagged.csv` |
| `training/00_prepare_data.py` | `filtered_*.npy`, `poi_fuzzy_memberships.npy`, `poi_cluster_ids.npy`, `usa_europe_geotagged.csv` |
| `training/01`–`04` | only `data_v*/prepared.npz` written by step 00 |
| `training/05_poi_diag.py` | `photo_poi_mapping.csv` |
| `training/06_export_bundle.py` | `photo_poi_mapping.csv`, `usa_europe_geotagged.csv`, a finished run |

So the server needs about 12 GB (every top-level file plus `ids/`). The 53 GB
`embeddings/` folder and the 7.3 GB context folder are only needed to retrain the
visual projector (NB 06) or redo the clustering (NB 07–08).
