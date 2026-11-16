# Known issues in the v1 pipeline

Found in a full review of notebooks 01–12. "Fixed in v2" means the scripts in
`training/` avoid the problem; the notebooks themselves are left unchanged as a record.

## Evaluation (notebooks 10, 11)

| # | Issue | Status |
|---|---|---|
| 1 | **AUC-ROC 0.998 is an artifact.** POIs outside the 1,000-candidate pool get a score of -999 and the test POIs are always inside the pool, so a random scorer would reach about 0.997. The model's AUC inside the pool works out to about 0.63. | Fixed in v2 (full ranking, no pool) |
| 2 | **Test POIs are added to the candidate pool**, so the model ranks about 1,004 places with the answers guaranteed present, not 169,845. | Fixed in v2 |
| 3 | **No train/test split.** Notebook 10 trains on every user–POI pair; the "test" POIs are chosen at random afterwards. User embeddings also average the photos taken at the test POIs. | Fixed in v2 (time-based split, train-only embeddings) |
| 4 | **Validation accuracy 85.8% is near trivial**: 80% of pairs are negatives. | Fixed in v2 (ranking metrics only) |
| 5 | No baselines, 500 users, no random seed. | Fixed in v2 |

## Pipeline

| # | Issue | Status |
|---|---|---|
| 6 | **Notebook 07 "self-attention filtering" uses untrained networks.** `CrossModalProjector` and `SelfAttentionScorer` are randomly initialised and never trained. The effective filter is a keyword list with substring matching (`art` matches "party"). | Replaced: v2 uses all photos, and `training/01b_train_attn.py` trains attention pooling end to end. The learned weights stay near uniform and ranking does not improve (NDCG@20 0.1198 vs 0.1184), so the final model keeps mean pooling |
| 7 | The filter drops 80% of photos (3.74M to 767k). 81% of remaining POIs have 3 or fewer photos. | Fixed: v2 uses all photos |
| 8 | The v1 model uses no geography and no collaborative signal. | Geography and popularity added in v2. Collaborative filtering tested twice (`v8` features, item-item blend): works alone (HR@10 0.083) but adds no significant gain on top of the model, so it is not included |
| 9 | **DBSCAN chaining (notebook 05).** One POI holds 99,240 photos; 109 POIs hold 18.7% of all photos. `training/05_poi_diag.py` reports this. | Fixed: `training/00a_split_places.py` splits 828 oversized places into 8,547 (max radius 250 m, largest place 0.43% of photos). Results in `runs/p1_split` |
| 10 | **Quintuplet training (notebook 06).** Random negatives (the semi-hard mining function is never called), margin 0.93 on unit vectors, train/val loss 1.30 vs 1.50. On held-out places the resulting projector retrieves same-place photos worse than raw ResNet features (0.106 vs 0.195). | Retrained with the four similarity levels and in-batch semi-hard negatives (`training/12_quintuplet_v2.py`): 0.219. Ranking does not improve (NDCG@20 0.1155 vs 0.1184), so the final model keeps the v1 embeddings; the new ones are saved as `quintuplet_embeddings_v3.npy` |
| 11 | **Notebook 08 "ICA" is a grid search.** All 16 values of K are evaluated before the loop starts. Silhouette is about 0.11; mean top membership is 0.63. Cluster labels were assigned by hand from 20 photos each. | Redone (`training/11_cluster_ica_fcm.py`): ICA now optimises the centres (no better than k-means++ initialisation), K = 4 is the best-supported and most stable choice and K = 6 is unstable, clusters are described by characteristic tags. The new memberships lower ranking accuracy, so the final model keeps the v1 memberships |
| 12 | Notebook 04 passes URL-encoded tags to the sentence encoder without decoding. | Fixed for new work: tags are decoded and encoded per place (`training/10_mood_profile.py`), used by the mood profiles and the new clustering. The v1 context embeddings are unchanged |

## Reproducibility

| # | Issue | Status |
|---|---|---|
| 13 | Notebooks hard-code paths from the old server (`/home/23dcs510/...`) and `CUDA_VISIBLE_DEVICES=2`, and mix working directories. They cannot be re-run top to bottom as they are. | By design: notebooks are a record, `training/` is the runnable pipeline |
| 14 | Notebook 05 shows 166,850 POIs in one cell and 201,623 in the next (stale output). 201,623 is the number in the saved data. | Documented |
| 15 | **The visual signal is a small part of the result.** A content-only v2 model scores NDCG@20 of about 0.003 on validation, and damaging the content inputs of the final model lowers NDCG@20 by about 9%. Better embeddings, attention pooling and new clusters did not change this. Geography and popularity carry the ranking. | Confirmed; this is a property of the data, not a bug |
