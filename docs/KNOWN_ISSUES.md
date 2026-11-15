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
| 6 | **Notebook 07 "self-attention filtering" uses untrained networks.** `CrossModalProjector` and `SelfAttentionScorer` are randomly initialised and never trained. The effective filter is a keyword list with substring matching (`art` matches "party"). | Open. v2 bypasses the filter (`00b_prepare_all_photos.py`) |
| 7 | The filter drops 80% of photos (3.74M to 767k). 81% of remaining POIs have 3 or fewer photos. | Bypassed in v2 |
| 8 | The v1 model uses no geography and no collaborative signal. | Geography and popularity added in v2. Collaborative filtering still open |
| 9 | **DBSCAN chaining (notebook 05).** One POI holds 99,240 photos; 109 POIs hold 18.7% of all photos. `training/05_poi_diag.py` reports this. | Open |
| 10 | **Quintuplet training (notebook 06).** Random negatives (the semi-hard mining function is never called), margin 0.93 on unit vectors, train/val loss 1.30 vs 1.50. | Open |
| 11 | **Notebook 08 "ICA" is a grid search.** All 16 values of K are evaluated before the loop starts. Silhouette is about 0.11; mean top membership is 0.63. Cluster labels were assigned by hand from 20 photos each. | Open |
| 12 | Notebook 04 passes URL-encoded tags to the sentence encoder without decoding. | Open |

## Reproducibility

| # | Issue | Status |
|---|---|---|
| 13 | Notebooks hard-code paths from the old server (`/home/23dcs510/...`) and `CUDA_VISIBLE_DEVICES=2`, and mix working directories. They cannot be re-run top to bottom as they are. | By design: notebooks are a record, `training/` is the runnable pipeline |
| 14 | Notebook 05 shows 166,850 POIs in one cell and 201,623 in the next (stale output). 201,623 is the number in the saved data. | Documented |
| 15 | A content-only v2 model (no geography) scores NDCG@20 of about 0.003 on validation: the visual embeddings alone do not rank places. Nearly all of the v2 gain comes from geography and popularity features. | Open. Run `training/04_ablate_content.py` to quantify |
