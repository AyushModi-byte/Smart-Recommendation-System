# Foursquare NYC next-POI benchmark

A separate module that evaluates the AuraPath v2 ranker on the Foursquare NYC split used by GETNext, so that numbers
can be put next to published next-POI results. The YFCC pipeline (`training/`, `app/`, `server/`) is not touched.

## Data source

- Repository: https://github.com/songyangme/GETNext, commit `6a79081aec6951ebec31bff6400d6ba0e40a613e`
  (cloned to `benchmarks/foursquare/external/GETNext/`, git-ignored; its code is read, not run).
- The splits ship as `dataset/NYC.zip` (not as a `dataset/NYC/` folder). Inside: `NYC/NYC_train.csv`,
  `NYC/NYC_val.csv`, `NYC/NYC_test.csv`, plus `graph_A.csv` (620 MB) and `graph_X.csv` for GETNext's flow graph,
  which this module does not need. The `__MACOSX/` entries are archive debris.
- Splits are used unchanged.

## Columns

| Column | Type | Meaning |
|---|---|---|
| `user_id` | int | user |
| `POI_id` | str | Foursquare venue id |
| `POI_catid` | str | Foursquare category id |
| `POI_catid_code` | int | integer code of the category |
| `POI_catname` | str | category name, e.g. "Historic Site" |
| `latitude`, `longitude` | float | venue coordinates |
| `timezone` | int | offset from UTC in minutes (-300 for New York) |
| `UTC_time`, `local_time` | str | check-in time |
| `day_of_week` | int | 0–6 |
| `norm_in_day_time` | float | time of day scaled to [0, 1) |
| `trajectory_id` | str | `<user_id>_<n>`: one user's check-ins grouped into a trajectory |
| `norm_day_shift`, `norm_relative_time` | int / float | time features used by GETNext |

## Split statistics

| Split | Check-ins | Users | POIs | Categories | Trajectories | Time range (UTC) |
|---|---|---|---|---|---|---|
| train | 83,228 | 1,047 | 4,980 | 318 | 11,022 | 2012-04-03 – 2012-11-09 |
| val | 10,339 | 513 | 2,558 | 265 | 1,559 | 2012-11-09 – 2012-12-21 |
| test | 10,374 | 461 | 2,403 | 268 | 1,566 | 2012-12-21 – 2013-02-14 |

- The split is chronological: every validation and test check-in is later than every training check-in, and no
  trajectory spans two splits.
- 6.6% of test check-ins are at POIs never seen in training and 7.6% are by users not in training; GETNext's
  protocol drops both (below).

## Evaluation protocol (from GETNext's code)

Read from `train.py` (`TrajectoryDatasetVal`) and `utils.py` (`*_last_timestep` functions):

1. For each trajectory in the evaluation file, skip it if its user is not in the training set.
2. Drop check-ins whose POI is not in the training set (the remaining ones are joined into one sequence).
3. Inputs are check-ins 1..n-1, labels 2..n. Skip the trajectory if it has fewer than 2 inputs
   (`--short-traj-thres 2`).
4. **Only the last step is scored:** one target per trajectory, the final check-in, given everything before it in
   that trajectory (and the model's knowledge from training).
5. Candidates are all POIs in the training set (4,980); the score vector covers every one, no sampling.
6. Acc@k = 1 if the target is in the top k. MRR = 1 / rank over the full list. "mAP@20" in their code is MRR cut at
   20. Ties are broken by `argsort` order.

Scored samples under this protocol: 1,399 validation trajectories, **1,351 test trajectories**.

**The released code never evaluates on the test file.** `train.py` takes only `--data-train` and `--data-val`,
reports metrics on validation every epoch, and picks the best epoch by `4 · Acc@1 + Acc@20` on validation. This
module applies the same sample construction and metric functions to `NYC_test.csv`, with every choice (epochs,
weights, hyper-parameters) made on validation only.

## Reproduce

From the repository root, with `pip install torch pandas pyyaml` (and `sentence-transformers` only for `--cat_text`):

```
git clone https://github.com/songyangme/GETNext benchmarks/foursquare/external/GETNext
python -m benchmarks.foursquare.test_leakage                 # 5 leakage checks
python -m benchmarks.foursquare.run --seed 0                 # full model; also writes baselines.json
python -m benchmarks.foursquare.run --seed 0 --variant no_cat
python -m benchmarks.foursquare.run --seed 0 --variant no_dist
python -m benchmarks.foursquare.run --seed 0 --cat_text
python -m benchmarks.foursquare.report                       # -> results/foursquare_nyc.md
```

Seeds 0, 1 and 2 were run. Settings are in `config.yaml`; values marked "tuned" came from a validation-only
sweep (`--val_only --set section.key=value`, results in `runs/foursquare_nyc/tuning/`) that never scores the test file.

Speed: about 1–3 s per epoch on one GPU (a shared 32 GB DGX GPU), about 160 s per epoch on a laptop CPU; runs stop
early after 11–33 epochs. The full data is used; nothing was subsampled.

## Differences from GETNext's setup

- Ranks are expected ranks under ties (GETNext's `argsort` breaks ties by index); identical when scores are distinct.
- AuraPath uses no time features and no trajectory order, by design (see `results/foursquare_nyc.md`).
- The user-frequency baseline (the user's own check-in counts) is added because 76.6% of test targets are revisits.
