# Results

All numbers use the v2 protocol: each user's last visit is held out, the model ranks it against
every place, and the user's earlier places are removed from the ranking. Test set: 14,955 users,
177,564 places (after splitting oversized places). Source files are in `runs/`.

## Final model (`runs/p1_split`)

| Method | HR@10 | HR@20 | NDCG@10 | NDCG@20 | MRR |
|---|---|---|---|---|---|
| **AuraPath v2** | **0.175** | **0.232** | **0.104** | **0.118** | **0.092** |
| Popular within 25 km | 0.068 | 0.109 | 0.035 | 0.046 | 0.033 |
| Nearest place | 0.058 | 0.100 | 0.028 | 0.039 | 0.027 |
| Popularity | 0.010 | 0.018 | 0.005 | 0.007 | 0.005 |

- 2.6 times the strongest baseline on HR@10 and NDCG@20.
- Local visits (within 50 km of the user's history, 75% of test): HR@10 0.231.
- Trips to a new area (25%): HR@10 0.005, about the same as popularity (0.007). Without a stated
  destination the model cannot guess where a user will travel.
- 18.8% of test visits go to places with no earlier photos and are excluded as unrecommendable.

### With a stated destination

The 3,753 new-area trips, ranking only places within 30 km of the destination (median 339 candidates):

| Method | HR@5 | HR@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|
| **Popularity + AuraPath** | **0.338** | **0.456** | **0.585** | **0.311** |
| Popularity in area | 0.305 | 0.424 | 0.561 | 0.292 |
| Visual taste only | 0.092 | 0.148 | 0.219 | 0.093 |
| Random in area | 0.055 | 0.101 | 0.169 | 0.067 |

## Proposal benchmarks

| Benchmark in the proposal | Result |
|---|---|
| Precision and MAP more than 23% above popularity | **Met.** Precision@10 0.0174 vs 0.0010, MAP 0.092 vs 0.005. Against the strongest baseline (popular within 25 km): +155% Precision@10, +178% MAP |
| 60% reduction in MAE | **Not met: 52%.** MAE of visit probability over each user's held-out place and 99 sampled others: 0.0089 vs 0.0184 for popularity. Against popular-within-25-km the reduction is 17% (`mae_results.json`) |
| Cold start in a new city | **Only with a stated destination**: HR@10 0.456 vs 0.424 for popularity. Without one, 0.005 |

## What each idea contributed

Every row changes one thing against the final model and is evaluated on the same test users.
"Gain" is the change in NDCG@20.

| Idea | HR@10 | NDCG@20 | Gain | Kept? |
|---|---|---|---|---|
| Final model (mean pooling, v1 embeddings, 6 clusters) | 0.1745 | 0.1184 | | |
| + collaborative filtering (item-item, blended) | 0.1761 | 0.1193 | +0.0010, CI [-0.0002, +0.0021] | No |
| + time of day, weekday, season, daylight | 0.1741 | 0.1194 | +0.0011, CI [+0.0003, +0.0017] | No (about 1%, nothing on HR@10) |
| + rain and temperature (NOAA observations) | 0.1748 | 0.1185 | +0.0001, CI [-0.0001, +0.0003] | No |
| + crowd level (traffic proxy) | 0.1745 | 0.1184 | 0 (best weight was 0) | No |
| + popularity in the previous 365 days | 0.1745 | 0.1184 | 0 (best weight was 0) | No |
| Trained attention pooling over photos (`p4_attn`) | 0.1739 | 0.1198 | +0.0014 | No (within noise) |
| Retrained quintuplet embeddings, four levels (`p5_quint`) | 0.1703 | 0.1155 | -0.0029 | No |
| New clustering, K = 4 (`p6_clusters`) | 0.1628 | 0.1112 | -0.0072 | No |
| New clustering, K = 6 (`p6_clusters_k6`) | 0.1597 | 0.1093 | -0.0091 | No |

The one change that mattered was structural: splitting the oversized places (below).

### Splitting oversized places

828 "places" held 27% of all photos (the largest 99,240). They became 8,547 places of at most
250 m radius. Absolute scores fell because a city-centre blob is an easy target, but the baselines
fell much further:

| | HR@10 before | HR@10 after |
|---|---|---|
| AuraPath v2 | 0.225 | 0.175 |
| Popular within 25 km | 0.129 | 0.068 |
| Popularity | 0.074 | 0.010 |
| Model / best baseline | 1.7x | 2.6x |

### Why context features did not help

Places average under three visits each, so a per-place profile of "visited more when it rains"
cannot be estimated; falling back to the place's cluster gives a profile too coarse to change a
ranking. Weather was known for 78% of test queries, so coverage is not the reason.

### Attention pooling

The learned weights stay close to uniform: on places with 20 or more photos the effective share
of photos carrying the weight is 0.93 (1.0 = plain mean). Two training pitfalls had to be fixed
before it even matched mean pooling: pooling small random samples in training but all photos at
evaluation, and pooling positives and negatives from different sample sizes (which the model can
detect). The code keeps both fixes.

### Quintuplet retraining

Retrieval on 3,434 places never seen in training (share of photos whose 10 nearest neighbours by
other users include one from the same place):

| Embedding | Recall@10 |
|---|---|
| Raw ResNet50 (2048-d) | 0.195 |
| v1 projector (notebook 06) | 0.106 |
| New four-level projector | **0.219** |

The v1 projector is worse than the features it was trained from, even though these places were in
its training set. The new projector fixes that, and the ablation shows the ranking model leans on
it more (shuffling user taste costs 15% of NDCG@20, against 9.5% with the v1 embeddings), but the
overall ranking score does not improve. Geography dominates either way.

### Clustering

| K | Xie-Beni (lower is better) | Partition coefficient | Silhouette | Stability across seeds (ARI) |
|---|---|---|---|---|
| **4** | **0.94** | **0.70** | **0.18** | 1.00 |
| 5 | 1.15 | 0.66 | 0.16 | 1.00 |
| 6 | 1.13 | 0.63 | 0.15 | 0.72 |
| 8 | 1.86 | 0.58 | 0.14 | 1.00 |
| 12 | 2.93 | 0.49 | 0.11 | 1.00 |

- Four clusters are the best-supported choice: coast and water; nature and wildlife; events, food
  and nightlife; architecture and heritage. Six, the v1 choice, is unstable across seeds.
- The Imperialist Competitive Algorithm, used properly to optimise the centres, ends at the same
  objective as k-means++ initialisation (difference within 0.25% either way). It adds nothing here.
- The new clusters are cleaner but less useful to the ranking model than the v1 memberships. A
  likely reason, not verified: the v1 features included location and time, so those memberships
  carry some geography.

### How much the visual signal contributes

Final model, same test users, content inputs deliberately damaged:

| Variant | HR@10 | NDCG@20 |
|---|---|---|
| Normal | 0.177 | 0.120 |
| User taste shuffled between users | 0.158 | 0.109 |
| Place content replaced by the global mean | 0.160 | 0.113 |
| Both | 0.160 | 0.111 |

About 9% of NDCG@20 comes from the visual embeddings and clusters; the rest from geography and popularity.

## Learning taste from liked places (swipe app)

`training/14_swipe_check.py`, 6,000 test users: the user's taste is the mean of the places they visited instead of
their own photos, and with "last k" only their k most recent places are known (a new app user with k likes).

| Input | AuraPath HR@10 | NDCG@20 | Best baseline HR@10 |
|---|---|---|---|
| Own photos (normal) | 0.175 | 0.118 | 0.066 |
| All visited places | 0.177 | 0.121 | |
| Last 1 place | 0.145 | 0.099 | 0.117 |
| Last 3 places | 0.156 | 0.108 | 0.097 |
| Last 5 places | 0.164 | 0.114 | 0.085 |
| Last 10 places | 0.171 | 0.117 | 0.073 |

Places substitute for photos with no loss, so the app can learn from likes alone. Details in
[SWIPE_APP.md](SWIPE_APP.md).

## Public benchmark: Foursquare NYC

`benchmarks/foursquare/`: the same ranker (category embedding instead of photos) on GETNext's NYC split and
protocol, 1,351 test trajectories, the last check-in ranked against all 4,980 POIs. Mean of 3 seeds; full table,
ablations and citations in [results/foursquare_nyc.md](../results/foursquare_nyc.md).

| Method | Acc@1 | Acc@5 | Acc@10 | Acc@20 | MRR |
|---|---|---|---|---|---|
| AuraPath v2 | 0.223 | 0.551 | 0.656 | 0.723 | 0.364 |
| User's own check-in counts | 0.234 | 0.525 | 0.648 | 0.731 | 0.367 |
| Markov chain | 0.151 | 0.317 | 0.368 | 0.400 | 0.225 |
| GETNext (published, not reproduced) | 0.2435 | 0.5089 | 0.6143 | 0.6880 | 0.3621 |

77% of NYC test targets are revisits, so counting the user's own check-ins is already competitive. AuraPath
matches that count and GETNext on MRR, is higher at Acc@5–20 and lower at Acc@1.

## Features that cannot be evaluated offline

- **Mood.** No user's mood is recorded. Each place has a mood profile from its tags (Relaxed, Lively,
  Romantic, Adventurous, Curious, Playful) that the user chooses from in the demo. Spot checks look
  right: museums under Curious, national parks under Adventurous, zoos under Playful.
- **Traffic.** No historical traffic data exists for these years. The stand-in is the number of
  photographers at a place in a time slot.

## Earlier results on the original places (`runs/v7`)

Kept for reference; superseded by the tables above. 13,236 users, 169,845 places: HR@10 0.225,
NDCG@20 0.156 (popular within 25 km 0.129 / 0.092); destination mode 0.692 vs 0.663 for popularity.
