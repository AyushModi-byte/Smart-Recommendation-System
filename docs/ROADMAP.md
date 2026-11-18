# Roadmap: fixing the open issues

Covers every open item in [KNOWN_ISSUES.md](KNOWN_ISSUES.md) plus the proposal
features that are still missing (self-attention, weather / temperature / traffic /
mood, real-time filtering, collaborative filtering, splitting giant places).

## Outcome

All phases are complete. Numbers are in [RESULTS.md](RESULTS.md). Deployment went to the swipe app
([SWIPE_APP.md](SWIPE_APP.md)) instead of the Gradio demo; what is left now is in [NEXT_STEPS.md](NEXT_STEPS.md).

| Phase | Outcome |
|---|---|
| 0 | Done. Compact ResNet features (15 GB) on the server; peak use 36 GB of the 50 GB quota |
| 1 | **Adopted.** 828 oversized places split into 8,547; model lead over the best baseline rose from 1.7x to 2.6x |
| 2 | Tested, not adopted. Collaborative filtering works alone but adds nothing to the model |
| 3 | Tested, not adopted for ranking. Time, weather, temperature and crowd level add at most 1% NDCG@20. Mood profiles built for the demo |
| 4 | Tested, not adopted. Trained attention pooling equals mean pooling; learned weights stay near uniform |
| 5 | Tested, not adopted. New embeddings retrieve better (0.219 vs 0.106 for v1) but ranking does not improve |
| 6 | Analysis done, not adopted for ranking. K = 4 is the supported choice; ICA adds nothing over k-means++ |
| 7 | Demo has mood, scenery, live weather and time, and an avoid-crowds option (local only). The swipe app is deployed (Render + Vercel) but has mood only so far |
| 8 | Final model is `p1_split`. Proposal benchmarks: Precision/MAP met, MAE reduction 52% (target 60%) |

The plan below is kept as written, with results added under Phases 2 and 3.

## Ground rules

- **One change at a time, measured.** Each phase ends with the v2 protocol
  (time-based leave-one-out, full ranking, three baselines) and one row in a results table.
  A phase that does not beat the previous row is reported as such and not kept in the model.
- **Never overwrite.** New artifacts get new names (`photo_poi_mapping_v2.csv`, `data_v4/`,
  `runs/p1_...`). `brain/` on the laptop stays the untouched original.
- **Compare lift, not raw numbers, across Phase 1.** Changing what a "place" is changes the
  test set, so results before and after Phase 1 are compared by gain over the baselines.

## Order and why

Places are the foundation: every embedding, cluster, feature and interaction is keyed by
place id. Splitting them first avoids redoing everything twice.

| Phase | What | Fixes | Depends on | Rough effort |
|---|---|---|---|---|
| 0 | Backup; staged uploads within the 50 GB quota | n/a | n/a | 0.5 day |
| 1 | Split giant places, new baseline | issue 9 | 0 | 1–2 days |
| 2 | Collaborative filtering | issue 8 | 1 | 1–2 days |
| 3 | Context: time, weather, temperature, crowding, mood | proposal | 1 | 3–4 days |
| 4 | Trained self-attention pooling | issues 6, 7, 15 | 1 | 2–3 days |
| 5 | Quintuplet retraining with four levels | issue 10, 15 | 0, 1 | 2–3 days |
| 6 | Tag decoding and honest clustering | issues 11, 12 | 1 (5 if done) | 2 days |
| 7 | Real-time filtering in the demo, deploy | proposal Phase 3 | 3 | 2 days |
| 8 | Final model, report numbers, MAE | proposal benchmarks | all | 1 day |

Phases 2, 3 and 4 are independent of each other once Phase 1 is done.

---

## Phase 0: data logistics

**Constraint: the DGX home directory has a 50 GB quota.** 13.6 GB is used today, so 36 GB is
free. The raw ResNet features (53 GB as float32) cannot be copied as they are.

| Item on the DGX | Size | When |
|---|---|---|
| Already there (data, environment, runs) | 13.6 GB | now |
| ResNet features, only the 3.74M photos that belong to a place, as float16 | 15.3 GB | Phase 5 only |
| Context embeddings, same photos, float16 | 2.9 GB | Phase 6 only |
| New 256-d photo embeddings from Phase 5 | 3.6 GB | Phase 5 |
| New prepared data, runs, bundle | about 2 GB | Phases 1–8 |
| **Peak** | **about 37 GB** | |

- The float16 subsets are built on the laptop from the originals (which stay untouched) and
  sent in parts of about 1 GB, so a dropped transfer costs one part, not the whole file.
- Nothing large is uploaded before the phase that needs it. Phases 1–4 and 7 need no upload.
- Reserve if space gets tight: delete `data_v2` (0.5 GB, superseded), and replace the 5.1 GB
  CSV on the server with a slim copy of the columns in use (about 1.5 GB). Check `quota -s`
  before every large write.
- Make the external-drive backup of `brain/` before anything else.

## Phase 1: split giant places (issue 9)

**Problem.** DBSCAN at 100 m chains photos along streets. 109 places hold 18.7% of photos;
one holds 99,240.

**Approach** (`training/00a_split_places.py`).
1. Keep every place whose radius is under 250 m (the large majority) with its current id.
2. Inside each larger place, find density peaks at least 75 m apart on a 25 m grid (a fast
   grid form of mean-shift); each peak becomes a place and every photo joins its nearest peak.
   Whatever is still larger than 250 m is cut into 150 m grid cells.
3. Write `photo_poi_mapping_v2.csv` and the parent table.

**Rebuild.** Photo embeddings do not depend on place ids, so only aggregation is redone:
unchanged places keep their fuzzy memberships; new places get one predicted from their mean
visual embedding (80% top-cluster agreement on held-out places). Then `00b --places _v2` →
`data_v4`, then retrain the `v7` configuration. Phase 6 replaces the predicted memberships.

**Done when.** No place over 300 m radius; no place holds more than 0.5% of photos; new
baselines and a retrained model are in the results table. Also re-run destination mode to
see whether the popularity baseline was inflated by the giant places.

## Phase 2: collaborative filtering (issue 8)

**What is already known.** Run `v8` added raw co-visit counts as features and did not improve
on `v7`. So "add co-visits" in its simplest form has been tried. What has not been tried is a
proper collaborative model.

**Approach, cheapest first.**
1. Item-item similarity with normalisation (cosine / EASE-style) on train visits, scored
   alone under the v2 protocol. This shows how much collaborative signal exists at all.
2. Learned place-id embeddings added to the place tower, with dropout on the id so cold
   places still work from content.
3. Blend: collaborative score as an input to the interaction layer.

**Risk to state up front.** The data is sparse (about 350k train visits over 170k places,
roughly two per place, and more places after Phase 1). The gain may be small. Step 1 costs an
hour and answers the question before more is built.

**Result of step 1** (`training/07_collab_filter.py`, `runs/p1_split/collab_results.json`).
14,955 test users on the split places:

| Method | HR@10 | NDCG@20 |
|---|---|---|
| Collaborative filtering alone | 0.083 | 0.053 |
| Model alone | 0.175 | 0.118 |
| Model + collaborative filtering (weight tuned on validation) | 0.176 | 0.119 |

The signal exists (alone it beats every non-learned baseline), but it adds nothing measurable
to the model: NDCG@20 gain 0.001, 95% CI [-0.000, 0.002]. Together with run `v8` this is two
null results, so the model already captures it through geography. **Decision: steps 2 and 3
are not pursued; collaborative filtering is not added to the model.**

## Phase 3: context features (weather, temperature, traffic, mood)

None of these exist in the dataset. Each needs a source, and two of them can only be proxies.

| Feature | Source | Honest status |
|---|---|---|
| Time of day, weekday, season | photo timestamp | real |
| Daylight / night | computed from time and latitude | real |
| Weather, temperature | historical reanalysis (ERA5, free through Open-Meteo's archive or Copernicus) joined on coarse grid cell and date | real, coarse (about 25 km, daily or hourly) |
| Traffic | no free historical traffic data for 2004–2014 | **proxy**: crowd level, the number of distinct photographers at a place by hour-of-week and month |
| Mood | no ground truth for a user's mood | **proxy**: a mood profile per place from its tags and titles (similarity to prompts such as relaxing, lively, romantic, adventurous); the user states their mood at query time |

**How they enter the model.** Not as raw inputs but as *place-by-context affinities* computed
from train visits only: how much more (or less) a place is visited when it rains, at night,
in winter, when it is cold. At ranking time the affinity is combined with the context of the
test visit (its timestamp, and the weather at the candidate's location on that day).

**Steps.**
1. Time and daylight affinities (no external data). Evaluate.
2. Weather join. Check current API limits first; requests are per grid cell for the whole date
   range, so the number of calls is the number of cells with photos, not the number of photos.
3. Crowd profile per place. Evaluate as a feature; also expose as "avoid crowds" in the demo.
4. Mood profile per place. This cannot be evaluated offline (no labels); report it as a
   demo feature and say so.

**Done when.** A table of each context group added alone and together, against Phase 1.

**Result of step 1** (`training/08_context_time.py`, `runs/p1_split/context_time*_results.json`).
Query context = the timestamp of the held-out visit. 14,955 test users, weights tuned on validation:

| Method | HR@10 | NDCG@20 | NDCG@20 gain over model, 95% CI |
|---|---|---|---|
| Model alone | 0.1745 | 0.1184 | |
| + time affinity per place | 0.1736 | 0.1192 | +0.0008 [+0.0001, +0.0015] |
| + time affinity, backing off to the place's aesthetic cluster | 0.1741 | 0.1194 | +0.0011 [+0.0003, +0.0017] |
| + popularity in the previous 365 days | 0.1745 | 0.1184 | none (best weight was 0) |

Time of day, weekday, season and daylight together add about 1% to NDCG@20 and nothing to
HR@10; no single group is distinguishable from zero. Places average under three visits each,
so a per-place time profile cannot be learned, and the cluster-level profile is too coarse to
change a ranking. **Decision: time context is not added to the ranking model.** It is kept for
the demo, where it acts as an explicit, explainable filter (Phase 7). The same sparsity applies
to weather affinities, so step 2 is evaluated the same cheap way before anything is retrained.

## Phase 4: trained self-attention (issues 6, 7, 15)

**Problem.** Notebook 07 scores photos with networks that were never trained, then a keyword
list discards 80% of photos.

**Approach.** Replace mean pooling with learned attention pooling, trained end to end with the
ranking loss:
- a place (and a user) is a set of photo embeddings; a small attention layer assigns each photo
  a weight, and the weighted sum is the embedding;
- during training, sample up to 16–32 photos per set; at inference, pool once and cache;
- no photo is discarded and no keyword list is used.

This is what the proposal describes: significant photos get higher weight, outliers lower.

**Leakage.** Held-out visits' photos stay out of the sets, exactly as in `00b`.

**Done when.** Attention pooling vs mean pooling under the same protocol, plus the content
ablation re-run to see whether the visual share rises above the current 8%.

## Phase 5: quintuplet retraining (issues 10, 15)

**Approach.** Implement the four similarity levels from the proposal:
same user and place > different user, same place > same user, different place > different
user and place. Use in-batch semi-hard negatives (the unused mining function exists), smaller
per-level margins, a validation split by place, and a retrieval metric (do held-out photos of
a place retrieve each other) in place of validation loss.

**Caveat.** "Same place" is a geographic label, not an aesthetic one. Better embeddings may
help only a little; the ablation in Phase 4 says how much there is to win. If the gain is
below noise, keep the existing embeddings.

Needs the ResNet features on the DGX: the 15.3 GB float16 subset from Phase 0, not the full 53 GB.

## Phase 6: tags and clustering (issues 11, 12)

- **Tags.** Decode URL-encoded tags, re-encode with the sentence model (about 75 minutes of GPU).
- **Choosing K.** Use the ICA to optimise the cluster centres (its usual role with FCM, to
  escape poor local minima), and choose K with fuzzy validity indices on the full data over
  several seeds. Report stability across seeds.
- **Labels.** Name clusters from their most characteristic tags (the tag table from
  `poi_names.py`), with a human check, in place of 20 hand-picked photos.
- Silhouette is 0.11 today. If structure stays this weak, say so in the report.

## Phase 7: real-time filtering and deployment

- Destination and local time give live weather from a free forecast API; the ranking is
  adjusted with the place affinities from Phase 3.
- Controls: mood, "avoid crowds", indoor-friendly when raining. Manual controls remain as a
  fallback if the weather call fails.
- Deploy to Hugging Face Spaces (`DEPLOY.md`).

## Phase 8: final numbers

- Train the final model with every phase that earned its place; run `02_evaluate.py` once.
- Report the proposal's benchmarks: Precision and MAP against popularity (already met),
  cold-start in a new city (destination mode), and MAE of predicted visit probability on
  held-out positives and sampled negatives against a popularity-based predictor.
- Update README, report and slides from one results table.

## What this plan will not deliver

- Real traffic data, or a model of the user's actual mood. Both are proxies and are labelled so.
- A guarantee of higher metrics. Phases 2, 4 and 5 are experiments; the plan is built so that
  a null result is cheap and visible.
