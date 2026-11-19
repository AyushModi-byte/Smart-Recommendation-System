# AuraPath v2 on Foursquare NYC (next-POI benchmark)

Splits, sample construction and metrics follow GETNext's released code ([benchmarks/foursquare/README.md](../benchmarks/foursquare/README.md)): one target per test trajectory (its last check-in), ranked against all 4,980 training POIs, users and POIs unseen in training skipped. Test samples: 1,351. Model choices were made on validation only; the test file was scored once per run.

## Our runs (test set)

| Method | Seeds | Acc@1 | Acc@5 | Acc@10 | Acc@20 | MRR | NDCG@10 |
|---|---|---|---|---|---|---|---|
| AuraPath v2 (category embedding) | 3 | 0.2233 ± 0.0253 | 0.5505 ± 0.0096 | 0.6556 ± 0.0063 | 0.7232 ± 0.0044 | 0.3635 ± 0.0184 | 0.4276 ± 0.0151 |
| AuraPath v2 + category-name text embedding (`--cat_text`) | 3 | 0.2228 ± 0.0114 | 0.5472 ± 0.0050 | 0.6504 ± 0.0034 | 0.7214 ± 0.0060 | 0.3633 ± 0.0065 | 0.4263 ± 0.0047 |
| Ablation: no category embedding | 3 | 0.1925 ± 0.0056 | 0.5093 ± 0.0097 | 0.6336 ± 0.0013 | 0.7071 ± 0.0057 | 0.3315 ± 0.0053 | 0.3982 ± 0.0044 |
| Ablation: no distance features | 3 | 0.0607 ± 0.0022 | 0.1660 ± 0.0182 | 0.2314 ± 0.0338 | 0.3022 ± 0.0330 | 0.1213 ± 0.0104 | 0.1392 ± 0.0159 |
| User's own check-in counts (training history + current trajectory) | – | 0.2339 | 0.5248 | 0.6477 | 0.7306 | 0.3673 | 0.4281 |
| Markov chain (POI → POI transitions) | – | 0.1510 | 0.3168 | 0.3679 | 0.4004 | 0.2251 | 0.2562 |
| Nearest place to the current check-in | – | 0.0674 | 0.2087 | 0.2894 | 0.3457 | 0.1394 | 0.1686 |
| Popular within 0.5 km of the current check-in | – | 0.0577 | 0.1917 | 0.2546 | 0.3168 | 0.1211 | 0.1457 |
| Popularity | – | 0.0074 | 0.0252 | 0.0392 | 0.0518 | 0.0191 | 0.0212 |

Mean ± sample standard deviation over seeds. Baselines are deterministic. The popular-within radius was chosen on validation.

## Published results on the same split (reported in paper, not reproduced)

| Method | Acc@1 | Acc@5 | Acc@10 | Acc@20 | MRR |
|---|---|---|---|---|---|
| GETNext (Yang et al., SIGIR 2022) | 0.2435 | 0.5089 | 0.6143 | 0.6880 | 0.3621 |
| STAN (Luo et al., WWW 2021), as reported by GETNext | 0.2231 | 0.4582 | 0.5734 | 0.6328 | 0.3253 |
| PLSPL, as reported by GETNext | 0.1917 | 0.3678 | 0.4523 | 0.5370 | 0.2806 |
| FPMC, as reported by GETNext | 0.1003 | 0.2126 | 0.2970 | 0.3323 | 0.1701 |

Source: Yang, Liu and Zhao, *GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation*, SIGIR 2022, Table 2, NYC columns ([arXiv:2303.04741](https://arxiv.org/abs/2303.04741)). STAN and the other baselines are GETNext's re-runs on this split, not numbers from their own papers. GETNext's released code reports validation only, so whether its Table 2 used exactly the protocol above on the test file cannot be checked from the code.

**LSTPM** (Sun et al., AAAI 2020) is not in GETNext's table, and its own paper uses a different NYC preprocessing and Recall/NDCG metrics, so no comparable number exists for this split.

## Notes

**This benchmark is mostly revisits.** 76.6% of test targets are places the user had already checked in at (73.4% in their training history, 38.0% earlier in the same trajectory). Counting the user's own check-ins, with no learning, already reaches the published GETNext numbers on Acc@5, Acc@10, Acc@20 and MRR under this protocol. Any comparison on NYC should show this baseline.

**AuraPath is about as good as that count, not better.** Against the user-frequency baseline it is lower on Acc@1 (0.223 vs 0.234) and Acc@20, slightly higher on Acc@5 and Acc@10, and equal on MRR, every difference within two seed standard deviations. Its distance features to the user's own past check-ins learn the revisit pattern (without them Acc@1 falls to 0.061); the category embedding adds about 3 points of Acc@1 on top (0.193 without it); the frozen text embedding of category names adds nothing.

**Against GETNext** (published, not reproduced) AuraPath is lower on Acc@1 (0.223 vs 0.2435), higher on Acc@5, Acc@10 and Acc@20, and equal on MRR. AuraPath has no sequential model: it sees the current location, recent check-ins as a set, category taste and popularity, but not the order of the trajectory or the time of day. Acc@1, picking the exact next place, is plausibly where order and time help most, which would fit the gap being there (not tested). Two cautions: the published numbers come from the authors' runs, and their test protocol cannot be checked from the released code, so a small protocol difference could move either side by a few points.
