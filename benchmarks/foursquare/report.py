"""
Build results/foursquare_nyc.md from runs/foursquare_nyc/*.json (test metrics, mean ± std over seeds).

  python -m benchmarks.foursquare.report
"""
import glob
import json
import os
import re

import numpy as np

RUNS = os.path.join("runs", "foursquare_nyc")
OUT = os.path.join("results", "foursquare_nyc.md")
COLS = ["Acc@1", "Acc@5", "Acc@10", "Acc@20", "MRR", "NDCG@10"]
VARIANTS = [("full_x-freq-time-trans", "**AuraPath v2 + transitions, visit counts, time slot** (`--extra trans,freq,time`)"),
            ("full", "AuraPath v2 (category embedding)"),
            ("full_cattext", "AuraPath v2 + category-name text embedding (`--cat_text`)"),
            ("no_cat", "Ablation: no category embedding"),
            ("no_dist", "Ablation: no distance features")]
BASE = [("user_frequency", "User's own check-in counts (training history + current trajectory)"),
        ("markov", "Markov chain (POI → POI transitions)"),("nearest", "Nearest place to the current check-in"),
        ("popular_within", "Popular within {r} km of the current check-in"), ("popularity", "Popularity")]

# Reported in: S. Yang, J. Liu, K. Zhao. GETNext: Trajectory Flow Map Enhanced Transformer for Next POI
# Recommendation. SIGIR 2022, Table 2 (NYC columns). arXiv:2303.04741.
PUBLISHED = [("GETNext (Yang et al., SIGIR 2022)", [0.2435, 0.5089, 0.6143, 0.6880, 0.3621]),
             ("STAN (Luo et al., WWW 2021), as reported by GETNext", [0.2231, 0.4582, 0.5734, 0.6328, 0.3253]),
             ("PLSPL, as reported by GETNext", [0.1917, 0.3678, 0.4523, 0.5370, 0.2806]),
             ("FPMC, as reported by GETNext", [0.1003, 0.2126, 0.2970, 0.3323, 0.1701])]


def load():
    runs = {}
    for f in sorted(glob.glob(os.path.join(RUNS, "*_seed*.json"))):
        m = re.match(r"(.+)_seed(\d+)\.json$", os.path.basename(f))
        if not m or "smoke" in m.group(1):
            continue
        j = json.load(open(f))
        if j.get("test"):
            runs.setdefault(m.group(1), []).append(j)
    return runs


def fmt(vals):
    v = np.array(vals)
    return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1 else f"{v[0]:.4f}"


EXTRA = ["trans", "freq", "time", "freq-trans", "time-trans", "freq-time", "freq-time-trans"]


def selection_table():
    """Validation-only choice of the --extra features (seeds 0-1, GETNext's monitor 4*Acc@1 + Acc@20)."""
    rows = [("none (tuned AuraPath v2)", sorted(glob.glob(os.path.join(RUNS, "tuning", "full_seed[01]_c_lr_near_bs.json"))))]
    rows += [(x.replace("-", " + "), sorted(glob.glob(os.path.join(RUNS, "tuning", f"full_x-{x}_seed[01].json")))) for x in EXTRA]
    L = ["| Extra features | Val Acc@1 | Val MRR | Val 4·Acc@1 + Acc@20 |", "|---|---|---|---|"]
    best = None
    for label, fs in rows:
        if not fs:
            continue
        v = [json.load(open(f))["val"] for f in fs]
        mon = np.mean([4 * x["Acc@1"] + x["Acc@20"] for x in v])
        L.append(f"| {label} | {np.mean([x['Acc@1'] for x in v]):.4f} | {np.mean([x['MRR'] for x in v]):.4f} | {mon:.3f} |")
        best = max(best or (mon, label), (mon, label))
    return L, best


def main():
    runs = load()
    base = json.load(open(os.path.join(RUNS, "baselines.json")))
    sel, best = selection_table()
    L = ["# AuraPath v2 on Foursquare NYC (next-POI benchmark)", "",
         "Splits, sample construction and metrics follow GETNext's released code "
         "([benchmarks/foursquare/README.md](../benchmarks/foursquare/README.md)): one target per test trajectory "
         "(its last check-in), ranked against all 4,980 training POIs, users and POIs unseen in training skipped. "
         f"Test samples: {base['markov']['test']['n']:,}. Model choices were made on validation only; "
         "the test file was scored once per run.", "",
         "## Our runs (test set)", "",
         "| Method | Seeds | " + " | ".join(COLS) + " |", "|---|---|" + "---|" * len(COLS)]
    for key, label in VARIANTS:
        if key in runs:
            rs = runs[key]
            L.append(f"| {label} | {len(rs)} | " + " | ".join(fmt([r['test'][c] for r in rs]) for c in COLS) + " |")
    for key, label in BASE:
        b = base[key]
        L.append(f"| {label.format(r=b.get('radius_km'))} | – | " + " | ".join(f"{b['test'][c]:.4f}" for c in COLS) + " |")
    L += ["", "Mean ± sample standard deviation over seeds. Baselines are deterministic. "
          "The popular-within radius was chosen on validation.", "",
          "## Published results on the same split (reported in paper, not reproduced)", "",
          "| Method | Acc@1 | Acc@5 | Acc@10 | Acc@20 | MRR |", "|---|---|---|---|---|---|"]
    for label, v in PUBLISHED:
        L.append(f"| {label} | " + " | ".join(f"{x:.4f}" for x in v) + " |")
    L += ["",
          "Source: Yang, Liu and Zhao, *GETNext: Trajectory Flow Map Enhanced Transformer for Next POI "
          "Recommendation*, SIGIR 2022, Table 2, NYC columns ([arXiv:2303.04741](https://arxiv.org/abs/2303.04741)). "
          "STAN and the other baselines are GETNext's re-runs on this split, not numbers from their own papers. "
          "GETNext's released code reports validation only, so whether its Table 2 used exactly the protocol above "
          "on the test file cannot be checked from the code.", "",
          "**LSTPM** (Sun et al., AAAI 2020) is not in GETNext's table, and its own paper uses a different NYC "
          "preprocessing and Recall/NDCG metrics, so no comparable number exists for this split.", "",
          "## Notes", "",
          "**This benchmark is mostly revisits.** 76.6% of test targets are places the user had already checked in "
          "at (73.4% in their training history, 38.0% earlier in the same trajectory). Counting the user's own "
          "check-ins, with no learning, already reaches the published GETNext numbers on Acc@5, Acc@10, Acc@20 and "
          "MRR under this protocol. Any comparison on NYC should show this baseline.", "",
          "**The base ranker is about as good as that count, not better.** AuraPath v2 as built for YFCC (category "
          "embedding, distance, density, popularity) is lower than the user-frequency baseline on Acc@1 (0.223 vs "
          "0.234) and equal on MRR. Its distance features to the user's own past check-ins learn the revisit pattern "
          "(without them Acc@1 falls to 0.061); the category embedding adds about 3 points of Acc@1 (0.193 without "
          "it); the frozen text embedding of category names adds nothing.", "",
          "**Three cheap sequence features close the gap** (`--extra`, `model.SeqFeats`): transitions from the "
          "current POI to the candidate (over all users and for this user), the user's own visit count, and how "
          "often the candidate and its category are visited in the current check-in's time slot (3-hour block x "
          "weekday/weekend). All counts come from the training file, with a training positive's own check-in left "
          "out; `test_leakage.py` checks both. The combination was chosen on validation only:", "",
          *sel, "",
          f"Chosen: {best[1]}. On test it reaches Acc@1 0.267 and MRR 0.399, above the user-frequency baseline and "
          "above the published GETNext numbers on every metric, with seed standard deviations of 0.006 or less. "
          "Time slot alone hurts on validation but helps together with the other two.", "",
          "**Against GETNext** (published, not reproduced): the published numbers come from the authors' runs, and "
          "their test protocol cannot be checked from the released code, so a small protocol difference could move "
          "either side by a few points. The fair claim is that a two-tower ranker with simple count features is "
          "competitive with GETNext on this split, not that it beats a transformer in general. It still has no "
          "learned sequence model: the trajectory enters only through the current POI, its time, and visit counts.", ""]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
