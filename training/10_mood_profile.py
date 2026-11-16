"""
10_mood_profile.py -- a mood profile for every place, from its tags (ROADMAP Phase 3, step 4).

There is no record of any user's mood in the data, so this is a property of PLACES, chosen by the user at
query time. Each place is described by its most used tags (decoded, counted by distinct users), encoded with
the sentence model all-MiniLM-L6-v2, and compared with a few phrases per mood. Scores are z-scored over places.
It cannot be evaluated offline (no labels); the script prints examples so the result can be judged by eye.

Also writes the tag table and the place text embeddings, which the clustering step reuses.

  CUDA_VISIBLE_DEVICES=0 python training/10_mood_profile.py --data data_v4 --brain brain --places _v2
Outputs: brain/poi_tags<places>.csv, brain/poi_text_emb<places>.npy, brain/poi_mood<places>.npz, brain/poi_names<places>.npz
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poi_names import build_names

MOODS = {
    "Relaxed": ["a calm, quiet and peaceful place to relax", "tranquil garden, lake or beach, serene and restful"],
    "Lively": ["a lively, crowded, energetic place with nightlife, music and parties", "festival, concert, bar, busy street market"],
    "Romantic": ["a romantic place for couples, sunset views and beautiful scenery", "charming old town, candlelight, scenic viewpoint"],
    "Adventurous": ["an adventurous outdoor place for hiking, climbing and exploring wild nature", "mountains, trails, canyon, wilderness"],
    "Curious": ["a place to learn about history, art and culture", "museum, gallery, monument, cathedral, ancient ruins, architecture"],
    "Playful": ["a fun place for families and children", "zoo, aquarium, amusement park, playground, fair"],
}

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_v4")
ap.add_argument("--brain", default="brain")
ap.add_argument("--places", default="_v2")
ap.add_argument("--top-tags", type=int, default=12)
ap.add_argument("--model", default="brain/models/all-MiniLM-L6-v2", help="local folder with the model files (or a hub name)")
a = ap.parse_args()
t0 = time.time()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = np.load(os.path.join(a.data, "prepared.npz"), allow_pickle=True)
poi_ids, n_pois = d["poi_ids"], len(d["poi_ids"])

print("[1/3] tags per place", flush=True)
tag_csv, name_npz = f"{a.brain}/poi_tags{a.places}.csv", f"{a.brain}/poi_names{a.places}.npz"
if os.path.exists(tag_csv) and os.path.exists(name_npz):               # reuse an earlier run
    names = np.load(name_npz)["name"]
    G = pd.read_csv(tag_csv, keep_default_na=False, dtype={"tag": str})      # a tag can literally be "nan" or "null"
    G["rank"] = G.groupby("p").cumcount()
else:
    names, urls, G = build_names(a.brain, poi_ids, d["poi_lat"].astype(np.float64), d["poi_lon"].astype(np.float64),
                                 places=a.places, with_tags=True)
    np.savez_compressed(name_npz, poi_ids=poi_ids, name=names, url=urls)
    G = G.sort_values(["p", "sup", "idf"], ascending=[True, False, False])
    G["rank"] = G.groupby("p").cumcount()
    G[G["rank"] < 30][["p", "tag", "sup"]].to_csv(tag_csv, index=False)
top = G[G["rank"] < a.top_tags]
text = top.groupby("p")["tag"].apply(", ".join).reindex(np.arange(n_pois)).fillna("")
has = (text != "").values
print(f"    places with tags: {has.mean():.1%}", flush=True)

print("[2/3] encoding with all-MiniLM-L6-v2", flush=True)
from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
enc = AutoModel.from_pretrained(a.model).to(dev).eval()


@torch.no_grad()
def encode(texts, bs=512):
    out = []
    for s in range(0, len(texts), bs):
        b = tok(list(texts[s:s + bs]), padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
        h = enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(e, dim=1).cpu())
    return torch.cat(out).numpy()


E = np.zeros((n_pois, 384), dtype=np.float32)
E[has] = encode(text.values[has])
np.save(f"{a.brain}/poi_text_emb{a.places}.npy", E.astype(np.float16))

print("[3/3] mood scores", flush=True)
M = np.stack([encode(v).mean(0) for v in MOODS.values()])
M /= np.linalg.norm(M, axis=1, keepdims=True)
sim = E @ M.T                                                          # (P, moods)
sim = sim - sim.mean(1, keepdims=True)                                # relative to the place's own average: which mood, not how "place-like"
mu, sd = sim[has].mean(0), sim[has].std(0)
score = np.where(has[:, None], (sim - mu) / sd, 0.0).astype(np.float32)   # places without tags are neutral
np.savez_compressed(f"{a.brain}/poi_mood{a.places}.npz", poi_ids=poi_ids, moods=np.array(list(MOODS)), score=score.astype(np.float16), has_tags=has)

cnt = np.bincount(np.concatenate([d["train_p"], d["val_p"]]).astype(np.int64), minlength=n_pois)
clu = d["poi_cluster"].astype(np.int64)
for j, mood in enumerate(MOODS):
    cand = np.nonzero(has & (cnt >= 30))[0]
    best = cand[np.argsort(-score[cand, j])[:8]]
    print(f"  {mood:12s} " + " | ".join(names[best]))
print("mean mood score per aesthetic cluster (rows = clusters 0-5):")
print(pd.DataFrame([score[has & (clu == c)].mean(0) for c in range(6)], columns=list(MOODS)).round(2).to_string())
print(f"correlation between moods:\n{pd.DataFrame(np.corrcoef(score[has].T), index=list(MOODS), columns=list(MOODS)).round(2).to_string()}")
print(f"done [{time.time() - t0:.0f}s]")
