"""
17_photo_scores.py -- how good is each photo as a picture OF ITS PLACE? (GPU server, reads brain/resnet_f16)

Per photo, from the stored ResNet50 features (no images are downloaded):
  * typicality  : cosine between the photo and the mean (L2-normalised) feature of its place. The landmark itself
                  scores high; selfies, food close-ups and blurry shots score low.
  * people      : probability mass the original ImageNet classifier (ResNet50 fc layer, applied to the stored
                  2048-d features) puts on clothing / accessory / person classes. High = people fill the frame.
  * junk        : the same for screens, documents, menus, packaging and similar non-place shots.
  * top_class   : the classifier's top ImageNet class (for spot checks).
The fc weights are taken from torchvision's ResNet50 checkpoint; both public versions are tried and the one whose
predictions are more confident on these features is used (the features came from one of them).

  CUDA_VISIBLE_DEVICES=2 python training/17_photo_scores.py --data data_v4 --places _v2
Outputs: brain/photo_scores<places>.npz (photo_id, place, typicality, people, junk, top_class) + a short report
"""
import argparse, glob, json, os, re, urllib.request
import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--brain", default="brain")
ap.add_argument("--data", default="data_v4")
ap.add_argument("--places", default="_v2")
ap.add_argument("--feats", default="brain/resnet_f16")
a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
out_path = f"{a.brain}/photo_scores{a.places}.npz"
assert not os.path.exists(out_path), f"{out_path} exists; refusing to overwrite"

# ---------- ImageNet classifier head ----------
WEIGHTS = {"v1": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
           "v2": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"}
CLASSES = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
cache = os.path.join(a.brain, "models", "resnet50_fc")
os.makedirs(cache, exist_ok=True)
heads = {}
for k, url in WEIGHTS.items():
    f = os.path.join(cache, f"fc_{k}.pt")
    if not os.path.exists(f):
        tmp = os.path.join(cache, "full.pth")
        print("downloading", url, flush=True)
        urllib.request.urlretrieve(url, tmp)
        sd = torch.load(tmp, map_location="cpu")
        torch.save({"weight": sd["fc.weight"], "bias": sd["fc.bias"]}, f)
        os.remove(tmp)                                   # keep only the 8 MB head, not the 100 MB checkpoint
    heads[k] = torch.load(f, map_location=dev)
cls_file = os.path.join(cache, "imagenet_classes.txt")
if not os.path.exists(cls_file):
    urllib.request.urlretrieve(CLASSES, cls_file)
names = [l.strip() for l in open(cls_file)]
assert len(names) == 1000

PEOPLE = r"suit|jersey|sweatshirt|cardigan|jean|miniskirt|sunglass|bow tie|windsor tie|lab coat|abaya|academic gown|apron|" \
         r"bikini|brassiere|cowboy hat|fur coat|gown|hoopskirt|kimono|military uniform|overskirt|pajama|poncho|sarong|" \
         r"swimming trunks|trench coat|wig|lipstick|groom|ballplayer|scuba diver|bulletproof vest|mortarboard|neck brace|" \
         r"stole|maillot|mask|cloak|vestment|bonnet|sombrero|crash helmet|football helmet|bearskin|face powder|hair spray|" \
         r"band aid|diaper|bib|necklace|feather boa|hat|cap|sweater|t-shirt|tee shirt|dress|polo|wool|shower cap|seat belt|" \
         r"bath towel|hand-held computer|cellular telephone|beer glass|wine bottle|soup bowl|plate$"
JUNK = r"web site|menu|book jacket|comic book|envelope|crossword|packet|carton|screen|monitor|television|laptop|notebook|" \
       r"desktop computer|ipod|remote control|computer keyboard|mouse|binder|rule|slide rule|letter opener|paper towel|" \
       r"toilet tissue|jigsaw puzzle|matchstick|digital clock|digital watch|oscilloscope|hard disc|modem|printer|" \
       r"photocopier|cassette|cd player|tape player|radio|loudspeaker|joystick|scoreboard|street sign|traffic light"
people_idx = torch.as_tensor([i for i, n in enumerate(names) if re.search(PEOPLE, n.lower())], device=dev)
junk_idx = torch.as_tensor([i for i, n in enumerate(names) if re.search(JUNK, n.lower())], device=dev)
print(f"people classes: {len(people_idx)}, junk classes: {len(junk_idx)}", flush=True)

# ---------- rows -> places ----------
pid = np.load(os.path.join(a.feats, "photo_ids.npy")).astype(np.int64)
pm = pd.read_csv(f"{a.brain}/photo_poi_mapping{a.places}.csv", usecols=["photoid", "poi_id"]).drop_duplicates("photoid")
poi_of_row = pd.Series(pm["poi_id"].values, index=pm["photoid"].values).reindex(pid).values
poi_ids = np.load(f"{a.data}/prepared.npz", allow_pickle=True)["poi_ids"].astype(np.int64)
place = np.full(len(pid), -1, dtype=np.int64)
ok = ~np.isnan(poi_of_row.astype(np.float64))
pos = np.searchsorted(poi_ids, poi_of_row[ok].astype(np.int64))
pos = np.clip(pos, 0, len(poi_ids) - 1)
hit = poi_ids[pos] == poi_of_row[ok].astype(np.int64)
idx_ok = np.nonzero(ok)[0]
place[idx_ok[hit]] = pos[hit]
P = len(poi_ids)
print(f"rows {len(pid):,}, rows with a place {int((place >= 0).sum()):,}, places {P:,}", flush=True)

parts = sorted(glob.glob(os.path.join(a.feats, "part_*.npy")))


def blocks():
    s = 0
    for f in parts:
        x = np.load(f, mmap_mode="r")
        for b in range(0, len(x), 65536):
            yield s + b, torch.as_tensor(np.array(x[b:b + 65536]), device=dev).float()
        s += len(x)


# ---------- choose the classifier head: more confident predictions = the matching checkpoint ----------
conf = {}
for k, h in heads.items():
    vals = []
    for n, (s, x) in enumerate(blocks()):
        vals.append(torch.softmax(x @ h["weight"].T + h["bias"], 1).max(1).values.mean().item())
        if n >= 2:
            break
    conf[k] = float(np.mean(vals))
best = max(conf, key=conf.get)
print("mean top-1 probability per checkpoint:", conf, "-> using", best, flush=True)
W, bias = heads[best]["weight"], heads[best]["bias"]

# ---------- pass 1: place means ----------
pl_t = torch.as_tensor(place, device=dev)
sums = torch.zeros(P, 2048, device=dev)
for s, x in blocks():
    p = pl_t[s:s + len(x)]
    m = p >= 0
    sums.index_add_(0, p[m], torch.nn.functional.normalize(x[m], dim=1))
means = torch.nn.functional.normalize(sums, dim=1)
del sums

# ---------- pass 2: per-photo scores ----------
typ = np.zeros(len(pid), dtype=np.float16); peo = typ.copy(); jnk = typ.copy()
top = np.zeros(len(pid), dtype=np.int16)
for s, x in blocks():
    p = pl_t[s:s + len(x)]
    prob = torch.softmax(x @ W.T + bias, 1)
    t = (torch.nn.functional.normalize(x, dim=1) * means[p.clamp(min=0)]).sum(1)
    t[p < 0] = 0
    e = s + len(x)
    typ[s:e] = t.cpu().numpy(); peo[s:e] = prob[:, people_idx].sum(1).cpu().numpy()
    jnk[s:e] = prob[:, junk_idx].sum(1).cpu().numpy(); top[s:e] = prob.argmax(1).cpu().numpy()

np.savez(out_path, photo_id=pid, place=place.astype(np.int32), typicality=typ, people=peo, junk=jnk, top_class=top)
rep = {"checkpoint": best, "confidence": conf, "rows": int(len(pid)),
       "people_mean": float(peo.astype(np.float32).mean()), "junk_mean": float(jnk.astype(np.float32).mean()),
       "top_classes": pd.Series([names[i] for i in top[place >= 0]]).value_counts().head(25).to_dict()}
json.dump(rep, open(f"{a.brain}/photo_scores{a.places}_report.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
print("saved", out_path)
