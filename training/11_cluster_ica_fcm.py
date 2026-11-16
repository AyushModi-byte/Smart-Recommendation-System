"""
11_cluster_ica_fcm.py -- aesthetic clusters, redone honestly (ROADMAP Phase 6; KNOWN_ISSUES 11 and 12).

What changed against notebook 08:
  * features per place = mean visual embedding + embedding of its DECODED tags (no time / location, which are
    not aesthetics); the notebook used URL-encoded tags and only the 20% of photos that passed a keyword filter
  * the Imperialist Competitive Algorithm optimises the cluster CENTRES (its usual role next to FCM: escaping
    poor local minima); the notebook's "ICA" only looped over K values it had already evaluated
  * K is chosen from fuzzy validity indices on all places over several seeds, with stability across seeds
  * clusters are described by their most characteristic tags, not by 20 hand-picked photos

  CUDA_VISIBLE_DEVICES=0 python training/11_cluster_ica_fcm.py --data data_v4 --brain brain --places _v2
Outputs: brain/poi_fuzzy_memberships_v3.npy, brain/poi_cluster_ids_v3.npy, brain/cluster_report_v3.json
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_v4")
ap.add_argument("--brain", default="brain")
ap.add_argument("--places", default="_v2")
ap.add_argument("--out", default="_v3")
ap.add_argument("--kmin", type=int, default=4)
ap.add_argument("--kmax", type=int, default=12)
ap.add_argument("--k", type=int, default=0, help="force K (0 = choose)")
ap.add_argument("--m", type=float, default=1.3, help="fuzzifier, as in notebook 08")
ap.add_argument("--pca", type=int, default=32)
ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t0 = time.time()

d = np.load(os.path.join(a.data, "prepared.npz"), allow_pickle=True)
poi_ids = d["poi_ids"]
MD = int(d["mem_dim"]) if "mem_dim" in d.files else 6
vis = d["poi_emb"][:, :-MD].astype(np.float32)
fit_ok = ~d["poi_no_clean"].astype(bool)                               # places with photos of their own
txt = np.load(f"{a.brain}/poi_text_emb{a.places}.npy").astype(np.float32)
has_txt = np.abs(txt).sum(1) > 0
unit = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
txt_fill = np.where(has_txt[:, None], unit(txt), unit(txt[has_txt]).mean(0))          # places without tags: average text
X_raw = np.concatenate([unit(vis), txt_fill], axis=1)
pca = PCA(n_components=a.pca, random_state=0).fit(X_raw[fit_ok])
X_all = pca.transform(X_raw).astype(np.float32)
print(f"places {len(poi_ids):,} (fitted on {int(fit_ok.sum()):,}), features {X_raw.shape[1]} -> {a.pca} PCA dims "
      f"({pca.explained_variance_ratio_.sum():.1%} variance)", flush=True)
X = torch.as_tensor(X_all[fit_ok], device=dev)
N = len(X)
EXP = 2.0 / (a.m - 1.0)


def memberships(Xc, C):
    d2 = (torch.cdist(Xc, C) ** 2).clamp(min=1e-10)
    inv = (d2 / d2.min(1, keepdim=True).values) ** (-EXP / 2.0)        # (1/d)^(2/(m-1)), scaled so it cannot overflow
    return inv / inv.sum(1, keepdim=True), d2


def objective(Xc, C):
    u, d2 = memberships(Xc, C)
    return float(((u ** a.m) * d2).sum())


def fcm(C, iters=150, tol=1e-5):
    for _ in range(iters):
        u, _ = memberships(X, C)
        um = u ** a.m
        C_new = (um.T @ X) / um.sum(0)[:, None]
        if float((C_new - C).abs().max()) < tol:
            C = C_new
            break
        C = C_new
    return C


def ica_centres(K, gen, n_countries=24, n_imp=4, iters=40):
    """Imperialist Competitive Algorithm over sets of K centres; cost = FCM objective on a fixed subsample."""
    sub = X[torch.randperm(N, generator=gen, device="cpu")[:20000].to(dev)]
    cost = lambda C: objective(sub, C)
    countries = [X[torch.randperm(N, generator=gen, device="cpu")[:K].to(dev)].clone() for _ in range(n_countries)]
    costs = [cost(c) for c in countries]
    order = np.argsort(costs)
    imps = [countries[i] for i in order[:n_imp]]; imp_cost = [costs[i] for i in order[:n_imp]]
    cols = [countries[i] for i in order[n_imp:]]; col_cost = [costs[i] for i in order[n_imp:]]
    owner = [j % n_imp for j in range(len(cols))]
    scale = float(X.std())
    for _ in range(iters):
        for j in range(len(cols)):
            e = owner[j]
            # assimilation: move towards the imperialist (matched centre by centre), with a random deviation
            match = torch.cdist(cols[j], imps[e]).argmin(1)
            step = 2.0 * torch.rand(K, 1, generator=gen).to(dev)
            cols[j] = cols[j] + step * (imps[e][match] - cols[j])
            if float(torch.rand(1, generator=gen)) < 0.15:             # revolution: re-seed one centre
                cols[j][int(torch.randint(K, (1,), generator=gen))] = X[int(torch.randint(N, (1,), generator=gen))]
            else:
                cols[j] = cols[j] + 0.02 * scale * torch.randn(K, X.shape[1], generator=gen).to(dev)
            col_cost[j] = cost(cols[j])
            if col_cost[j] < imp_cost[e]:                              # the colony takes over its empire
                imps[e], cols[j] = cols[j], imps[e]
                imp_cost[e], col_cost[j] = col_cost[j], imp_cost[e]
        # imperialistic competition: the weakest empire hands its weakest colony to the strongest
        power = [imp_cost[e] + 0.1 * np.mean([col_cost[j] for j in range(len(cols)) if owner[j] == e] or [imp_cost[e]]) for e in range(n_imp)]
        weak, strong = int(np.argmax(power)), int(np.argmin(power))
        mine = [j for j in range(len(cols)) if owner[j] == weak]
        if mine and weak != strong:
            owner[max(mine, key=lambda j: col_cost[j])] = strong
    return imps[int(np.argmin(imp_cost))]


def kpp_centres(K, gen):
    C = [X[int(torch.randint(N, (1,), generator=gen))]]
    for _ in range(K - 1):
        d2 = torch.cdist(X, torch.stack(C)).min(1).values ** 2
        C.append(X[int(torch.multinomial((d2 / d2.sum()).cpu(), 1, generator=gen))])
    return torch.stack(C)


def run(K, seed):
    gen = torch.Generator().manual_seed(seed)
    C_ica = fcm(ica_centres(K, gen))
    C_kpp = fcm(kpp_centres(K, gen))
    j_ica, j_kpp = objective(X, C_ica), objective(X, C_kpp)
    C = C_ica if j_ica <= j_kpp else C_kpp
    u, _ = memberships(X, C)
    sep = torch.cdist(C, C)
    sep = float((sep + torch.eye(K, device=dev) * 1e9).min()) ** 2
    return {"C": C, "u": u, "J_ica": j_ica, "J_kmeanspp": j_kpp, "xie_beni": min(j_ica, j_kpp) / (N * sep),
            "partition_coef": float((u ** 2).sum(1).mean()), "labels": u.argmax(1).cpu().numpy()}


sil_idx = np.random.RandomState(0).choice(N, 6000, replace=False)
ari_idx = np.random.RandomState(1).choice(N, 20000, replace=False)       # subsample: the pair counts overflow on all places
Xs = X[torch.as_tensor(sil_idx, device=dev)].cpu().numpy()
rows, runs = [], {}
ks = [a.k] if a.k else list(range(a.kmin, a.kmax + 1))
print(f"{'K':>3s} {'Xie-Beni':>9s} {'part.coef':>10s} {'silhouette':>11s} {'stability':>10s} {'ICA better':>11s} {'ICA gain':>9s}   (mean over {a.seeds} seeds)", flush=True)
for K in ks:
    rs = [run(K, s) for s in range(a.seeds)]
    runs[K] = min(rs, key=lambda r: min(r["J_ica"], r["J_kmeanspp"]))
    ari = [adjusted_rand_score(rs[i]["labels"][ari_idx], rs[j]["labels"][ari_idx]) for i in range(len(rs)) for j in range(i + 1, len(rs))]
    row = {"K": K, "xie_beni": float(np.mean([r["xie_beni"] for r in rs])), "partition_coef": float(np.mean([r["partition_coef"] for r in rs])),
           "silhouette": float(np.mean([silhouette_score(Xs, r["labels"][sil_idx]) for r in rs])),
           "stability_ari": float(np.mean(ari)) if ari else 1.0,
           "ica_better_share": float(np.mean([r["J_ica"] < r["J_kmeanspp"] for r in rs])),
           "ica_objective_gain": float(np.mean([(r["J_kmeanspp"] - r["J_ica"]) / r["J_kmeanspp"] for r in rs]))}
    rows.append(row)
    print(f"{K:3d} {row['xie_beni']:9.4f} {row['partition_coef']:10.3f} {row['silhouette']:11.3f} {row['stability_ari']:10.3f} "
          f"{row['ica_better_share']:11.0%} {row['ica_objective_gain']:+9.2%}   [{time.time() - t0:.0f}s]", flush=True)

tab = pd.DataFrame(rows)
if a.k:
    K = a.k
else:                                                                  # stable solutions first, then the most compact / separated
    stable = tab[tab["stability_ari"] >= max(0.80, tab["stability_ari"].max() - 0.05)]
    K = int(stable.sort_values("xie_beni").iloc[0]["K"])
best = runs[K]
print(f"chosen K = {K}", flush=True)

u_all, _ = memberships(torch.as_tensor(X_all, device=dev), best["C"])
u_all = u_all.cpu().numpy().astype(np.float32)
np.save(f"{a.brain}/poi_fuzzy_memberships{a.out}.npy", u_all)
np.save(f"{a.brain}/poi_cluster_ids{a.out}.npy", poi_ids.astype(np.int64))

# ---------- describe the clusters by their most characteristic tags ----------
hard = u_all.argmax(1)
tags = pd.read_csv(f"{a.brain}/poi_tags{a.places}.csv", keep_default_na=False, dtype={"tag": str})
tags = tags[tags["p"].isin(np.nonzero(fit_ok)[0])]
tags["c"] = hard[tags["p"].values]
n_c = np.bincount(hard[fit_ok], minlength=K)
tot = tags.groupby("tag").size()
tot = tot[tot >= 200]
cnt = tags[tags["tag"].isin(tot.index)].groupby(["c", "tag"]).size().unstack(fill_value=0)
share = n_c / n_c.sum()
names = np.load(f"{a.brain}/poi_names{a.places}.npz")["name"]
pop = np.bincount(np.concatenate([d["train_p"], d["val_p"]]).astype(np.int64), minlength=len(poi_ids))
old = d["poi_cluster"].astype(np.int64)
report = {"K": K, "m": a.m, "pca_dims": a.pca, "table": rows, "clusters": {}}
print(f"\nclusters (max membership mean {u_all.max(1).mean():.2f}):")
for c in range(K):
    lift = (cnt.loc[c] / tot.loc[cnt.columns]) / share[c] if c in cnt.index else pd.Series(dtype=float)
    top = (lift * np.log(cnt.loc[c].clip(lower=1))).sort_values(ascending=False).head(14).index.tolist() if len(lift) else []
    members = np.nonzero((hard == c) & fit_ok)[0]
    ex = members[np.argsort(-(pop[members] * (u_all[members, c] > 0.6)))[:6]]
    prev = np.bincount(old[members], minlength=6) / max(len(members), 1)
    report["clusters"][str(c)] = {"places": int(n_c[c]), "top_tags": top, "examples": names[ex].tolist(),
                                  "overlap_with_v1_clusters": prev.round(2).tolist()}
    print(f"  {c}: {n_c[c]:6,} places | tags: {', '.join(top)}\n       e.g. {' | '.join(names[ex])}\n       v1 clusters: {prev.round(2).tolist()}")
json.dump(report, open(f"{a.brain}/cluster_report{a.out}.json", "w"), indent=1)
print(f"saved memberships{a.out} [{time.time() - t0:.0f}s]")
