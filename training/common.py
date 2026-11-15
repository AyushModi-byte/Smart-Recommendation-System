"""Shared model + evaluation code for AuraPath v2 (train/eval scripts import this)."""
import numpy as np
import torch
import torch.nn as nn

KS = (5, 10, 20, 50)
R_KM = 6371.0


class GeoCtx:
    """Distance / density / popularity features for (user, candidate POI) pairs, from TRAIN visits only.
    Eval: reference visits = the user's last M train visits.
    Training: reference visits = the M visits strictly BEFORE the positive (never the positive itself),
    and the popularity feature excludes the positive's own visit -> no train/eval mismatch."""

    MC = 10          # recent visits used for co-visit lookups
    W = 60           # co-visit table uses each user's last W train visits

    def __init__(self, poi_lat, poi_lon, tr_u, tr_p, n_users, device, M=30, pop=None, v2=False):
        self.dev = device
        self.v2 = v2
        self.n_feat = 9 if v2 else 5
        assert (np.diff(tr_u) >= 0).all(), "train arrays must be grouped by user and chronological"
        cnt = np.bincount(tr_p, minlength=len(poi_lat)) if pop is None else np.asarray(pop)
        self.cnt = torch.as_tensor(cnt, dtype=torch.float32, device=device)
        self.pop_norm = float(np.log1p(cnt.max()))
        lat = np.radians(np.asarray(poi_lat, dtype=np.float64))
        lon = np.radians(np.asarray(poi_lon, dtype=np.float64))
        self.lat = torch.as_tensor(lat, dtype=torch.float32, device=device)
        self.lon = torch.as_tensor(lon, dtype=torch.float32, device=device)
        self.coslat = torch.cos(self.lat)
        counts = np.bincount(tr_u, minlength=n_users)
        indptr = np.concatenate([[0], np.cumsum(counts)])
        pos_in_user = np.arange(len(tr_u)) - indptr[tr_u]
        # eval refs: last M train visits of each user, index 0 = most recent
        from_end = counts[tr_u] - 1 - pos_in_user
        keep = from_end < M
        ref = -np.ones((n_users, M), dtype=np.int64)
        ref[tr_u[keep], from_end[keep]] = tr_p[keep]
        self.ref = torch.as_tensor(ref, device=device)
        # training refs: the M visits before each train visit, index 0 = immediately before
        hist = -np.ones((len(tr_u), M), dtype=np.int64)
        for j in range(M):
            ok = np.nonzero(pos_in_user > j)[0]
            hist[ok, j] = tr_p[ok - 1 - j]
        self.hist = torch.as_tensor(hist, device=device)
        self.has_hist = pos_in_user > 0
        if v2:
            self._init_v2(poi_lat, poi_lon, lat, lon, tr_u, tr_p, counts, indptr, pos_in_user, cnt, device)

    def _init_v2(self, poi_lat, poi_lon, lat, lon, tr_u, tr_p, counts, indptr, pos_in_user, cnt, device):
        n_p = len(poi_lat)
        self.P = n_p
        # --- travel propensity: share of past transitions that jumped > 50 km (only visits BEFORE the row) ---
        prev = np.concatenate([[0], tr_p[:-1]])
        a = np.sin((lat[tr_p] - lat[prev]) / 2) ** 2 + np.cos(lat[tr_p]) * np.cos(lat[prev]) * np.sin((lon[tr_p] - lon[prev]) / 2) ** 2
        dkm = 2 * R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        far = ((dkm > 50.0) & (pos_in_user > 0)).astype(np.int64)
        cfp = np.concatenate([[0], np.cumsum(far)])
        before = cfp[np.arange(len(tr_u))] - cfp[indptr[tr_u]]
        self.trav_row = torch.as_tensor((before + 0.5) / (np.maximum(pos_in_user - 1, 0) + 2.0), dtype=torch.float32, device=device)
        tot = cfp[indptr[1:]] - cfp[indptr[:-1]]
        self.trav_user = torch.as_tensor((tot + 0.5) / (np.maximum(counts - 1, 0) + 2.0), dtype=torch.float32, device=device)
        self.row_fe = torch.as_tensor(counts[tr_u] - 1 - pos_in_user, device=device)
        # --- area popularity: visits in the POI's ~55 km cell ---
        dlat, dlon = np.asarray(poi_lat, dtype=np.float64), np.asarray(poi_lon, dtype=np.float64)
        cell = np.floor((dlat + 90) / 0.5).astype(np.int64) * 100000 + np.floor((dlon + 180) / 0.5).astype(np.int64)
        cell = np.unique(cell, return_inverse=True)[1]
        cell_tot = np.bincount(cell, weights=np.asarray(cnt, dtype=np.float64))
        self.cell = torch.as_tensor(cell, device=device)
        self.cell_tot = torch.as_tensor(cell_tot, dtype=torch.float32, device=device)
        self.area_norm = float(np.log1p(cell_tot.max()))
        # --- co-visit table: number of users that visited both a and b (last W train visits per user) ---
        keys = []
        for u in range(len(counts)):
            s, e = indptr[u], indptr[u + 1]
            v = tr_p[max(s, e - self.W):e]
            n = len(v)
            if n < 2:
                continue
            x, y = np.repeat(v, n), np.tile(v, n)
            m = x != y
            keys.append(x[m] * n_p + y[m])
        uk, ct = np.unique(np.concatenate(keys), return_counts=True)
        self.ck = torch.as_tensor(uk, device=device)
        self.cv = torch.as_tensor(ct, dtype=torch.float32, device=device)

    @torch.no_grad()
    def dist_ref(self, r, p):
        """r: (B,M) reference poi idx (-1 padding), p: (B,K) candidate idx -> (B,K,M) km (inf for padding)."""
        valid = r >= 0
        rc = r.clamp(min=0)
        rl, rn, rcos = self.lat[rc][:, None, :], self.lon[rc][:, None, :], self.coslat[rc][:, None, :]
        pl, pn, pcos = self.lat[p][:, :, None], self.lon[p][:, :, None], self.coslat[p][:, :, None]
        a = torch.sin((pl - rl) / 2) ** 2 + pcos * rcos * torch.sin((pn - rn) / 2) ** 2
        d = 2 * R_KM * torch.asin(torch.sqrt(a.clamp(0, 1)))
        return d.masked_fill(~valid[:, None, :], float("inf"))

    def dist(self, u, p):
        return self.dist_ref(self.ref[u], p)

    @torch.no_grad()
    def feats(self, u, p, rows=None, pos=None):
        """u: (B,), p: (B,K). Training: rows=(B,) train-visit row ids and pos=(B,) positive poi ids."""
        r = self.hist[rows] if rows is not None else self.ref[u]
        d = self.dist_ref(r, p)                                       # (B,K,M)
        nval = torch.isfinite(d).sum(-1).clamp(min=1).float()
        f10 = (d <= 10.0).sum(-1).float() / nval                      # share of recent visits within 10 km
        f2 = (d <= 2.0).sum(-1).float() / nval                        # ... within 2 km
        c = self.cnt[p]
        if pos is not None:
            c = c - (p == pos[:, None]).float()                       # drop the positive's own visit
        popf = torch.log1p(c.clamp(min=0)) / self.pop_norm
        base = torch.stack([torch.log1p(d.min(-1).values) / 5.0, torch.log1p(d[..., 0]) / 5.0, f10, f2, popf], dim=-1)
        if not self.v2:
            return base
        B, K = p.shape
        trav = (self.trav_row[rows] if rows is not None else self.trav_user[u])[:, None].expand(B, K)
        ca = self.cell_tot[self.cell[p]]
        if pos is not None:
            ca = ca - (self.cell[p] == self.cell[pos][:, None]).float()   # drop the positive's own visit
        area = torch.log1p(ca.clamp(min=0)) / self.area_norm
        rc = r[:, :self.MC]
        valid = (rc >= 0)[:, None, :]
        q = (rc[:, None, :] * self.P + p[:, :, None]).reshape(-1)
        idx = torch.searchsorted(self.ck, q).clamp(max=len(self.ck) - 1)
        val = torch.where(self.ck[idx] == q, self.cv[idx], torch.zeros_like(self.cv[idx])).view(B, K, -1)
        if pos is not None:                                                # remove this user's own contribution
            fe = self.row_fe[rows][:, None, None] + torch.arange(1, rc.shape[1] + 1, device=p.device)[None, None, :]
            val = val - ((p == pos[:, None])[:, :, None] & (fe < self.W) & valid).float()
        val = val.clamp(min=0) * valid.float()
        cs = torch.log1p(val.sum(-1)) / 5.0
        cm = torch.log1p(val.max(-1).values) / 5.0
        return torch.cat([base, torch.stack([trav, area, cs, cm], dim=-1)], dim=-1)

    @torch.no_grad()
    def dmin_all(self, u, n_pois, chunk=32768):
        """(B, n_pois) distance in km from each POI to the user's nearest recent train visit."""
        B = len(u)
        outs = []
        for i in range(0, n_pois, chunk):
            idx = torch.arange(i, min(i + chunk, n_pois), device=self.dev)
            outs.append(self.dist(u, idx[None].expand(B, -1)).min(-1).values)
        return torch.cat(outs, dim=1)


class NIL(nn.Module):
    """v1 NeuralInteractionLayer + a small geo branch. Outputs LOGITS (apply sigmoid for probability)."""

    def __init__(self, user_dim=262, poi_dim=262, h=128, drop=(0.2, 0.3, 0.2), geo_dim=9, geo_h=32):
        super().__init__()
        self.user_tower = nn.Sequential(nn.Linear(user_dim, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop[0]))
        self.poi_tower = nn.Sequential(nn.Linear(poi_dim, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop[0]))
        self.geo = nn.Sequential(nn.Linear(geo_dim, geo_h), nn.GELU(), nn.Linear(geo_h, geo_h), nn.GELU())
        self.interaction = nn.Sequential(
            nn.Linear(h * 3 + geo_h, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(drop[1]),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(drop[2]),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
        )
        self.h = h

    def head(self, ut, pt, g):
        """ut, pt: (..., h); g: (..., geo_dim) -> (...,) logits"""
        x = torch.cat([ut, pt, ut * pt, self.geo(g)], dim=-1)
        shp = x.shape[:-1]
        return self.interaction(x.reshape(-1, x.shape[-1])).view(shp)


@torch.no_grad()
def score_all(model, ut, pt, geo, ub, chunk=32768):
    """ut: (B,h) user tower outputs, pt: (P,h) poi tower outputs, ub: (B,) user idx -> (B,P) logits."""
    B, P = ut.shape[0], pt.shape[0]
    outs = []
    for i in range(0, P, chunk):
        idx = torch.arange(i, min(i + chunk, P), device=ut.device)
        C = len(idx)
        g = geo.feats(ub, idx[None].expand(B, C))
        u = ut[:, None, :].expand(B, C, -1)
        pp = pt[idx][None].expand(B, C, -1)
        outs.append(model.head(u, pp, g))
    return torch.cat(outs, dim=1)


def build_csr(u, p, n_users):
    order = np.argsort(u, kind="stable")
    u_s, p_s = u[order], p[order]
    indptr = np.zeros(n_users + 1, dtype=np.int64)
    np.add.at(indptr, u_s + 1, 1)
    return np.cumsum(indptr), p_s.astype(np.int64)


@torch.no_grad()
def evaluate(scorer, users, targets, exclude_csrs, device, bs=8, ks=KS, invalid=None):
    """Rank the single held-out POI of each user against ALL POIs (no candidate-pool tricks).
    scorer(user_idx_tensor)->(B,P) scores; exclude_csrs: list of (indptr, indices) of POIs to mask out.
    Ties are resolved by expected rank (so constant scores cannot look good). Returns (metrics, ranks)."""
    ranks = []
    for s in range(0, len(users), bs):
        ub = users[s:s + bs]
        tb = targets[s:s + bs]
        sc = scorer(torch.as_tensor(ub, device=device)).float()
        rows, cols = [], []
        for r, u in enumerate(ub):
            for indptr, idx in exclude_csrs:
                seg = idx[indptr[u]:indptr[u + 1]]
                rows.append(np.full(len(seg), r))
                cols.append(seg)
        rows = torch.as_tensor(np.concatenate(rows), device=device)
        cols = torch.as_tensor(np.concatenate(cols), device=device)
        sc[rows, cols] = float("-inf")
        if invalid is not None:
            sc[:, invalid] = float("-inf")        # POIs that did not exist yet (no photos before the hold-out)
        tt = torch.as_tensor(tb, device=device).long()
        tgt = sc[torch.arange(len(ub), device=device), tt]
        gt = (sc > tgt[:, None]).sum(1).float()
        eq = (sc == tgt[:, None]).sum(1).float() - 1.0
        ranks.append((gt + 1.0 + eq / 2.0).cpu().numpy())
    ranks = np.concatenate(ranks)
    return summarize(ranks, ks), ranks


def summarize(ranks, ks=KS):
    m = {"n_users": int(len(ranks)), "MRR(=MAP)": float((1.0 / ranks).mean())}
    for k in ks:
        hit = ranks <= k
        m[f"HR@{k}(=Recall@{k})"] = float(hit.mean())
        m[f"Precision@{k}"] = float(hit.mean() / k)
        m[f"NDCG@{k}"] = float(np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0).mean())
    m["median_rank"] = float(np.median(ranks))
    return m


# ---------- non-learned baselines (same protocol) ----------
def popularity_scorer(pop, n_users_dummy=None):
    pop_t = torch.as_tensor(pop, dtype=torch.float32)
    return lambda ub: pop_t.to(ub.device)[None].expand(len(ub), -1).clone()


def nearest_scorer(geo, n_pois):
    return lambda ub: -geo.dmin_all(ub, n_pois)


def geopop_scorer(geo, pop, n_pois, radius_km=25.0):
    """Most popular POIs within radius_km of the user's recent visits; farther POIs ranked by proximity."""
    pop_t = torch.log1p(torch.as_tensor(pop, dtype=torch.float32, device=geo.dev))

    def f(ub):
        d = geo.dmin_all(ub, n_pois)
        return torch.where(d <= radius_km, 1e3 + pop_t[None] - 1e-3 * d, -d)
    return f
