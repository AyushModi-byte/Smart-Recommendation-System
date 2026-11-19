"""AuraPath v2 ranker adapted to Foursquare: the NIL two-tower + interaction layer and the GeoCtx features from
training/common.py, with the visual tower and fuzzy aesthetic clusters replaced by POI-category embeddings."""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training"))
from common import NIL, GeoCtx  # noqa: E402  (reused unchanged)

N_DIST = 4  # the first four geo features are distance-based; the fifth is popularity


class GeoNYC(GeoCtx):
    """GeoCtx (v1 features) with explicit reference check-ins and city-scale 'nearby' radii.
    Features: log distance to the nearest reference, log distance to the CURRENT (most recent) check-in,
    share of references within near_km[0] and near_km[1], log training popularity."""

    def __init__(self, ix, stream, device, M, near_km):
        super().__init__(ix.lat, ix.lon, stream.u, stream.p, ix.n_users, device, M=M, v2=False)
        self.near = tuple(float(x) for x in near_km)

    @torch.no_grad()
    def feats_ref(self, r, p, pos=None, no_dist=False):
        """r: (B,M) reference POIs, index 0 = current; p: (B,K) candidates; pos: (B,) training positive or None."""
        d = self.dist_ref(r, p)
        nval = torch.isfinite(d).sum(-1).clamp(min=1).float()
        fa = (d <= self.near[0]).sum(-1).float() / nval
        fb = (d <= self.near[1]).sum(-1).float() / nval
        c = self.cnt[p]
        if pos is not None:
            c = c - (p == pos[:, None]).float()                      # drop the positive's own check-in
        popf = torch.log1p(c.clamp(min=0)) / self.pop_norm
        f = torch.stack([torch.log1p(d.min(-1).values) / 5.0, torch.log1p(d[..., 0]) / 5.0, fa, fb, popf], dim=-1)
        if no_dist:
            f = torch.cat([torch.zeros_like(f[..., :N_DIST]), f[..., N_DIST:]], dim=-1)
        return f


class CatRanker(nn.Module):
    """POI input = category embedding (+ optional frozen text embedding of the category name).
    User input = the mean of the category inputs over the user's training check-ins."""

    def __init__(self, n_cats, cat_dim, hidden, drop, geo_dim, text=None, use_cat=True):
        super().__init__()
        self.emb = nn.Embedding(n_cats, cat_dim)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.register_buffer("text", None if text is None else torch.as_tensor(text, dtype=torch.float32))
        in_dim = cat_dim + (0 if text is None else self.text.shape[1])
        self.use_cat = use_cat
        self.nil = NIL(user_dim=in_dim, poi_dim=in_dim, h=hidden, drop=(drop, drop + 0.1, drop), geo_dim=geo_dim)

    def cat_vec(self):
        v = self.emb.weight if self.text is None else torch.cat([self.emb.weight, self.text], dim=1)
        return v if self.use_cat else torch.zeros_like(v)

    def towers(self, counts, poi_cat):
        """counts: (B, n_cats) user check-ins per category; poi_cat: (...,) -> user (B,h), poi (..., h)."""
        cv = self.cat_vec()
        uin = counts @ cv / counts.sum(1, keepdim=True).clamp(min=1)
        ut = self.nil.user_tower(uin)
        flat = poi_cat.reshape(-1)
        pt = self.nil.poi_tower(cv[flat]).view(*poi_cat.shape, -1)
        return ut, pt

    def score(self, counts, poi_cat, g):
        """counts (B,C), poi_cat (B,K) candidate categories, g (B,K,F) geo features -> (B,K) logits."""
        ut, pt = self.towers(counts, poi_cat)
        return self.nil.head(ut[:, None, :].expand_as(pt), pt, g)


def cat_text_embeddings(names, model_name, cache_dir):
    """Frozen sentence-transformer embedding of each category name (cached)."""
    safe = model_name.replace("/", "_")
    path = os.path.join(cache_dir, f"cat_text_{safe}_{len(names)}.npy")
    if os.path.exists(path):
        return np.load(path)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("--cat_text needs sentence-transformers:  uv pip install sentence-transformers")
    m = SentenceTransformer(model_name, device="cpu")
    e = m.encode(list(names), normalize_embeddings=True, batch_size=64).astype(np.float32)
    np.save(path, e)
    return e
