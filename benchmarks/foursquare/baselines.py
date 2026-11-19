"""Non-learned baselines on the same samples: popularity, nearest place, popular within a radius, Markov chain.
All statistics come from the training file. 'Current' = the last check-in of the trajectory prefix."""
import numpy as np
import torch


def _dist_to_current(geo, ref, n_pois):
    p = torch.arange(n_pois, device=geo.dev)[None].expand(len(ref), -1)
    return geo.dist_ref(ref[:, :1], p)[..., 0]                         # (B, P) km from the current check-in


def popularity(geo, ref, n_pois):
    return geo.cnt[None].expand(len(ref), -1).clone()


def nearest(geo, ref, n_pois):
    return -_dist_to_current(geo, ref, n_pois)


def popular_within(radius_km):
    """GeoCtx-style 'popular within R': popular places within R km of the current check-in first, the rest by
    proximity (the same rule as training/common.py geopop_scorer, centred on the current check-in)."""
    def f(geo, ref, n_pois):
        d = _dist_to_current(geo, ref, n_pois)
        lp = torch.log1p(geo.cnt)[None]
        return torch.where(d <= radius_km, 1e3 + lp - 1e-3 * d, -d)
    return f


class UserFrequency:
    """The user's own check-in counts: training history plus the current trajectory prefix; ties broken by
    popularity. No learning. Measures how much of the benchmark is revisits."""

    def __init__(self, stream, n_users, n_pois, samples):
        self.H = np.zeros((n_users, n_pois), dtype=np.float32)
        np.add.at(self.H, (stream.u, stream.p), 1.0)
        self.S = samples                                                  # {split: eval_samples(...)}

    def __call__(self, geo, ref, n_pois, ctx):
        split, i, j = ctx
        u = self.S[split]["u"][i:j]
        s = torch.as_tensor(self.H[u], device=geo.dev).clone()
        for r, pre in enumerate(self.S[split]["prefix"][i:j]):
            s[r] += torch.bincount(torch.as_tensor(pre, device=geo.dev), minlength=n_pois).float()
        return s + 1e-3 * geo.cnt[None] / (geo.cnt.max() + 1.0)


class Markov:
    """First-order POI -> POI transition counts from consecutive check-ins inside training trajectories;
    unseen transitions fall back to popularity (as a tie-breaker scaled far below one transition)."""

    def __init__(self, pairs, n_pois):
        key, cnt = np.unique(pairs[:, 0] * n_pois + pairs[:, 1], return_counts=True)
        self.src, self.dst, self.cnt = key // n_pois, key % n_pois, cnt.astype(np.float32)
        self.start = np.searchsorted(self.src, np.arange(n_pois + 1))
        self.n = n_pois

    def __call__(self, geo, ref, n_pois):
        pop = geo.cnt / (geo.cnt.max() + 1.0)
        s = pop[None].expand(len(ref), -1).clone() * 1e-3
        cur = ref[:, 0].cpu().numpy()
        for i, c in enumerate(cur):
            a, b = self.start[c], self.start[c + 1]
            if b > a:
                s[i, torch.as_tensor(self.dst[a:b], device=s.device)] += torch.as_tensor(self.cnt[a:b], device=s.device)
        return s
