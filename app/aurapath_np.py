"""Numpy-only inference for the trained AuraPath NIL model (no torch needed in the deployed app).
Reproduces training/common.py::NIL (eval mode) and GeoCtx features (feat_v=1: 5 geo features)."""
import numpy as np

R_KM = 6371.0


def _erf(x):  # Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7)
    s = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def gelu(x):
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


class AuraNP:
    def __init__(self, weights):
        w = {k: np.asarray(weights[k], dtype=np.float32) for k in weights.files} if hasattr(weights, "files") else weights
        self.w = w

    def _lin(self, x, name):
        return x @ self.w[name + ".weight"].T + self.w[name + ".bias"]

    def _bn(self, x, name):
        w = self.w
        return (x - w[name + ".running_mean"]) / np.sqrt(w[name + ".running_var"] + 1e-5) * w[name + ".weight"] + w[name + ".bias"]

    def tower(self, x, name):                      # name = "user_tower" | "poi_tower"
        return gelu(self._bn(self._lin(x, name + ".0"), name + ".1"))

    def head(self, ut, pt, g):
        """ut (C,h) or (h,), pt (C,h), g (C,5) -> (C,) logits"""
        ut = np.broadcast_to(ut, pt.shape)
        ge = gelu(self._lin(gelu(self._lin(g, "geo.0")), "geo.2"))
        x = np.concatenate([ut, pt, ut * pt, ge], axis=1)
        x = gelu(self._bn(self._lin(x, "interaction.0"), "interaction.1"))
        x = gelu(self._bn(self._lin(x, "interaction.4"), "interaction.5"))
        x = gelu(self._lin(x, "interaction.8"))
        return self._lin(x, "interaction.10")[:, 0]


def haversine_km(lat1, lon1, lat2, lon2):
    """degrees in, km out; broadcasts"""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def geo_feats(ref, cand, lat, lon, cnt, pop_norm):
    """ref: recent visit POI indices, most recent first (<=30). cand: (C,) candidate indices. -> (C,5) float32"""
    d = haversine_km(lat[cand][:, None], lon[cand][:, None], lat[ref][None, :], lon[ref][None, :])   # (C,M)
    n = d.shape[1]
    return np.stack([np.log1p(d.min(1)) / 5.0, np.log1p(d[:, 0]) / 5.0, (d <= 10.0).sum(1) / n, (d <= 2.0).sum(1) / n,
                     np.log1p(cnt[cand]) / pop_norm], axis=1).astype(np.float32)
