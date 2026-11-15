"""AuraPath numpy inference core (no torch). Mirrors NeuralInteractionLayer from NB10/NB12 in eval mode."""
import numpy as np

def _erf(x):  # Abramowitz-Stegun 7.1.26, max err 1.5e-7
    s = np.sign(x); x = np.abs(x); t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y

def gelu(x): return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def _fold(sd, lin, bn, eps=1e-5):
    W, b = sd[f"{lin}.weight"], sd[f"{lin}.bias"]
    g, be, m, v = (sd[f"{bn}.{k}"] for k in ("weight", "bias", "running_mean", "running_var"))
    s = g / np.sqrt(v + eps)
    return (W * s[:, None]).T.astype(np.float32), ((b - m) * s + be).astype(np.float32)

def fold_state_dict(sd):
    """BatchNorm folded into the preceding Linear; returns dict of arrays for npz."""
    out = {}
    for name, (lin, bn) in {"u": ("user_tower.0", "user_tower.1"), "p": ("poi_tower.0", "poi_tower.1"),
                            "i1": ("interaction.0", "interaction.1"), "i2": ("interaction.4", "interaction.5")}.items():
        out[f"{name}_W"], out[f"{name}_b"] = _fold(sd, lin, bn)
    out["i3_W"], out["i3_b"] = sd["interaction.8.weight"].T.astype(np.float32), sd["interaction.8.bias"]
    out["i4_W"], out["i4_b"] = sd["interaction.10.weight"].T.astype(np.float32), sd["interaction.10.bias"]
    return out

class NIL:
    def __init__(self, w): self.w = w
    def score(self, u, P):
        """u: (262,) user emb; P: (n,262) poi embs -> (n,) visit probability."""
        w = self.w
        ut = gelu(u[None, :] @ w["u_W"] + w["u_b"])
        pt = gelu(P @ w["p_W"] + w["p_b"])
        x = np.concatenate([np.broadcast_to(ut, pt.shape), pt, ut * pt], axis=1)
        x = gelu(x @ w["i1_W"] + w["i1_b"]); x = gelu(x @ w["i2_W"] + w["i2_b"])
        x = gelu(x @ w["i3_W"] + w["i3_b"])
        return sigmoid(x @ w["i4_W"] + w["i4_b"]).ravel()
