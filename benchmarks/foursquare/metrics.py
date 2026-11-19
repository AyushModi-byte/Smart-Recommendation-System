"""Next-POI metrics over the full candidate list (GETNext's *_last_timestep functions, one target per trajectory)."""
import numpy as np
import torch

KS = (1, 5, 10, 20)


@torch.no_grad()
def ranks_of(scores, targets):
    """scores (B,P), targets (B,) -> 1-based rank of each target. Ties count as the expected rank, so a scorer
    cannot gain from ties (GETNext's argsort breaks ties by index; identical when scores are distinct)."""
    t = scores[torch.arange(len(targets)), targets]
    gt = (scores > t[:, None]).sum(1).float()
    eq = (scores == t[:, None]).sum(1).float() - 1.0
    return (gt + 1.0 + eq / 2.0).cpu().numpy()


def summarize(ranks):
    m = {"n": int(len(ranks))}
    for k in KS:
        m[f"Acc@{k}"] = float((ranks <= k).mean())
    m["MRR"] = float((1.0 / ranks).mean())
    m["NDCG@10"] = float(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1.0), 0.0).mean())
    return m


def monitor(m):
    """GETNext's model-selection score on validation: 4 * Acc@1 + Acc@20."""
    return 4 * m["Acc@1"] + m["Acc@20"]
