"""
Leakage test for the Foursquare benchmark: no validation / test check-in may reach a training-derived feature.

  python -m benchmarks.foursquare.test_leakage        (or: pytest benchmarks/foursquare/test_leakage.py)
"""
import numpy as np
import torch

from .data import Index, TrainStream, eval_refs, eval_samples, load_splits, user_cat_counts
from .model import GeoNYC

D = load_splits()
ix = Index(D["train"])
st = TrainStream(D["train"], ix)
M = 30


def _keys(df):
    return set(zip(df["user_id"], df["POI_id"], df["UTC_time"]))


def test_feature_source_is_train_only():
    """Every row behind popularity, user taste, transitions and training references is a training check-in."""
    train_keys = _keys(D["train"])
    assert set(st.key) <= train_keys
    for s in ("val", "test"):
        assert not (set(st.key) & _keys(D[s])), f"{s} check-ins appear in the training feature source"


def test_test_timestamps_not_in_training_features():
    """No test (or validation) check-in time appears in, or precedes, the training feature rows."""
    for s in ("val", "test"):
        assert st.t.max() < D[s]["t"].min()
        assert not (set(st.t.tolist()) & set(D[s]["t"].tolist()))


def test_counts_are_train_only():
    geo = GeoNYC(ix, st, torch.device("cpu"), M, (2.0, 0.5))
    assert int(geo.cnt.sum()) == len(D["train"])                     # popularity = training check-ins only
    C = user_cat_counts(st, ix)
    assert int(C.sum()) == len(D["train"])                           # user taste = training check-ins only
    assert len(st.pairs) <= len(D["train"])                          # Markov pairs come from the stream itself


def test_eval_inputs_precede_target():
    """Evaluation references are the trajectory prefix plus training history, all strictly before the target,
    and the target check-in itself is never an input."""
    for s in ("val", "test"):
        S = eval_samples(D[s], ix)
        ref, ref_t = eval_refs(S, st, ix.n_users, M)
        assert ((ref_t < S["target_t"][:, None]) | (ref < 0)).all()
        assert all(len(p) >= 2 for p in S["prefix"])
        assert not (set(S["key"]) & set(st.key))


def test_training_refs_precede_positive():
    """During training, a positive's references are earlier check-ins of the same user, never the positive."""
    geo = GeoNYC(ix, st, torch.device("cpu"), M, (2.0, 0.5))
    rows = st.positives
    h = geo.hist[torch.as_tensor(rows)].numpy()
    for j in range(M):
        ok = h[:, j] >= 0
        src = rows[ok] - 1 - j
        assert (st.u[src] == st.u[rows[ok]]).all()
        assert (st.t[src] <= st.t[rows[ok]]).all()
        assert (st.p[src] == h[ok, j]).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    print("all leakage checks passed")
