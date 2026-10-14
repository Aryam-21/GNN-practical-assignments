"""Automated leakage / validity checks (used by the notebook AND the tests)."""
import torch
from data import pair_keys


def _dkeys(e, n):                               # directed key
    return e[0] * n + e[1]


def assert_no_leak(split, mp_ei=None):
    """No held-out link, in either direction, is in the message-passing graph."""
    mp = split.mp_ei if mp_ei is None else mp_ei
    mpk = _dkeys(mp, split.n)
    for name in ("val_pos", "test_pos"):
        e = getattr(split, name)
        assert not torch.isin(_dkeys(e, split.n), mpk).any(), f"{name}: link found in message-passing graph"
        assert not torch.isin(_dkeys(e.flip(0), split.n), mpk).any(), f"{name}: REVERSE link found in message-passing graph"
    return True


def assert_valid_split(split):
    n = split.n
    ktr, kva, kte = (pair_keys(getattr(split, f"{s}_pos"), n) for s in ("train", "val", "test"))
    for name, k in (("train", ktr), ("val", kva), ("test", kte)):
        assert torch.unique(k).numel() == k.numel(), f"duplicate {name} positives"
    assert not torch.isin(kva, ktr).any() and not torch.isin(kte, ktr).any() and not torch.isin(kte, kva).any(), "positive splits overlap"
    for name in ("train_pos", "val_pos", "test_pos"):
        e = getattr(split, name); assert (e[0] < e[1]).all(), f"{name} not stored once per link"
    # negatives
    nva, nte = pair_keys(split.val_neg, n), pair_keys(split.test_neg, n)
    for name, e, k in (("val", split.val_neg, nva), ("test", split.test_neg, nte)):
        assert (e[0] != e[1]).all(), f"{name} negatives contain self-links"
        assert not torch.isin(k, split.all_pos_keys).any(), f"{name} negatives contain observed links"
        assert torch.unique(k).numel() == k.numel(), f"duplicate {name} negatives"
    assert not torch.isin(nva, nte).any(), "val/test negatives overlap"
    assert torch.isin(nva, split.train_forbidden).all() and torch.isin(nte, split.train_forbidden).all(), \
        "training negatives could hit evaluation negatives"
    # message-passing graph = train links, both directions, no self-loops
    mp = split.mp_ei
    assert (mp[0] != mp[1]).all()
    assert torch.equal(_dkeys(mp, n).sort().values, _dkeys(mp.flip(0), n).sort().values), "mp graph not symmetric"
    assert torch.equal(torch.unique(pair_keys(mp, n)), ktr.sort().values), "mp graph != train links"
    assert_no_leak(split)
    return True


@torch.no_grad()
def assert_symmetric_decoder(model, z, pairs, atol=1e-6):
    model.eval()
    assert torch.allclose(model.score(z, pairs), model.score(z, pairs.flip(0)), atol=atol), "decoder not symmetric"
    return True