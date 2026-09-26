"""
NNTQ variants for the metamodel Bayesian search (run.build_NNTQ_variants):
N+2 trainings with fixed seeds (N = number of meta runs), best and worst
dropped, the middle N cached, variant 0 being the median. Training is faked.
"""
import json
import os
import pickle

import pytest

torch = pytest.importorskip("torch", reason="run imports torch")

import run

# loss per seed (seeds SEED_NNTQ_VARIANTS + k): deliberately not sorted
LOSSES = [25., 17., 31., 22., 40., 19., 28., 35., 12.]


class _FakeBundle(dict):
    """Stands for the DatasetBundle: what is pickled is `for_cache()`."""
    def for_cache(self):
        return dict(self, slimmed=True)


def _build(tmp_path, monkeypatch, num_variants):
    monkeypatch.setattr(run, "loss_NNTQ", lambda qdc, worst: worst)
    trained = []

    def fake_train():
        seed = torch.initial_seed()                  # set by build_NNTQ_variants
        trained.append(seed)
        return (_FakeBundle(seed=seed), None, LOSSES[seed - run.SEED_NNTQ_VARIANTS])

    paths, _ = run.NNTQ_variants_paths(str(tmp_path), "key", num_variants)
    run.build_NNTQ_variants(fake_train, paths, str(tmp_path), "key")
    return paths, trained


def _loss(path):
    with open(path, "rb") as f:
        _, _, loss = pickle.load(f)
    return loss


@pytest.mark.parametrize("num_variants, expected", [
    (5, [25., 22., 28., 19., 31.]),   # 7 trainings: median, then alternately
    (3, [25., 22., 31.]),             # 5 trainings
    (1, [25.]),                       # 3 trainings
])
def test_variants_are_the_middle_median_first(tmp_path, monkeypatch,
                                              num_variants, expected):
    paths, trained = _build(tmp_path, monkeypatch, num_variants)
    assert trained == [run.SEED_NNTQ_VARIANTS + k for k in range(num_variants + 2)]
    assert [_loss(p) for p in paths] == expected
    assert sorted(expected) == sorted(LOSSES[:num_variants + 2])[1:-1]


def test_only_variants_and_summary_left(tmp_path, monkeypatch):
    paths, _ = _build(tmp_path, monkeypatch, 5)
    assert sorted(os.listdir(tmp_path)) == sorted(
        [os.path.basename(p) for p in paths] + ["NNTQ_variants_key.json"])
    with open(tmp_path / "NNTQ_variants_key.json") as f:
        summary = json.load(f)
    assert [d["loss_NNTQ"] for d in summary["variants"]] == [25., 22., 28., 19., 31.]
    assert len(summary["all"]) == 7


def test_cache_valid_only_for_the_same_number_of_variants(tmp_path, monkeypatch):
    assert not run.NNTQ_variants_cached(str(tmp_path), "key", 5)
    _build(tmp_path, monkeypatch, 5)
    assert     run.NNTQ_variants_cached(str(tmp_path), "key", 5)
    assert not run.NNTQ_variants_cached(str(tmp_path), "key", 3)  # other median
    assert not run.NNTQ_variants_cached(str(tmp_path), "key", 6)  # v5 missing


def test_run_model_once_default_uses_no_variant():
    """Other modes are unchanged (the per-run variant of the meta search is
    tested in test_search_objective.py)."""
    import inspect
    params = inspect.signature(run.run_model_once).parameters
    assert params['NNTQ_variant'].default is None


def test_rebuild_with_fewer_variants_removes_obsolete_ones(tmp_path, monkeypatch):
    """A rebuild with fewer variants (smaller num_runs) deletes the extra
    `_v{k}` files of the earlier build, for the same key only."""
    _build(tmp_path, monkeypatch, 5)
    other = tmp_path / "NNTQ_preds_otherkey_v4.pkl"
    other.write_bytes(b"x")                                  # another key: kept
    paths, _ = _build(tmp_path, monkeypatch, 3)
    files = sorted(f for f in os.listdir(tmp_path) if f.endswith(".pkl"))
    assert files == sorted([os.path.basename(p) for p in paths] + [other.name])


def test_variants_are_pickled_slimmed(tmp_path, monkeypatch):
    """What is cached is the bundle's for_cache() copy (no torch loaders)."""
    paths, _ = _build(tmp_path, monkeypatch, 1)
    with open(paths[0], "rb") as f:
        bundle, _, _ = pickle.load(f)
    assert bundle.get("slimmed") is True
