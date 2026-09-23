import numpy as np
import pytest
from ranking.data import split, matrix, histories
from ranking.evaluation import rank, metrics, make_cases, paired_ci
from ranking.models import negative_samples, item_cf, cold_scores, train_bpr


def test_global_split_boundaries_and_no_future_vocabulary():
    rows = np.array([[1, 10, 5, 0], [1, 20, 2, 9], [2, 30, 5, 10], [1, 40, 5, 20]])
    train, val, test = split(rows, "1970-01-01T00:00:10+00:00", "1970-01-01T00:00:20+00:00")
    assert train[:, 3].tolist() == [0, 9]
    assert val[:, 3].tolist() == [10]
    assert test[:, 3].tolist() == [20]
    users, items, x, seen = matrix(train)
    assert users == [1] and items == [10, 20]
    assert x.tolist() == [[1, 0]] and seen.tolist() == [[True, True]]


def test_all_ratings_consumed_and_candidate_exclusions():
    previous = histories(np.array([[1, 10, 1, 0]]), {10: 0, 20: 1})
    target = np.array([[1, 10, 5, 30], [1, 20, 5, 31], [2, 99, 5, 31]])
    cases, excluded = make_cases(target, previous, {10: 0, 20: 1}, {1: 1}, 4)
    assert len(cases) == 1 and cases[0]["relevant"] == {1}
    assert excluded["positive_targets_for_unseen_items"] == 1
    assert excluded["users_without_eligible_targets"] == 1
    assert excluded["already_consumed_positive_targets"] == 1
    assert rank([100, 1], previous[1], 10).tolist() == [1]


def test_metrics_against_hand_calculation():
    recall, ndcg = metrics([8, 2, 5], {2, 5, 9}, 3)
    assert recall == 2 / 3
    expected = (1 / np.log2(3) + 1 / np.log2(4)) / (1 + 1 / np.log2(3) + 1 / np.log2(4))
    assert ndcg == pytest.approx(expected)
    assert metrics([2], {2}, 10) == (1, 1)
    assert metrics([8], {2}, 10) == (0, 0)
    with pytest.raises(ValueError):
        metrics([], set(), 10)


def test_stable_ties_and_exhausted_candidates():
    assert rank([1, 1, 1], {1}, 10).tolist() == [0, 2]
    assert rank([1, 1], {0, 1}, 10).tolist() == []
    with pytest.raises(ValueError):
        rank([np.nan])


def test_negatives_never_include_consumed_low_ratings():
    consumed = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
    u = np.array([0, 1] * 50)
    result = negative_samples(u, consumed, np.random.default_rng(3))
    assert result.tolist() == [2, 0] * 50
    with pytest.raises(ValueError):
        negative_samples(np.array([0]), np.ones((1, 2), dtype=bool), np.random.default_rng(3))


def test_item_cf_zero_diagonal_and_shrinkage():
    x = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.float32)
    sim = item_cf(x, 0)
    assert sim[0, 1] == pytest.approx(1 / np.sqrt(2))
    assert np.diag(sim).tolist() == [0, 0, 0]
    assert np.isfinite(sim).all()
    assert item_cf(x, 5)[0, 1] < sim[0, 1]


def test_cold_start_fallback_scale_invariance_and_deduplication():
    v = np.array([[2., 0], [0, 3], [4, 4], [0, 0]])
    pop = [3, 7, 1, 0]
    assert cold_scores(v, pop, []).tolist() == pop
    score = cold_scores(v, pop, [0, 0, 1])
    assert score == pytest.approx([.5, .5, 1 / np.sqrt(2), 0])
    assert rank(score, [0, 1]).tolist() == [2, 3]
    with pytest.raises(ValueError):
        cold_scores(v, pop, [99])


def test_paired_bootstrap_identity_and_constant_difference():
    a = np.ones((10, 2)) * .4
    assert paired_ci(a, a, 100)["ndcg"] == {"delta": 0, "low": 0, "high": 0}
    ci = paired_ci(a, a - .1, 100)["recall"]
    assert list(ci.values()) == pytest.approx([.1, .1, .1])


def test_seeded_training_is_reproducible():
    x = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float32)
    cfg = dict(seed=2, learning_rate=.01, regularization=.001, batch_size=4, bpr_epochs=[2])
    a = train_bpr(x, x > 0, cfg, 3, lambda *args: None)
    b = train_bpr(x, x > 0, cfg, 3, lambda *args: None)
    assert np.array_equal(a.item.weight.detach().numpy(), b.item.weight.detach().numpy())
