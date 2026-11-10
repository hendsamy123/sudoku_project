import numpy as np
from nn.model import MLP, expected_calibration_error, softmax


def test_normal_mlp_learns_a_toy_problem_and_roundtrips(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 9, 600)
    X = (rng.normal(size=(9, 784))[y] + 0.5 * rng.normal(size=(600, 784))).astype(np.float32)
    net = MLP(seed=0).fit(X[:500], y[:500], X[500:], y[500:], epochs=15, verbose=False)
    acc = (net.predict_proba(X[500:]).argmax(1) == y[500:]).mean()
    assert acc > 0.9
    net.save(tmp_path / "m.npz")
    net2 = MLP.load(tmp_path / "m.npz")
    assert np.allclose(net.predict_proba(X[:5]), net2.predict_proba(X[:5]), atol=1e-6)


def test_edge_probabilities_sum_to_one_and_temperature_keeps_argmax():
    net = MLP(seed=1)
    X = np.random.default_rng(1).normal(size=(20, 784)).astype(np.float32)
    p1, p2 = net.predict_proba(X, 1.0), net.predict_proba(X, 3.0)
    assert np.allclose(p1.sum(1), 1) and (p1.argmax(1) == p2.argmax(1)).all()


def test_edge_ece_is_zero_for_perfect_confident_predictions():
    p = np.eye(9)[[0, 1, 2]] * 1.0
    assert expected_calibration_error(p, np.array([0, 1, 2])) == 0.0
