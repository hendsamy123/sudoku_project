"""Small MLP (784-128-64-9) written in plain NumPy, with Adam, early stopping and temperature scaling.

Classes are digits 1..9 mapped to indices 0..8.  `predict_proba` returns an (N, 9) matrix.
"""
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs, labels_idx, n_bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == labels_idx).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def nll(probs, labels_idx):
    return float(-np.log(np.clip(probs[np.arange(len(labels_idx)), labels_idx], 1e-12, 1)).mean())


class MLP:
    def __init__(self, sizes=(784, 128, 64, 9), seed=0):
        self.sizes = tuple(sizes)
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.temperature = 1.0

    # ---- forward / backward -------------------------------------------------
    def _forward(self, X):
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(np.maximum(z, 0) if i < len(self.W) - 1 else z)
        return acts

    def logits(self, X):
        return self._forward(X)[-1]

    def predict_proba(self, X, temperature=None):
        T = self.temperature if temperature is None else temperature
        return softmax(self.logits(X) / T)

    def fit(self, X, y_idx, Xv, yv_idx, epochs=20, batch=128, lr=1e-3, wd=1e-4, seed=0, verbose=True):
        rng = np.random.default_rng(seed)
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t, best, best_params, bad = 0, np.inf, None, 0
        n = len(X)
        for ep in range(epochs):
            perm = rng.permutation(n)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                acts = self._forward(X[idx])
                p = softmax(acts[-1])
                d = p
                d[np.arange(len(idx)), y_idx[idx]] -= 1
                d /= len(idx)
                grads_W, grads_b = [None] * len(self.W), [None] * len(self.W)
                for i in range(len(self.W) - 1, -1, -1):
                    grads_W[i] = acts[i].T @ d + wd * self.W[i]
                    grads_b[i] = d.sum(0)
                    if i > 0:
                        d = (d @ self.W[i].T) * (acts[i] > 0)
                grads = grads_W + grads_b
                t += 1
                for k, (p_, g) in enumerate(zip(params, grads)):
                    m[k] = 0.9 * m[k] + 0.1 * g
                    v[k] = 0.999 * v[k] + 0.001 * g * g
                    p_ -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            pv = softmax(self.logits(Xv))
            vl, va = nll(pv, yv_idx), float((pv.argmax(1) == yv_idx).mean())
            if verbose:
                print(f"epoch {ep + 1:2d}  val_nll={vl:.4f}  val_acc={va:.4f}")
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_params = [p_.copy() for p_ in params]
            else:
                bad += 1
                if bad >= 3:
                    break
        if best_params is not None:
            for p_, bp in zip(params, best_params):
                p_[...] = bp
        return self

    def fit_temperature(self, Xv, yv_idx):
        """Temperature scaling: pick T minimising validation NLL (does not change the argmax)."""
        z = self.logits(Xv)
        res = minimize_scalar(lambda T: nll(softmax(z / T), yv_idx), bounds=(0.3, 5.0), method="bounded")
        self.temperature = float(res.x)
        return self.temperature

    # ---- persistence --------------------------------------------------------
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, sizes=np.array(self.sizes), T=self.temperature,
                 **{f"W{i}": w for i, w in enumerate(self.W)}, **{f"b{i}": b for i, b in enumerate(self.b)})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        net = cls(tuple(int(s) for s in d["sizes"]))
        net.W = [d[f"W{i}"] for i in range(len(net.W))]
        net.b = [d[f"b{i}"] for i in range(len(net.b))]
        net.temperature = float(d["T"])
        return net
