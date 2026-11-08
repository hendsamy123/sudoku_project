"""Train the MLP, calibrate it, compare against logistic regression, save metrics and figures.

Run:  python -m nn.train
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from data.make_data import load_data
from nn.model import MLP, expected_calibration_error, nll

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mlp.npz"


def reliability(probs, y, n_bins=10):
    conf, correct = probs.max(1), (probs.argmax(1) == y)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            xs.append(conf[m].mean()); ys.append(correct[m].mean())
    return xs, ys


def main(seed=0):
    d = load_data()
    ytr, yv, yte = d["y_train"] - 1, d["y_val"] - 1, d["y_test"] - 1
    t0 = time.time()
    net = MLP(seed=seed).fit(d["x_train"], ytr, d["x_val"], yv, seed=seed)
    p_raw = net.predict_proba(d["x_test"], temperature=1.0)
    ece_raw = expected_calibration_error(p_raw, yte)
    T = net.fit_temperature(d["x_val"], yv)
    p_cal = net.predict_proba(d["x_test"])
    net.save(MODEL_PATH)
    print(f"MLP trained in {time.time() - t0:.0f}s, temperature T={T:.3f}")

    print("fitting logistic regression baseline ...")
    lr = LogisticRegression(max_iter=200, C=1.0)
    lr.fit(d["x_train"], ytr)
    p_lr = lr.predict_proba(d["x_test"])

    metrics = {
        "mlp_temp_scaled": {"accuracy": float((p_cal.argmax(1) == yte).mean()), "nll": nll(p_cal, yte),
                            "ece": expected_calibration_error(p_cal, yte), "temperature": T},
        "mlp_uncalibrated": {"accuracy": float((p_raw.argmax(1) == yte).mean()), "nll": nll(p_raw, yte), "ece": ece_raw},
        "logistic_regression": {"accuracy": float((p_lr.argmax(1) == yte).mean()), "nll": nll(p_lr, yte),
                                "ece": expected_calibration_error(p_lr, yte)},
        "n_test": int(len(yte)),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "nn_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    cm = confusion_matrix(yte, p_cal.argmax(1))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].imshow(cm, cmap="Blues")
    ax[0].set_xticks(range(9)); ax[0].set_xticklabels(range(1, 10)); ax[0].set_yticks(range(9)); ax[0].set_yticklabels(range(1, 10))
    ax[0].set_xlabel("predicted"); ax[0].set_ylabel("true"); ax[0].set_title("MLP confusion matrix (test)")
    for i in range(9):
        for j in range(9):
            if cm[i, j]:
                ax[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax[1].plot([0, 1], [0, 1], "k--", label="perfect")
    for name, p in (("MLP uncalibrated", p_raw), ("MLP temp-scaled", p_cal), ("Logistic reg.", p_lr)):
        xs, ys = reliability(p, yte)
        ax[1].plot(xs, ys, "o-", label=name)
    ax[1].set_xlabel("confidence"); ax[1].set_ylabel("accuracy"); ax[1].set_title("Reliability diagram (test)"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "nn_confusion_calibration.png", dpi=130)


if __name__ == "__main__":
    main()
