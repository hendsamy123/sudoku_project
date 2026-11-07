import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
needs_assets = pytest.mark.skipif(not (ROOT / "models" / "mlp.npz").exists() or not (ROOT / "data" / "mnist_1to9.npz").exists(),
                                  reason="run make_data and nn.train first")


@needs_assets
@pytest.mark.parametrize("seed,misreads", [(1, 2), (2, 0), (3, 4), (5, 1)])
def test_normal_demo_prints_every_required_section(capsys, monkeypatch, seed, misreads):
    import run_demo
    monkeypatch.setattr(sys, "argv", ["run_demo.py", "--seed", str(seed), "--misreads", str(misreads), "--no-plot"])
    run_demo.main()
    out = capsys.readouterr().out
    for section in ("ground-truth puzzle", "FIRST READING", "PIPELINE (UCS, NN ON)", "correction log", "NN-GUIDED CORRECTION OFF",
                    "SEARCH STRATEGIES", "CSP PRUNING"):
        assert section in out, section
    assert "Traceback" not in out and ("RESULT: solved" in out or "RESULT: NO SOLUTION" in out)


@needs_assets
def test_edge_demo_rejects_invalid_arguments(monkeypatch):
    import run_demo
    monkeypatch.setattr(sys, "argv", ["run_demo.py", "--misreads", "9"])
    with pytest.raises(SystemExit):
        run_demo.main()
