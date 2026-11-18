"""Exercise the user-facing quickstart without downloading a model."""

import os
import re
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_quickstart_trains_reloads_and_searches(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    from train_example import build_tiny_model

    initial = tmp_path / "initial"
    build_tiny_model(tmp_path).save_pretrained(initial)
    output = tmp_path / "run"
    script = tmp_path / "quickstart.py"
    script.write_bytes((EXAMPLES / "quickstart.py").read_bytes())
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "PYTHONIOENCODING": "cp1252",
    }
    result = subprocess.run(
        [sys.executable, str(script), "--model", str(initial), "--output", str(output)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        encoding="cp1252",
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "model" / "modules.json").is_file()
    assert f"Saved model: {output / 'model'}" in result.stdout
    assert "Top result: [" in result.stdout
    original_weights = load_file(initial / "model.safetensors")
    trained_weights = load_file(output / "model" / "model.safetensors")
    assert original_weights.keys() == trained_weights.keys()
    assert any(not torch.equal(original_weights[name], trained_weights[name]) for name in original_weights)
    for metric in ("ndcg", "recall"):
        scores = re.search(rf"^[ \t]*{metric}@3: (\d\.\d{{3}}) -> (\d\.\d{{3}})$", result.stdout, re.MULTILINE)
        assert scores and all(0.0 <= float(score) <= 1.0 for score in scores.groups())
