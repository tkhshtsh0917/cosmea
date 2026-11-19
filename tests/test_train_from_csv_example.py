"""Exercise the CSV training CLI with a local checkpoint on CPU and Apple MPS."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from sentence_transformers import SentenceTransformer

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def cli_args(tmp_path):
    return [
        "--model",
        str(tmp_path / "initial"),
        "--train",
        str(tmp_path / "train.csv"),
        "--corpus",
        str(tmp_path / "corpus.csv"),
        "--output",
        str(tmp_path / "run"),
    ]


@pytest.mark.parametrize(
    "device,available,visible,expected",
    [
        ("mps", False, 0, "requires an available PyTorch MPS device"),
        ("cuda", False, 0, "requires an available CUDA GPU"),
        ("cuda", True, 2, "select a GPU with CUDA_VISIBLE_DEVICES"),
    ],
)
def test_cli_rejects_unavailable_or_multiple_gpus(monkeypatch, capsys, cli_args, device, available, visible, expected):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    import train_from_csv

    backend = torch.backends.mps if device == "mps" else torch.cuda
    monkeypatch.setattr(backend, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: visible)
    monkeypatch.setattr(sys, "argv", ["train_from_csv.py", *cli_args, "--device", device])

    with pytest.raises(SystemExit) as error:
        train_from_csv.main()

    assert error.value.code == 2
    assert expected in capsys.readouterr().err


@pytest.mark.parametrize(
    "device",
    [None, pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS"))],
    ids=["cpu-default", "mps"],
)
def test_csv_training_saves_an_updated_model(tmp_path, monkeypatch, cli_args, device):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    from train_example import build_tiny_model

    initial = tmp_path / "initial"
    build_tiny_model(tmp_path).save_pretrained(initial)
    (tmp_path / "train.csv").write_text(
        "anchor,positive,query_id,document_id\n"
        "query nurse,job nurse,q1,d1\n"
        "query nurse,job teacher,q1,d2\n"
        "query driver,job driver,q2,d3\n",
        encoding="utf-8",
    )
    (tmp_path / "corpus.csv").write_text(
        "document_id,text\nd1,job nurse\nd2,job teacher\nd3,job driver\nd4,job chef\n",
        encoding="utf-8",
    )
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"}
    environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / "train_from_csv.py"),
            *cli_args,
            "--batch-size",
            "3",
            "--uniform",
            "2",
            *(["--device", device] if device else []),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )

    expected_device = device or "cpu"
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Training device: {expected_device}" in result.stdout
    model_dir = tmp_path / "run" / "model"
    assert f"Saved model: {model_dir}" in result.stdout
    before = load_file(initial / "model.safetensors")
    after = load_file(model_dir / "model.safetensors")
    assert before.keys() == after.keys()
    assert any(not torch.equal(before[name], after[name]) for name in before)
    model = SentenceTransformer(str(model_dir), device=expected_device)
    embedding = model.encode(["query nurse"], convert_to_tensor=True)
    assert embedding.device.type == expected_device
    assert torch.isfinite(embedding).all()
