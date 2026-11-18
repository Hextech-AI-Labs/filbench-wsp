from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

from filbench_wsp.lighteval_tasks import (
    TATOEBA_PARQUET_REVISION,
    build_patched_tasks_table,
)


def test_build_patched_tasks_table_patches_kalahi_and_tatoeba() -> None:
    mock_kalahi = SimpleNamespace(
        name="kalahi_tgl_mcf",
        hf_repo="aisingapore/cultural_evaluation-kalahi",
        hf_subset="default",
        hf_avail_splits=["train", "validation", "test"],
        evaluation_splits=["tl"],
        hf_revision=None,
    )
    mock_tatoeba = SimpleNamespace(
        name="tatoeba_tgl",
        hf_repo="Helsinki-NLP/tatoeba_mt",
        hf_subset="eng-tgl",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        hf_revision=None,
    )
    mock_other = SimpleNamespace(
        name="other_task",
        hf_repo="other/repo",
        hf_subset="default",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        hf_revision=None,
    )

    upstream = [mock_kalahi, mock_tatoeba, mock_other]
    patched = build_patched_tasks_table(upstream)  # type: ignore[arg-type]

    # Verify original objects were not mutated
    assert mock_kalahi.hf_subset == "default"
    assert mock_kalahi.evaluation_splits == ["tl"]
    assert mock_tatoeba.hf_revision is None

    # Verify patched kalahi
    patched_kalahi = patched[0]
    assert patched_kalahi.hf_subset == "tl"
    assert patched_kalahi.hf_avail_splits == ["eval"]
    assert patched_kalahi.evaluation_splits == ["eval"]

    # Verify patched tatoeba
    patched_tatoeba = patched[1]
    assert patched_tatoeba.hf_revision == TATOEBA_PARQUET_REVISION

    # Verify other task untouched
    patched_other = patched[2]
    assert patched_other.hf_repo == "other/repo"
    assert patched_other.hf_revision is None


def test_ensure_hf_token_propagates_from_hub() -> None:
    import importlib.util
    from pathlib import Path

    p02 = Path(__file__).resolve().parent.parent / "scripts" / "02_prepare_splits.py"
    spec02 = importlib.util.spec_from_file_location("mod02", p02)
    assert spec02 and spec02.loader
    mod02 = importlib.util.module_from_spec(spec02)
    spec02.loader.exec_module(mod02)

    with patch.dict(os.environ, {}, clear=True):
        with patch("huggingface_hub.get_token", return_value="fake_hub_token"):
            token = mod02.ensure_hf_token()
            assert token == "fake_hub_token"
            assert os.environ.get("HF_TOKEN") == "fake_hub_token"

