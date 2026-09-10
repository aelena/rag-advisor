"""Tests for the speaker_split and row_based evaluation chunkers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from rag_adviser.evaluators.chunking_strategies import (
    _detect_delimiter,
    _split_turns,
    chunk_rows,
    chunk_speaker_turns,
)
from rag_adviser.evaluators.pipeline_runner import EvalConfig, EvalPipelineRunner
from rag_adviser.evaluators.retrieval_modes import tokenize

TRANSCRIPT = """\
[09:00] Alice: Good morning everyone, let's start with the pump incident.
[09:01] Bob: The pump overheated after error code E4021 appeared on the panel.
[09:02] Alice: Did the cooling fan trip first or the pump?
[09:03] Bob: The fan. Maintenance replaced it and the pump ran fine afterwards.
[09:04] Carol: I'll file the report and attach the E4021 log excerpt.
"""

CSV = """\
id,part,status,note
1,pump,ok,replaced seal
2,fan,failed,bearing noise
3,valve,ok,none
4,sensor,failed,E4021 error
"""


class TestSpeakerSplit:
    def test_turns_are_detected_and_never_split(self) -> None:
        turns = _split_turns(TRANSCRIPT)
        assert len(turns) == 5
        assert turns[1].startswith("[09:01] Bob:")

        corpus = chunk_speaker_turns([("meeting.txt", TRANSCRIPT)], chunk_size=160)
        assert corpus.strategy_name == "speaker_split"
        assert all(c.metadata["speaker_turns"] for c in corpus.chunks)
        assert 2 <= corpus.total_chunks <= 4
        # Every chunk boundary is a turn boundary: chunks start with a timestamp/speaker.
        assert all(c.text.startswith("[09:") for c in corpus.chunks)
        assert all(len(c.text) <= 160 or c.text.count("\n") == 0 for c in corpus.chunks)

    def test_prose_falls_back_to_recursive(self) -> None:
        prose = "No speakers here. " * 40
        corpus = chunk_speaker_turns([("notes.txt", prose)], chunk_size=200)
        assert corpus.total_chunks >= 2
        assert all(c.metadata["speaker_turns"] is False for c in corpus.chunks)
        assert all(c.strategy == "speaker_split" for c in corpus.chunks)


class TestRowBased:
    def test_delimiter_detection(self) -> None:
        assert _detect_delimiter(CSV.strip().splitlines()) == ","
        assert _detect_delimiter(["a\tb\tc", "1\t2\t3", "4\t5\t6"]) == "\t"
        prose_lines = ["just prose here", "another line of prose", "and one more"]
        assert _detect_delimiter(prose_lines) is None

    def test_rows_grouped_with_header_repeated(self) -> None:
        corpus = chunk_rows([("parts.csv", CSV)], chunk_size=70)
        assert corpus.total_chunks >= 2
        for c in corpus.chunks:
            assert c.text.startswith("id,part,status,note\n")
            assert c.metadata["tabular"] is True and c.metadata["delimiter"] == ","
        # All data rows appear exactly once across chunks.
        rows = [ln for c in corpus.chunks for ln in c.text.splitlines()[1:]]
        assert sorted(rows) == sorted(CSV.strip().splitlines()[1:])

    def test_non_tabular_falls_back(self) -> None:
        corpus = chunk_rows([("notes.txt", "Plain text. " * 50)], chunk_size=150)
        assert corpus.total_chunks >= 2
        assert all(c.metadata["tabular"] is False for c in corpus.chunks)


class _FakeModel:
    VOCAB = ["pump", "overheated", "e4021", "fan", "seal", "valve", "sensor", "report"]

    def encode(self, texts, **_):
        import numpy as np

        out = []
        for t in texts:
            toks = tokenize(t)
            vec = np.array([float(toks.count(w)) for w in self.VOCAB]) + 1e-3
            out.append(vec / np.linalg.norm(vec))
        return np.array(out)


class TestPipelineIntegration:
    def test_new_strategies_run_end_to_end(self, tmp_path: Path) -> None:
        pytest.importorskip("numpy")
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "meeting.txt").write_text(TRANSCRIPT, "utf-8")
        (docs / "parts.csv").write_text(CSV, "utf-8")
        gt = tmp_path / "gt.jsonl"
        gt.write_text(
            json.dumps({"query": "which part had error E4021", "relevant_docs": ["parts.csv"]})
            + "\n"
        )
        config = EvalConfig(
            corpus_path=docs, ground_truth_path=gt,
            strategies=["speaker_split", "row_based"], top_k=2,
            chunk_size=120, vector_backend="memory",
        )

        def fake_load(self, model_id, trust_remote_code=False):
            self._model = _FakeModel()
            self._dimension = len(_FakeModel.VOCAB)

        with patch.object(EvalPipelineRunner, "_load_embedding_model", fake_load):
            report = EvalPipelineRunner().run(config)

        names = [m.strategy_name for m in report.strategy_results]
        assert names == ["speaker_split", "row_based"]
        assert all(m.num_chunks > 0 for m in report.strategy_results)
