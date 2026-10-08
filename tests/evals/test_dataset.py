from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import EvalCase, EvalDatasetError, load_eval_cases


def _write_lines(path: Path, *rows: object) -> None:
    lines = [row if isinstance(row, str) else json.dumps(row) for row in rows]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_load_eval_cases_reads_valid_rows_and_skips_blank_lines(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n"
        '{"case_id":"c1","query":"问题","tags":["local"],'
        '"gold_retrieval_ids":["chunk-1"],'
        '"required_points":["边界上下文"],'
        '"constraints":["简洁"],"fixture_set":"fixed-v1",'
        '"metadata":{"owner":"human"}}\n'
        "  \n",
        encoding="utf-8",
    )

    assert load_eval_cases(path) == [
        EvalCase(
            case_id="c1",
            query="问题",
            tags=("local",),
            gold_retrieval_ids=("chunk-1",),
            required_points=("边界上下文",),
            constraints=("简洁",),
            fixture_set="fixed-v1",
            metadata={"owner": "human"},
        )
    ]


def test_load_eval_cases_reports_json_path_line_and_column(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('\n{"case_id": "c1", bad}\n', encoding="utf-8")

    with pytest.raises(EvalDatasetError) as exc_info:
        load_eval_cases(path)

    message = str(exc_info.value)
    assert f"{path}:2:" in message
    assert "invalid JSON" in message


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ([], "row must be a JSON object"),
        ({"query": "Question"}, "case_id"),
        ({"case_id": "  ", "query": "Question"}, "case_id"),
        ({"case_id": "c1", "query": "  "}, "query"),
        (
            {"case_id": "c1", "query": "Question", "tags": "local"},
            "tags must be an array",
        ),
        (
            {"case_id": "c1", "query": "Question", "tags": ["local", " "]},
            "tags items must be non-empty strings",
        ),
        (
            {
                "case_id": "c1",
                "query": "Question",
                "tags": ["local", "local"],
            },
            "tags contains duplicate value",
        ),
        (
            {"case_id": "c1", "query": "Question", "fixture_set": " "},
            "fixture_set",
        ),
        (
            {"case_id": "c1", "query": "Question", "metadata": []},
            "metadata must be a JSON object",
        ),
        (
            {"case_id": "c1", "query": "Question", "gold_ids": []},
            "unknown field",
        ),
    ],
)
def test_load_eval_cases_rejects_invalid_rows_with_source_line(
    tmp_path: Path,
    row: object,
    message: str,
) -> None:
    path = tmp_path / "invalid.jsonl"
    _write_lines(path, "", row)

    with pytest.raises(EvalDatasetError, match=message) as exc_info:
        load_eval_cases(path)

    assert f"{path}:2:" in str(exc_info.value)


def test_load_eval_cases_reports_duplicate_id_and_original_line(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicates.jsonl"
    _write_lines(
        path,
        {"case_id": "same", "query": "First"},
        "",
        {"case_id": "same", "query": "Second"},
    )

    with pytest.raises(EvalDatasetError) as exc_info:
        load_eval_cases(path)

    message = str(exc_info.value)
    assert f"{path}:3:" in message
    assert "duplicate case_id 'same'" in message
    assert "first defined at line 1" in message


def test_load_eval_cases_returns_empty_list_for_blank_dataset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n  \n", encoding="utf-8")

    assert load_eval_cases(path) == []
