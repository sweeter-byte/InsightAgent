"""Subprocess bridge to a pinned checkout of the official evaluator."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from datetime import datetime, timezone

from evals.deepresearch_bench.dataset import OFFICIAL_COMMIT, sha256_file


CommandRunner = Callable[..., Any]


class OfficialEvaluator:
    def __init__(
        self,
        official_root: str | Path,
        scores_root: str | Path,
        export_path: str | Path,
        *,
        command_runner: CommandRunner = subprocess.run,
        python: str = sys.executable,
    ) -> None:
        self.official_root = Path(official_root).resolve()
        self.scores_root = Path(scores_root).resolve()
        self.export_path = Path(export_path).resolve()
        self.command_runner = command_runner
        self.python = python

    def score(
        self,
        *,
        race: bool,
        fact: bool,
        confirm_paid: bool,
        environ: Mapping[str, str] | None = None,
        workers: int = 1,
        force: bool = False,
    ) -> dict[str, Any]:
        if not race and not fact:
            raise ValueError("enable at least one of RACE or FACT")
        if not confirm_paid:
            raise ValueError("paid evaluation requires --confirm-paid")
        if workers <= 0:
            raise ValueError("workers must be positive")
        values = dict(os.environ if environ is None else environ)
        backend = values.get("LLM_BACKEND", "openrouter").lower()
        key_name = {"openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY"}.get(backend)
        if key_name is None:
            raise RuntimeError("LLM_BACKEND must be 'openrouter' or 'openai'")
        if not values.get(key_name, "").strip():
            raise RuntimeError(f"missing required judge key: {key_name}")
        if fact and not values.get("JINA_API_KEY", "").strip():
            raise RuntimeError("missing required FACT scraping key: JINA_API_KEY")
        self._validate_inputs()
        self.scores_root.mkdir(parents=True, exist_ok=True)
        endpoint = _public_endpoint(
            values.get(
                "OPENROUTER_BASE_URL" if backend == "openrouter" else "OPENAI_BASE_URL",
                "https://openrouter.ai/api/v1"
                if backend == "openrouter"
                else "https://api.openai.com/v1",
            )
        )
        race_model = values.get(
            "RACE_MODEL", "openai/gpt-5.5" if backend == "openrouter" else "gpt-5.5"
        )
        clean_model = values.get(
            "CLEAN_MODEL",
            "openai/gpt-5.6-luna" if backend == "openrouter" else "gpt-5.6-luna",
        )
        fact_model = values.get(
            "FACT_MODEL",
            "openai/gpt-5.4-mini" if backend == "openrouter" else "gpt-5.4-mini",
        )
        invocation_protocol = (
            "official"
            if not any(name in values for name in ("RACE_MODEL", "CLEAN_MODEL", "FACT_MODEL"))
            else "custom_judge"
        )
        metadata_path = self.scores_root / "evaluation.json"
        previous: dict[str, Any] = {}
        if metadata_path.is_file():
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous = loaded
        orphaned_artifacts = not previous and any(
            path.is_file()
            for name in ("race", "fact")
            for path in (self.scores_root / name).rglob("*")
            if (self.scores_root / name).is_dir()
        )
        if orphaned_artifacts and not force:
            raise ValueError(
                "score artifacts have no provenance metadata; pass --force to rescore"
            )
        if orphaned_artifacts:
            self._clear_stage("race")
            self._clear_stage("fact")
        export_sha256 = sha256_file(self.export_path)
        race_provenance = _stage_provenance(
            "race",
            export_sha256=export_sha256,
            backend=backend,
            endpoint=endpoint,
            models={"race": race_model, "clean": clean_model},
            protocol=invocation_protocol,
        )
        fact_provenance = _stage_provenance(
            "fact",
            export_sha256=export_sha256,
            backend=backend,
            endpoint=endpoint,
            models={"fact": fact_model},
            protocol=invocation_protocol,
        )
        prior_sha256 = previous.get("export_sha256")
        export_changed = bool(previous) and prior_sha256 != export_sha256
        if export_changed and not force:
            raise ValueError(
                "benchmark export has changed since the saved scores; pass --force to rescore"
            )
        if export_changed:
            previous = {}
            self._clear_stage("race")
            self._clear_stage("fact")
        for requested, name, provenance in (
            (race, "race", race_provenance),
            (fact, "fact", fact_provenance),
        ):
            if not requested:
                continue
            prior = previous.get(f"{name}_provenance")
            prior_status = previous.get(f"{name}_status", "not_requested")
            changed = prior_status != "not_requested" and (
                not isinstance(prior, dict)
                or prior.get("fingerprint") != provenance["fingerprint"]
            )
            if changed and not force:
                raise ValueError(
                    f"{name.upper()} judge configuration has changed; pass --force to rescore"
                )
            if force:
                self._clear_stage(name)
        stage_protocols = [
            provenance.get("protocol")
            for provenance in (
                race_provenance if race else previous.get("race_provenance"),
                fact_provenance if fact else previous.get("fact_provenance"),
            )
            if isinstance(provenance, dict)
        ]
        metadata = {
            **previous,
            "backend": backend,
            "api_base_url": endpoint,
            "race_model": race_model,
            "clean_model": clean_model,
            "fact_model": fact_model,
            "race_requested": race,
            "fact_requested": fact,
            "protocol": "custom_judge" if "custom_judge" in stage_protocols else "official",
            "official_commit": OFFICIAL_COMMIT,
            "api_version": None,
            "token_usage": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "export_path": str(self.export_path),
            "export_sha256": export_sha256,
            "export_size_bytes": self.export_path.stat().st_size,
            "race_status": "pending" if race else previous.get("race_status", "not_requested"),
            "fact_status": "pending" if fact else previous.get("fact_status", "not_requested"),
        }
        if race:
            metadata.pop("race_error", None)
            metadata["race_provenance"] = race_provenance
        if fact:
            metadata.pop("fact_error", None)
            metadata["fact_provenance"] = fact_provenance
        self._write_metadata(metadata)
        safe_env = dict(values)
        query = self.official_root / "data" / "prompt_data" / "query.jsonl"
        first_error: BaseException | None = None
        if race:
            race_dir = self.scores_root / "race"
            command = [
                self.python,
                "-u",
                "deepresearch_bench_race.py",
                "insightagent",
                "--raw_data_dir",
                str(self.export_path.parent),
                "--cleaned_data_dir",
                str(race_dir / "cleaned"),
                "--max_workers",
                str(workers),
                "--query_file",
                str(query),
                "--output_dir",
                str(race_dir),
            ]
            if force:
                command.append("--force")
            try:
                self._run(command, safe_env)
                self._validate_race_outputs(race_dir)
            except BaseException as exc:
                metadata["race_status"] = "failed"
                metadata["race_error"] = f"{type(exc).__name__}: {exc}"
                self._write_metadata(metadata)
                if not isinstance(exc, Exception):
                    metadata["ended_at"] = datetime.now(timezone.utc).isoformat()
                    self._write_metadata(metadata)
                    raise
                first_error = exc
            else:
                metadata["race_status"] = "succeeded"
            self._write_metadata(metadata)
        if fact:
            fact_dir = self.scores_root / "fact"
            fact_dir.mkdir(parents=True, exist_ok=True)
            if force:
                for name in (
                    "extracted.jsonl",
                    "deduplicated.jsonl",
                    "scraped.jsonl",
                    "validated.jsonl",
                    "fact_result.txt",
                ):
                    (fact_dir / name).unlink(missing_ok=True)
            stages = [
                ("utils.extract", self.export_path, fact_dir / "extracted.jsonl", True),
                ("utils.deduplicate", fact_dir / "extracted.jsonl", fact_dir / "deduplicated.jsonl", True),
                ("utils.scrape", fact_dir / "deduplicated.jsonl", fact_dir / "scraped.jsonl", False),
                ("utils.validate", fact_dir / "scraped.jsonl", fact_dir / "validated.jsonl", True),
            ]
            try:
                for module, source, output, needs_query in stages:
                    command = [
                        self.python,
                        "-u",
                        "-m",
                        module,
                        "--raw_data_path",
                        str(source),
                        "--output_path",
                        str(output),
                        "--n_total_process",
                        str(workers),
                    ]
                    if needs_query:
                        command.extend(["--query_data_path", str(query)])
                    self._run(command, safe_env)
                self._run(
                    [
                        self.python,
                        "-u",
                        "-m",
                        "utils.stat",
                        "--input_path",
                        str(fact_dir / "validated.jsonl"),
                        "--output_path",
                        str(fact_dir / "fact_result.txt"),
                    ],
                    safe_env,
                )
                self._validate_fact_outputs(fact_dir)
            except BaseException as exc:
                metadata["fact_status"] = "failed"
                metadata["fact_error"] = f"{type(exc).__name__}: {exc}"
                self._write_metadata(metadata)
                if not isinstance(exc, Exception):
                    metadata["ended_at"] = datetime.now(timezone.utc).isoformat()
                    self._write_metadata(metadata)
                    raise
                if first_error is None:
                    first_error = exc
            else:
                metadata["fact_status"] = "succeeded"
        metadata["ended_at"] = datetime.now(timezone.utc).isoformat()
        self._write_metadata(metadata)
        if first_error is not None:
            raise first_error
        return metadata

    def _run(self, command: list[str], environ: Mapping[str, str]) -> None:
        log_path = self.scores_root / "evaluation.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"$ {' '.join(command)}\n")
            log.flush()
            completed = self.command_runner(
                command,
                cwd=self.official_root,
                env=dict(environ),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"official evaluator command failed with exit code {completed.returncode}: "
                + " ".join(command[:4])
            )

    def _write_metadata(self, metadata: Mapping[str, Any]) -> None:
        destination = self.scores_root / "evaluation.json"
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(dict(metadata), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    def _clear_stage(self, name: str) -> None:
        directory = self.scores_root / name
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_race_outputs(directory: Path) -> None:
        _require_jsonl(directory / "raw_results.jsonl", "RACE raw results")
        _require_metric_file(directory / "race_result.txt", "RACE summary")

    @staticmethod
    def _validate_fact_outputs(directory: Path) -> None:
        _require_jsonl(directory / "validated.jsonl", "FACT validated results")
        _require_metric_file(directory / "fact_result.txt", "FACT summary")

    def _validate_inputs(self) -> None:
        if not self.export_path.is_file():
            raise FileNotFoundError(f"benchmark export not found: {self.export_path}")
        if self.export_path.name != "insightagent.jsonl":
            raise ValueError(
                "official RACE target name requires export file insightagent.jsonl"
            )
        _require_jsonl(self.export_path, "benchmark export")
        required = [
            "deepresearch_bench_race.py",
            "data/prompt_data/query.jsonl",
            "data/criteria_data/criteria.jsonl",
            "data/test_data/cleaned_data/reference.jsonl",
            "utils/extract.py",
            "utils/deduplicate.py",
            "utils/scrape.py",
            "utils/validate.py",
            "utils/stat.py",
        ]
        missing = [name for name in required if not (self.official_root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"official checkout is incomplete: {missing}")


def _public_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def _stage_provenance(
    stage: str,
    *,
    export_sha256: str,
    backend: str,
    endpoint: str,
    models: Mapping[str, str],
    protocol: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "stage": stage,
        "export_sha256": export_sha256,
        "official_commit": OFFICIAL_COMMIT,
        "backend": backend,
        "api_base_url": endpoint,
        "models": dict(models),
        "protocol": protocol,
    }
    value["fingerprint"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return value


def _require_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"{label} is missing or empty: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{label} contains invalid JSON at line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"{label} line {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise RuntimeError(f"{label} contains no records")
    return rows


def _require_metric_file(path: Path, label: str) -> None:
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"{label} is missing or empty: {path}")
    parsed = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        _, raw_value = line.split(":", 1)
        try:
            float(raw_value.strip())
        except ValueError:
            continue
        parsed = True
        break
    if not parsed:
        raise RuntimeError(f"{label} contains no numeric metrics: {path}")
