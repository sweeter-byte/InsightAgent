from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[3] / "scripts" / "benchmarks" / "scifact.sh"


def _fake_python(tmp_path: Path, exit_code: int = 0) -> tuple[Path, Path]:
    output = tmp_path / "invocation.txt"
    executable = tmp_path / "python"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$PWD" "$@" > "$SCIFACT_TEST_OUTPUT"\n'
        f"exit {exit_code}\n"
    )
    executable.chmod(0o755)
    return executable, output


def test_script_uses_repo_root_active_python_and_forwards_arguments(
    tmp_path: Path,
) -> None:
    executable, output = _fake_python(tmp_path)
    environment = {
        **os.environ,
        "SCIFACT_PYTHON": str(executable),
        "SCIFACT_TEST_OUTPUT": str(output),
    }

    completed = subprocess.run(
        ["bash", str(SCRIPT), "smoke", "--query-limit", "2"],
        cwd=tmp_path,
        env=environment,
        check=False,
    )

    invocation = output.read_text().splitlines()
    assert completed.returncode == 0
    assert invocation[0] == str(SCRIPT.parents[2])
    assert invocation[1:] == [
        "-m",
        "evals.beir_scifact",
        "smoke",
        "--query-limit",
        "2",
    ]


def test_script_propagates_nonzero_python_exit(tmp_path: Path) -> None:
    executable, output = _fake_python(tmp_path, exit_code=17)
    environment = {
        **os.environ,
        "SCIFACT_PYTHON": str(executable),
        "SCIFACT_TEST_OUTPUT": str(output),
    }

    completed = subprocess.run(
        ["bash", str(SCRIPT), "index"],
        cwd=tmp_path,
        env=environment,
        check=False,
    )

    assert completed.returncode == 17


def test_script_has_strict_bash_mode_and_does_not_auto_run_full() -> None:
    text = SCRIPT.read_text()

    assert "set -euo pipefail" in text
    assert 'COMMAND="${1:-}"' in text
    assert 'exec "$SCIFACT_PYTHON" -m evals.beir_scifact "$COMMAND" "$@"' in text
