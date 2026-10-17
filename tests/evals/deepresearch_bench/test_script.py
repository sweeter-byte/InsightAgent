from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[3] / "scripts" / "benchmarks" / "deepresearch.sh"


def test_script_uses_repo_root_active_python_and_forwards_arguments(tmp_path: Path) -> None:
    output = tmp_path / "invocation.txt"
    executable = tmp_path / "python"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$PWD" "$@" > "$DEEPRESEARCH_TEST_OUTPUT"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    env = {
        **os.environ,
        "DEEPRESEARCH_PYTHON": str(executable),
        "DEEPRESEARCH_TEST_OUTPUT": str(output),
    }

    completed = subprocess.run(
        ["bash", str(SCRIPT), "smoke", "--run-id", "sample"],
        cwd=tmp_path,
        env=env,
        check=False,
    )

    assert completed.returncode == 0
    invocation = output.read_text(encoding="utf-8").splitlines()
    assert invocation[0] == str(SCRIPT.parents[2])
    assert invocation[1:] == [
        "-m",
        "evals.deepresearch_bench",
        "smoke",
        "--run-id",
        "sample",
    ]


def test_script_is_strict_and_does_not_auto_run_paid_commands() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert 'COMMAND="${1:-}"' in text
    assert "score" not in text.split('COMMAND="${1:-}"', 1)[0]


def test_script_propagates_python_failure(tmp_path: Path) -> None:
    executable = tmp_path / "python"
    executable.write_text("#!/usr/bin/env bash\nexit 17\n", encoding="utf-8")
    executable.chmod(0o755)
    completed = subprocess.run(
        ["bash", str(SCRIPT), "doctor"],
        cwd=tmp_path,
        env={**os.environ, "DEEPRESEARCH_PYTHON": str(executable)},
        check=False,
    )
    assert completed.returncode == 17
