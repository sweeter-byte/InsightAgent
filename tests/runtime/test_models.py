from __future__ import annotations

import pytest
from pydantic import ValidationError

from insight_agent.runtime.models import CreateResearchRunRequest, RunStatus
from insight_agent.runtime.policies import RuntimePolicy


def test_run_status_contains_only_executable_states() -> None:
    assert {status.value for status in RunStatus} == {
        "queued",
        "running",
        "completed",
        "failed",
        "timed_out",
        "interrupted",
    }


def test_create_request_strips_query_and_rejects_blank_input() -> None:
    assert CreateResearchRunRequest(query="  useful query  ").query == "useful query"
    with pytest.raises(ValidationError):
        CreateResearchRunRequest(query="   ")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_concurrency": 0}, "max_concurrency"),
        ({"run_timeout_seconds": 0}, "run_timeout_seconds"),
        ({"infra_retry_attempts": 0}, "infra_retry_attempts"),
        ({"infra_retry_backoff_seconds": -1}, "infra_retry_backoff_seconds"),
        ({"event_ttl_seconds": 0}, "event_ttl_seconds"),
    ],
)
def test_runtime_policy_rejects_invalid_values(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RuntimePolicy(**kwargs)
