# Report Self Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-repair, evidence-grounded self-check gate before a research report can be rendered and returned.

**Architecture:** Add an independent `insight_agent.self_check` package with dataclass models, a semantic LLM checker, and a controlled LLM repairer. Runtime code retains identity, provenance, repair-budget, and routing authority by reusing Chapter 13 validators and rendering Markdown only after a validated `PASS`.

**Tech Stack:** Python 3.11+, dataclasses, Enum, the existing OpenAI-compatible `LLMClient`, LangGraph, pytest, and the `insight-agent` Conda environment.

---

## File map

- Create `insight_agent/self_check/models.py` for finite result types and errors.
- Create `insight_agent/self_check/prompts.py` for the two system prompts.
- Create `insight_agent/self_check/checker.py` for context construction and result validation.
- Create `insight_agent/self_check/repair.py` for controlled report repair.
- Create `insight_agent/self_check/__init__.py` for public exports.
- Modify `insight_agent/planning/models.py` for the two state fields.
- Modify `insight_agent/research/workflow.py` for check/repair/finalize/fail nodes.
- Modify `insight_agent/__main__.py` for production dependency composition.
- Create `tests/test_self_check.py` for model/checker/repairer unit tests.
- Create `tests/test_self_check_workflow.py` for terminal workflow integration.
- Modify existing report, research-workflow, and composition tests for the new gate.

### Task 1: Add finite self-check models and state

**Files:**
- Create: `insight_agent/self_check/models.py`
- Create: `insight_agent/self_check/__init__.py`
- Modify: `insight_agent/planning/models.py`
- Test: `tests/test_self_check.py`

- [ ] **Step 1: Write failing model tests**

```python
def test_self_check_models_use_finite_vocabulary() -> None:
    issue = SelfCheckIssue(
        SelfCheckIssueCode.UNSUPPORTED_CLAIM,
        "Claim is broader than Evidence.",
        "T1",
        "T1-C1",
    )
    result = SelfCheckResult(SelfCheckStatus.REVISE, [issue], "Repair required.")
    assert {item.value for item in SelfCheckStatus} == {"pass", "revise"}
    assert {item.value for item in SelfCheckIssueCode} == {
        "unsupported_claim",
        "citation_mismatch",
        "missing_gap_disclosure",
        "constraint_violation",
        "inconsistent_claims",
    }
    assert result.issues == [issue]


def test_research_state_defaults_self_check_fields() -> None:
    state = ResearchState(query="query")
    assert state.self_check_result is None
    assert state.self_check_rounds == 0
```

- [ ] **Step 2: Verify RED**

Run `conda run --no-capture-output -n insight-agent python -m pytest tests/test_self_check.py -q`.
Expected: import failure because the package does not exist.

- [ ] **Step 3: Implement the model**

```python
class SelfCheckStatus(str, Enum):
    PASS = "pass"
    REVISE = "revise"


class SelfCheckIssueCode(str, Enum):
    UNSUPPORTED_CLAIM = "unsupported_claim"
    CITATION_MISMATCH = "citation_mismatch"
    MISSING_GAP_DISCLOSURE = "missing_gap_disclosure"
    CONSTRAINT_VIOLATION = "constraint_violation"
    INCONSISTENT_CLAIMS = "inconsistent_claims"


class SelfCheckError(RuntimeError):
    pass


@dataclass(slots=True)
class SelfCheckIssue:
    code: SelfCheckIssueCode
    reason: str
    task_id: str | None = None
    claim_id: str | None = None


@dataclass(slots=True)
class SelfCheckResult:
    status: SelfCheckStatus
    issues: list[SelfCheckIssue]
    summary: str
```

Add `self_check_result: SelfCheckResult | None = None` and
`self_check_rounds: int = 0` to `ResearchState`, using a `TYPE_CHECKING` import.

- [ ] **Step 4: Verify GREEN and commit**

```bash
conda run --no-capture-output -n insight-agent python -m pytest tests/test_self_check.py -q
git add insight_agent/self_check insight_agent/planning/models.py tests/test_self_check.py
git commit -m "feat: add report self-check models"
```

### Task 2: Build and validate semantic self-check results

**Files:**
- Create: `insight_agent/self_check/prompts.py`
- Create: `insight_agent/self_check/checker.py`
- Modify: `insight_agent/self_check/__init__.py`
- Test: `tests/test_self_check.py`

- [ ] **Step 1: Write a failing PASS/context test**

Use a scripted fake LLM returning this exact JSON object:

```python
{
    "status": "pass",
    "issues": [],
    "summary": "Claims are supported and recorded gaps are disclosed.",
}
```

Assert the checker returns the equivalent dataclass, passes `tools=None`, and
the serialized `claim_contexts[0]` contains the Claim, only its bound Evidence,
the owning task, and latest assessment.

- [ ] **Step 2: Verify RED**

Run the focused checker test. Expected: `ReportSelfChecker` is missing.

- [ ] **Step 3: Implement the checker interface**

```python
def check(
    self,
    *,
    plan: ResearchPlan,
    report: StructuredReport,
    evidence_pool: dict[str, list[Evidence]],
    assessments: dict[str, list[EvidenceAssessment]],
) -> SelfCheckResult:
```

The system prompt must constrain the model to the five issue types and supplied
state, ban outside knowledge/retrieval/research/language-polish suggestions, and
require one strict JSON object. Validate the existing report by rebuilding it
with `assemble_report` and comparing citations. Require plan-ordered sections
and one latest assessment per task. Serialize using `ensure_ascii=False` and
`allow_nan=False`.

- [ ] **Step 4: Implement strict output parsing**

Require exactly `status`, `issues`, and `summary`. Require every issue to contain
exactly `code`, `reason`, `task_id`, and `claim_id`. Convert only through the
finite enums. Reject empty reasons and summaries.

- [ ] **Step 5: Add failing issue-vocabulary and runtime-invariant tests**

First parameterize all five valid issue codes and assert each parses into its
matching enum. Then parameterize unknown task, unknown Claim, Claim/task
mismatch, PASS with issues, REVISE without issues, invalid status/code, empty
reason, and empty summary. Every invalid case must raise `SelfCheckError`.

- [ ] **Step 6: Implement identity and terminal-status validation**

```python
if status is SelfCheckStatus.PASS and issues:
    raise SelfCheckError("pass self-check result must not contain issues")
if status is SelfCheckStatus.REVISE and not issues:
    raise SelfCheckError("revise self-check result must contain at least one issue")
```

Validate every non-null task/Claim ID against runtime objects and validate their
ownership when both are present.

- [ ] **Step 7: Verify and commit**

```bash
conda run --no-capture-output -n insight-agent python -m pytest tests/test_self_check.py -k checker -q
git add insight_agent/self_check tests/test_self_check.py
git commit -m "feat: add evidence-grounded report checker"
```

### Task 3: Add controlled repair with Chapter 13 validation

**Files:**
- Create: `insight_agent/self_check/repair.py`
- Modify: `insight_agent/self_check/prompts.py`
- Modify: `insight_agent/self_check/__init__.py`
- Test: `tests/test_self_check.py`

- [ ] **Step 1: Write a failing repair test**

Start from an overgeneralized Claim and script a complete report response with a
narrowed Claim and the existing Evidence ID. Assert deterministic `T1-C1`,
runtime-built Citations, and assessment-derived sufficiency/gaps.

- [ ] **Step 2: Verify RED**

Run the focused repair test. Expected: `ReportRepairer` is missing.

- [ ] **Step 3: Implement the repair interface and prompt**

```python
def repair(
    self,
    *,
    plan: ResearchPlan,
    report: StructuredReport,
    self_check_result: SelfCheckResult,
    evidence_pool: dict[str, list[Evidence]],
    assessments: dict[str, list[EvidenceAssessment]],
) -> StructuredReport:
```

Require a REVISE result. Supply current report, issues, latest assessments, and
assessment-allowed candidate Evidence. Permit only narrowing/deleting Claims,
removing bindings, restoring recorded gaps, and resolving explicit conflicts.
Ban new facts, Evidence, sources, assessment edits, sufficiency changes,
retrieval, and replanning.

- [ ] **Step 4: Rebuild runtime-owned fields**

Require exact top-level `objective` and `sections`. Require exact section fields
`task_id`, `title`, `claims`, `sufficient`, and `missing_information`. Objective,
task order, title, sufficiency, and gaps must equal runtime values. Assign Claim
IDs as `f"{task.id}-C{index}"`, then call `select_report_candidates`,
`validate_claims`, and `assemble_report`.

- [ ] **Step 5: Add fail-closed tests**

Reject unknown/cross-task Evidence, duplicate bindings, model Claim IDs,
model Citation/source/URL fields, changed objective/task order/title,
changed sufficiency/gaps, and invocation with PASS.

- [ ] **Step 6: Verify and commit**

```bash
conda run --no-capture-output -n insight-agent python -m pytest tests/test_self_check.py -q
git add insight_agent/self_check tests/test_self_check.py
git commit -m "feat: add controlled report repair"
```

### Task 4: Defer Markdown rendering and carry state through LangGraph

**Files:**
- Modify: `insight_agent/research/workflow.py`
- Modify: `tests/test_report_workflow.py`
- Modify: `tests/test_research_workflow.py`

- [ ] **Step 1: Change the generation test first**

Replace direct Markdown assertions with `assert "final_output" not in update`.
Add graph-state round-trip assertions for `self_check_result` and
`self_check_rounds`.

- [ ] **Step 2: Verify RED**

Run the focused `test_generate_report_skips_llm_for_no_allowed_evidence_and_keeps_gap`.
Expected: it fails because generation still renders Markdown.

- [ ] **Step 3: Split generation from finalization**

Change the terminal return to:

```python
return {"report": assemble_report(plan.objective, sections, state.evidence_pool)}
```

Add both new fields to `_GraphState`, `_to_graph_state`, and
`_from_graph_state` without mutating input objects.

- [ ] **Step 4: Verify focused tests and commit**

```bash
conda run --no-capture-output -n insight-agent python -m pytest tests/test_report_workflow.py::test_generate_report_skips_llm_for_no_allowed_evidence_and_keeps_gap tests/test_research_workflow.py::test_graph_state_round_trip_preserves_self_check_fields -q
git add insight_agent/research/workflow.py tests/test_report_workflow.py tests/test_research_workflow.py
git commit -m "refactor: defer rendering until report self-check"
```

### Task 5: Gate finalization on PASS

**Files:**
- Modify: `insight_agent/research/workflow.py`
- Create: `tests/test_self_check_workflow.py`
- Modify: `tests/test_research_workflow.py`

- [ ] **Step 1: Write a failing direct-PASS test**

Inject an always-PASS checker and a repairer that fails if called. Include an
insufficient task whose recorded `missing_information` remains in its report
section. Assert one check, zero repairs, zero repair rounds, stored PASS, the
gap in rendered Markdown, and a rendered final output.

- [ ] **Step 2: Verify RED**

Run `tests/test_self_check_workflow.py::test_pass_finalizes_without_repair`.
Expected: workflow constructor or graph failure because nodes are absent.

- [ ] **Step 3: Add injected protocols and methods**

Require `self_checker` and `report_repairer` keyword dependencies. Add fixed
`MAX_REPAIR_ROUNDS = 1`. Implement `self_check_report`,
`choose_after_self_check`, and `finalize_report`. Finalization must require
PASS, require a report, reject an existing output, and be the sole renderer.

- [ ] **Step 4: Wire the PASS graph**

```text
generate_report → self_check_report
self_check_report → conditional(finalize_report | repair_report | failed)
finalize_report → END
```

Add adapters matching existing node/edge methods.

- [ ] **Step 5: Adapt offline workflow fakes**

The test adapter in `tests/test_research_workflow.py` injects an always-PASS
checker and never-called repairer. Ordered-event expectations append
`self_check_report` and `finalize_report` after `generate_report`.

- [ ] **Step 6: Verify and commit**

```bash
conda run --no-capture-output -n insight-agent python -m pytest tests/test_report_workflow.py tests/test_research_workflow.py tests/test_self_check_workflow.py -q
git add insight_agent/research/workflow.py tests/test_report_workflow.py tests/test_research_workflow.py tests/test_self_check_workflow.py
git commit -m "feat: gate report finalization on self-check"
```

### Task 6: Add one Repair and safe exhaustion

**Files:**
- Modify: `insight_agent/research/workflow.py`
- Modify: `tests/test_self_check_workflow.py`

- [ ] **Step 1: Write failing REVISE→Repair→PASS test**

Script REVISE then PASS and one repaired report. Assert two checks, one Repair,
`self_check_rounds == 1`, repaired state, and Markdown from the repaired Claim.

- [ ] **Step 2: Verify RED and implement Repair**

Run the focused test, then implement `repair_report` to require REVISE and
remaining budget, call only the injected repairer, store its report, increment
the counter, and route back to self-check. Never route Repair to finalization.

- [ ] **Step 3: Write failing exhaustion test**

Script REVISE twice. Assert `SelfCheckError`, two checks, one Repair, no second
Repair, and no finalization.

- [ ] **Step 4: Implement explicit failure**

The failed node requires REVISE and `self_check_rounds == MAX_REPAIR_ROUNDS`,
then raises `SelfCheckError` containing the result summary without rendering.

- [ ] **Step 5: Verify and commit**

```bash
conda run --no-capture-output -n insight-agent python -m pytest tests/test_self_check_workflow.py -q
git add insight_agent/research/workflow.py tests/test_self_check_workflow.py
git commit -m "feat: bound report repair to one round"
```

### Task 7: Compose production dependencies

**Files:**
- Modify: `insight_agent/__main__.py`
- Modify: `tests/test_main.py`

- [ ] **Step 1: Add a failing composition assertion**

Patch both new component constructors and the workflow. Assert both receive the
shared LLM and the workflow receives those exact instances. Add a CLI test whose
fake app raises `SelfCheckError` and assert exit code 1 plus the dedicated error
prefix.

- [ ] **Step 2: Verify RED**

Run `conda run --no-capture-output -n insight-agent python -m pytest tests/test_main.py::test_build_app_shares_single_llm_client -q`.

- [ ] **Step 3: Compose with no retrieval authority**

```python
self_checker = ReportSelfChecker(llm=llm)
report_repairer = ReportRepairer(llm=llm)
```

Inject them into `ResearchRoutingWorkflow`; pass no retriever, router, or tool
registry to either component. Add `SelfCheckError` to the CLI's explicit domain
error handling and print it as `[self-check error] ...` with exit code 1.

- [ ] **Step 4: Verify and commit**

```bash
conda run --no-capture-output -n insight-agent python -m pytest tests/test_main.py tests/test_app.py tests/test_coordinator.py -q
git add insight_agent/__main__.py tests/test_main.py
git commit -m "feat: compose report self-check services"
```

### Task 8: Final verification and scope audit

**Files:**
- Modify only files needed to correct failures caused by this feature.

- [ ] **Step 1: Run Chapter 14 tests**

```bash
conda run --no-capture-output -n insight-agent python -m pytest tests/test_self_check.py tests/test_self_check_workflow.py -q
```

- [ ] **Step 2: Run related regressions**

```bash
conda run --no-capture-output -n insight-agent python -m pytest tests/test_reporting.py tests/test_report_workflow.py tests/test_evidence_grader.py tests/test_evidence_collector.py tests/test_research_workflow.py -q
```

- [ ] **Step 3: Run the complete suite**

```bash
conda run --no-capture-output -n insight-agent python -m pytest -q
```

The repository defines no formatter, linter, or type-check command, so do not
invent one.

- [ ] **Step 4: Audit the final diff**

Run `git diff --check` and `git status --short`. Confirm no retrieval calls or
Evidence construction exist under `insight_agent/self_check`, no assessment is
mutated, only one Repair is possible, and rendering occurs only after PASS.

- [ ] **Step 5: Commit a scoped verification correction only if needed**

Use `git commit -m "test: verify report self-check workflow"` only when Step 1–4
required an actual file correction; do not create an empty commit.
