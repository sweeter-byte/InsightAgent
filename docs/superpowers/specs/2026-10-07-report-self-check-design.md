# Report Self Check Design

## Goal

Add one bounded quality gate between structured report generation and Markdown
delivery. The gate checks whether the generated report is faithful to the
existing research plan, Evidence, and latest Evidence assessments. It does not
retrieve information, change the research plan, or treat model knowledge as
Evidence.

## Existing Architecture

The current terminal workflow is:

```text
all research tasks complete
→ generate_report
  → generate and validate Claims
  → assemble StructuredReport and Citations
  → render Markdown
  → set report and final_output
→ END
```

The project uses dataclasses and enums for domain models. LLM-backed components
call the small `LLMClient.chat(messages, tools=None)` interface and apply strict
JSON parsing plus deterministic runtime validation. Dependencies are passed to
the workflow constructor and composed in `insight_agent.__main__`.

Chapter 13 already owns the structural invariants for report candidates,
Claims, Evidence bindings, task ownership, citation provenance, and Markdown
rendering. Chapter 14 reuses those validators instead of duplicating them.

## Target Workflow

```text
generate_report
→ self_check_report
  ├─ PASS
  │  → finalize_report
  │  → Markdown render
  │  → END
  └─ REVISE
     ├─ repair budget remains
     │  → repair_report
     │  → deterministic report validation
     │  → increment self_check_rounds
     │  → self_check_report
     └─ budget exhausted
        → failed
```

`max_repair_rounds` defaults to and remains bounded at one for this chapter.
`self_check_rounds` records completed Repair operations, not Checker calls.

## Package and Components

Create an independent `insight_agent.self_check` package:

- `models.py` defines the finite result vocabulary and the self-check domain
  error.
- `prompts.py` contains the checker and controlled-repair system prompts.
- `checker.py` constructs claim-centered context, invokes the LLM, parses its
  response, and validates the result against runtime identities.
- `repair.py` asks the LLM for a bounded report correction, rebuilds runtime
  identities and citations, and reuses Chapter 13 validators.
- `__init__.py` exposes the package's public API.

The checker and repairer depend only on `LLMClient` and existing domain models.
They have no Retriever, SearchProvider, Tool Registry, or Retrieval Router
dependency.

## Data Model

Use the repository's existing dataclass and enum style:

```text
SelfCheckStatus
  PASS = "pass"
  REVISE = "revise"

SelfCheckIssueCode
  UNSUPPORTED_CLAIM = "unsupported_claim"
  CITATION_MISMATCH = "citation_mismatch"
  MISSING_GAP_DISCLOSURE = "missing_gap_disclosure"
  CONSTRAINT_VIOLATION = "constraint_violation"
  INCONSISTENT_CLAIMS = "inconsistent_claims"

SelfCheckIssue
  code
  reason
  task_id | None
  claim_id | None

SelfCheckResult
  status
  issues
  summary
```

`ResearchState` gains only `self_check_result` and `self_check_rounds`.

`SelfCheckError` is the explicit domain failure used for malformed checker or
repair output and for a report that remains `REVISE` after the single Repair
budget is exhausted. A failed report never receives `final_output`.

## ReportSelfChecker

The checker accepts a plan, structured report, task-keyed Evidence pool, and
assessment histories. It selects the latest assessment for every plan task and
builds context around each Claim:

```text
Claim
→ bound Evidence only
→ owning Task
→ latest Assessment
```

It also supplies the plan objective and constraints plus report-section and
task-assessment summaries so that missing gaps and cross-section conflicts can
be checked without dumping unrelated Evidence into the prompt.

The LLM judges only these five issue codes:

1. unsupported Claim;
2. semantically mismatched citation/Evidence binding;
3. omitted gap from a latest insufficient assessment;
4. objective or constraint violation;
5. obvious cross-section Claim inconsistency.

Runtime validation enforces:

- exact status and issue-code vocabularies;
- `PASS` has no issues;
- `REVISE` has at least one issue;
- every supplied task ID exists in the plan;
- every supplied Claim ID exists in the report;
- a Claim ID belongs to the supplied task ID when both are present;
- reasons and summary are non-empty strings;
- the report and input research structures are valid domain objects.

The checker does not modify the report.

## Controlled Repair

The repairer accepts the current report, the validated `REVISE` result, the
plan, Evidence pool, and latest assessments. The prompt permits only narrowing
or deleting unsupported Claims, removing irrelevant Evidence bindings,
restoring recorded gaps, and resolving obvious wording conflicts.

The raw model response contains section Claim text and Evidence bindings, but
does not own provenance or control data. Runtime reconstructs the returned
`StructuredReport` as follows:

- objective, task order, task IDs, and section titles come from the plan;
- `sufficient` and `missing_information` come from each latest assessment;
- Claim IDs are assigned deterministically in section order;
- allowed Evidence is recalculated with `select_report_candidates`;
- Claims are checked with `validate_claims`;
- Citations are rebuilt from trusted Evidence with `assemble_report`.

Consequently, Repair cannot create Evidence, cross task boundaries, invent a
URL or Citation, change an assessment, or raise sufficiency. It returns a full,
runtime-validated `StructuredReport` to the workflow.

## Workflow Integration

`ResearchRoutingWorkflow` receives injected self-checker and repairer
collaborators and a positive `max_repair_rounds` value whose default is one.
The canonical and LangGraph state schemas carry the two new fields.

`generate_report` stops rendering and returns only `report`.
`self_check_report` returns only `self_check_result`.
The conditional edge after self-check is decided by runtime status and repair
budget. `repair_report` replaces only the report and increments the repair
counter. `finalize_report` requires `PASS` and is the only place that invokes
`MarkdownReportRenderer`. The failed node raises `SelfCheckError` and therefore
cannot return a known-bad report as a successful result.

The production composition creates `ReportSelfChecker` and `ReportRepairer`
from the existing shared LLM client. Existing retrieval, grading, and report
generation paths are otherwise unchanged.

## Error Handling

Malformed LLM JSON, unexpected fields, invalid enums, invalid identities, and
invalid repaired Claim bindings fail closed with `SelfCheckError` or the reused
`ReportGenerationError`. This chapter does not introduce general retries or a
second repair attempt. Exhaustion after one Repair is an explicit workflow
failure, never an implicit pass.

## Test Strategy

Use scripted fake LLMs and injected checker/repairer fakes; tests remain offline.
Follow test-first red/green cycles for:

- valid result parsing and claim-centered context;
- each of the five semantic issue codes;
- unknown Claim and Task IDs;
- `PASS` with issues and `REVISE` without issues;
- repair reconstruction and Chapter 13 validation reuse;
- direct PASS without Repair followed by Markdown finalization;
- REVISE routing to one Repair and a second check;
- Repair followed by PASS with `self_check_rounds == 1`;
- Repair followed by REVISE, no second Repair, and explicit failure;
- state defaults and application dependency composition.

Verification runs the new Chapter 14 tests, the reporting/evidence/research
workflow regression set, and then the complete test suite in the repository's
documented `insight-agent` Conda environment. The repository has no configured
formatter, linter, or static type checker beyond pytest.

## Explicit Non-Goals

No new retrieval, search, Evidence, assessment changes, replanning, external
fact checking, judge model, multi-agent review, persistence, checkpointing,
observability, retry framework, or unbounded self-refinement is introduced.
