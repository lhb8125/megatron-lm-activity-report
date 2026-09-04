# Megatron-LM Activity Report

This repository generates bilingual, theme-oriented monthly activity reports for
[`NVIDIA/Megatron-LM`](https://github.com/NVIDIA/Megatron-LM). The English report
is published to one upstream Issue per month. The same Issue is updated weekly;
the Chinese report has a stable path in this repository and keeps its weekly
history through Git.

## Update model

- Monday at approximately 09:00 Asia/Shanghai: regenerate the current
  month-to-date report through the preceding Sunday.
- The first day of each month at approximately 09:00 Asia/Shanghai: finalize the
  preceding month through its last calendar day.
- A Monday/month-start collision is serialized and deduplicated by the committed
  cutoff state.
- Issue bodies and report files are replaced with the complete month-to-date
  report. Weekly PR-by-PR logs are never appended to the narrative.

The schedule is defined in [`.github/workflows/report.yml`](.github/workflows/report.yml).
Repository names, report paths, filters, timezone, and model settings are defined
in [`configs/megatron-lm.yaml`](configs/megatron-lm.yaml).

## Report contract

The collector records PR snapshots, changed files, commits, and state transitions
in a rebuildable DuckDB ledger. A PR is active in a window when it was opened,
received a commit, merged, closed, or reopened in that window. Comment-only
updates do not create report activity.

Pure unit-test, CI, test/CI, and formatting-only changes are excluded from the
report and its substantive statistics. A PR that changes implementation code
alongside tests or CI remains eligible. Closed-unmerged PRs remain in the ledger
and aggregate count but are not narrated.

Every PR citation shows its actual target branch, such as `[dev #6022]` or
`[main #6870]`. Related `dev`/`main` counterparts still share one subproject
description, with both branch-labelled links placed directly after it.

The model first extracts bounded English facts for each high-confidence change
group, then selects concrete Delivered and In Progress themes. Chinese is a
translation of that validated structure. Validation requires identical theme
order, theme IDs, highlight counts, and PR citations in both languages. Each
subproject bullet carries only its directly supporting PR links; there is no
theme-level citation dump.

Chinese keeps established English model names, acronyms, APIs, kernel names, and
technical identifiers. The configurable glossary is defined under
`translation.preserve_terms` in [`configs/megatron-lm.yaml`](configs/megatron-lm.yaml),
and code-like terms are also detected from each English report automatically.

Only activity inside the current month-to-date window is eligible. An open PR is
not carried forward merely because it appeared in the previous report or remains
open. Fact extraction receives only commit subjects from the current window, so
old implementation history is not retold as new work.

Reports are stored at stable paths:

```text
reports/en-US/YYYY/MM.md
reports/zh-CN/YYYY/MM.md
```

## Local usage

Create a virtual environment and install the package:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Generate a preview without changing GitHub:

```bash
export MEGATRON_GH_TOKEN=...
export OPENAI_API_KEY=...
.venv/bin/megatron-activity-report run --cutoff 2026-08-23 --dry-run
```

Publish or backfill:

```bash
.venv/bin/megatron-activity-report run --cutoff 2026-08-23 --publish
.venv/bin/megatron-activity-report backfill --month 2026-07 --publish
```

`--publish` performs a two-stage transaction: it commits both report files first,
updates or creates the official Issue second, and commits `state/issues.json`
last. The stable Issue marker recovers safely if the final state commit fails.

For a local authenticated Codex backfill, set
`REPORT_SUMMARIZER_PROVIDER=codex`. GitHub Actions always uses the OpenAI
Responses API with strict JSON-schema output. `OPENAI_MODEL` can override the
configured `gpt-5` default.

## GitHub Actions setup

Add these repository secrets:

- `OPENAI_API_KEY`: used only for structured report generation.
- `MEGATRON_GH_TOKEN`: a least-privilege token that can read PR metadata and
  write Issues in `NVIDIA/Megatron-LM`.

For NVIDIA Inference Hub, set these repository variables:

- `OPENAI_BASE_URL=https://inference-api.nvidia.com/v1`
- `OPENAI_MODEL=<an exact model ID returned by the key's model catalog>`

`OPENAI_BASE_URL` may temporarily be stored as a secret and is supported for
backward compatibility, but it is not sensitive and should normally be a
repository variable. The Inference Hub API key itself must be stored under the
exact secret name `OPENAI_API_KEY`.

The workflow's repository `GITHUB_TOKEN` receives `contents: write` only so it
can commit report and state files. Set `REPORT_AUTOMATION_ENABLED=false` while
validating a manual dry-run. After a manual publication succeeds, change it to
`true`. Optionally set `OPENAI_MODEL`; otherwise `gpt-5` is used. The workflow
fails early with a clear error when either required secret is missing.

PR text is treated as untrusted input. It is bounded, passed as data, and cannot
alter the model role or publication target. Model output cannot introduce
unknown groups or PRs, move work between status sections, or change bilingual
citations.

## Development

```bash
.venv/bin/python -m pytest
```

The runtime DuckDB file and generated previews are ignored by Git. Actions
retains them as a 90-day audit artifact and restores the latest compatible cache;
if the cache is unavailable, the public GitHub data is collected again. A manual
workflow run can optionally supply `seed_run_id` to restore the DuckDB ledger
from that prior run's `activity-report-<run-id>` artifact.
When that artifact already contains a successfully collected copy of the exact
cutoff, the retry reuses the frozen window and does not spend API quota reacting
to PR updates that happened after the cutoff.
