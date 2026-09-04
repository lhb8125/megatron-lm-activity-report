# Activity report design

## Reporting window and eligibility

Each report is a month-to-date snapshot from the first local calendar day of the
month through an inclusive cutoff in `Asia/Shanghai`. A PR is discovered from
GitHub search, but discovery alone never makes it report-eligible.

The raw ledger records openings, commits, merges, closes, and reopens. Narrative
eligibility requires substantive activity inside the current report window:

- a merge in the window; or
- an open-at-cutoff PR that was opened, received a commit, or was reopened in
  the window.

Closed-unmerged PRs remain in the ledger and aggregate counts but are not
narrated. Comments, reviews, labels, or a PR merely remaining open do not carry
it into a new month. Because the narrative window starts on the first day of the
month, a PR with no activity for more than one month cannot be cited.

Fact extraction receives only commit subjects whose commit timestamps fall in
the report window. The PR title, description, labels, and changed paths provide
identity and context, but the model is instructed not to restate prior-period
work unless current-window activity supports it.

The model returns prose classifications keyed by a constrained `group_id` enum.
Report section and PR-number citations are never model-authored: the application
attaches both from the trusted input group after generation and validates the
complete group set. This prevents a structurally valid model response from
mis-citing another PR.

Fact-cache keys include the month and extraction-policy version. Weekly updates
within one month reuse unchanged facts, while a new month or policy revision
cannot silently reuse facts generated from an older activity window.
The rebuildable DuckDB ledger is saved to the Actions cache even when a later
generation or validation step fails, so a retry does not need to fetch every PR
snapshot again.
Completed concurrent fetches are written as they arrive, making partial progress
durable if a later request reaches GitHub's rate limit. A manual recovery run may
also seed the ledger from a prior run's audit artifact by supplying its run ID.

## Theme and citation structure

Facts are aggregated independently into at most ten Delivered and ten In
Progress themes. Splitting the sections prevents a large Delivered input from
crowding out ongoing work. A non-empty fact section must yield at least one
theme; busy sections require approximately one theme per fifty facts, up to the
configured maximum. Every importance-5 fact is mandatory, so the model cannot
silently discard the highest-value work. Every theme contains one to four
subproject highlights. A highlight owns a non-empty,
ordered subset of the theme's change-group IDs. Highlights form an exact
partition of the theme's group IDs: no group can be missing, duplicated, or
assigned to another theme.

Markdown citations are derived from those audited highlight group IDs and
rendered directly after the corresponding bullet. There is no theme-level PR
dump. English and Chinese highlight text may differ, but highlight group IDs,
ordering, and resulting PR links must be identical.

Each English section is validated immediately. When an otherwise valid section
omits a small number of mandatory groups, a bounded repair call receives only
those facts and the existing theme skeleton. It assigns each fact to a distinct,
semantically matching highlight and revises that highlight's text. The complete
section is then validated again. No translation or publication occurs unless
both sections pass.

## Bilingual terminology

English is the structural source of truth. Chinese is a translation of the
validated English structure, not an independent summary. Technical proper nouns
are supplied to the translator as an exact-preservation glossary. The glossary
combines configured multi-word terms with dynamically detected acronyms,
model/kernel names, versioned names, CamelCase identifiers, and backtick-wrapped
identifiers from the English report.

Validation rejects a translation that changes theme IDs, group assignments,
highlight ordering, statistics, PR citations, or drops a required technical
term. Configured terminology lives under `translation.preserve_terms` in
`configs/megatron-lm.yaml`.

## Publication transaction

Generation and bilingual validation complete before any external mutation.
Publication commits both stable report paths, creates or updates the monthly
official Issue, then commits `state/issues.json`. A stable hidden marker recovers
Issue identity if the final state commit fails. Repeating the same publication
updates the existing Issue and creates no additional Git commit.
