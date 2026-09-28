# Workstream handoff template

Copy this file to a named record under `docs/` when a method is ready for team
review. Keep the record factual; write `not applicable` with a reason when a
field does not fit the method. Do not paste raw data, private credentials,
checkpoints, or model weights into the repository.

## Ownership and implementation

- Owner and `tasks.txt` workstream:
- Method name and short explanation:
- Source branch and commit:
- Code/notebook path:
- Setup and exact run command (or ordered notebook steps):
- Inputs and how another member obtains them:

## Experiment protocol

- Dataset name, version/source, and preprocessing:
- Population and item counts after filtering:
- Train/validation/test split and random seed, if used:
- Candidate set or recommendation universe:
- Metric definitions, including K for Top-K measures:
- Parameters varied and selection rule:
- Baseline or comparison method:

## Results and interpretation

- Results table path and the actual numbers:
- Parameter or ablation analysis, if applicable:
- Strengths, weaknesses, and main performance factors:
- One success and one failure or boundary case, if applicable:
- Known limitations and what the results do **not** establish:
- Small evidence/figure paths suitable for Git:

## Reproduction check

- Environment and command used for the check:
- Date and outcome:
- Any missing input or unresolved failure:

Only put methods in the same numeric comparison table when their dataset,
split, candidate set, and metric definitions are compatible. Otherwise state
the differences next to the numbers and compare conclusions qualitatively.
