# Demo Stage D Validation — 2026-09-05

## Automated checks

```text
python -m pytest -q
36 passed
python -m compileall -q demo
git diff --check
STREAMLIT_SMOKE status=200 body=ok
```

The tests cover checkpoint construction and loading, mapping validation, seen
item filtering, deterministic percentile tie-breaking, Top-K validation,
metadata missing-value handling, missing-bundle/hash fail-fast, and the
Streamlit entry-point import.  The smoke check started the app locally and
received the Streamlit health response on port 8765; the process was stopped
after the check.

## Isolation checks

```text
DEMO_TEST_READ_SCAN=PASS matches=0
DEMO_SENSITIVE_SCAN=PASS matches=0  (Python source and tracked text)
```

The recommender reads only `demo_bundle/` and
`demo/assets/anime_metadata.parquet`.  It does not open training, validation,
test candidate, or prediction files, and it contains no network or server
connection code.  All 12 real anonymous demo users were also scored
individually and returned a 10-row result.  The real bundle remains Git-ignored.
