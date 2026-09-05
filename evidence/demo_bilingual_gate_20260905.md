# Demo Bilingual UI Gate — 2026-09-05

## Scope

The single-panel Streamlit recommender now supports a complete Chinese/English
UI switch. Anime titles and genres remain the original dataset metadata rather
than being machine-translated or relabelled.

## Implemented

- One top-right language button switches the full interface between Chinese and English.
- Page title, instructions, user selector, metrics, section headings, rating labels,
  history table headings, model note, poster note, loading state, and bundle error are localized.
- Anonymous users display as `匿名用户 01` through `匿名用户 20` or
  `Anonymous User 01` through `Anonymous User 20`.
- The selected anonymous user is preserved across language switches.
- The translation table is centralized in `demo/i18n.py`.

## Verification

- `python -m pytest`: `41 passed`
- `git diff --check`: passed (only expected Git line-ending notices on Windows)
- Live Streamlit inspection at `http://127.0.0.1:8501`:
  - Chinese page rendered successfully.
  - English page rendered successfully.
  - Switching English `Anonymous User 02` to Chinese retained `匿名用户 02`.
  - Top-10 recommendations, posters, and explanatory notes rendered in both modes.

## Privacy and repository boundary

- No checkpoint, poster cache, interaction data, SSH material, API token, or local
  demo bundle is included in this change.
- `demo_bundle/` remains excluded from Git.
