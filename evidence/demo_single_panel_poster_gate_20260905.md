# Demo single-panel and poster gate — 2026-09-05

## Scope

The visual demo was reduced to one full-catalog recommendation panel. A viewer
selects one anonymous model-known user, sees five examples from that user's
observed history, and receives a fixed Top-10 from the frozen seed-42 BPR and
Weighted NeuMF models with the locked `0.7 / 0.3` percentile-rank blend.

Posters are presentation metadata only. They do not enter model features,
candidate filtering, score calculation, ranking, or evaluation.

## Gate result

```text
SINGLE_PANEL_GATE=PASS
anonymous_users=20
observed_rows=1798
warm_catalog_items=7223
poster_display_items=187
poster_cached=187
poster_missing=0
test_named_bundle_files=0
tests=38 passed
```

Browser verification confirmed that Streamlit exposes one page without the
automatic multipage sidebar, the selector reports 20 options, five history
posters render, and the Top-10 poster grid renders with names, genres, dataset
ratings, and stable ranks.

## Poster provenance and boundaries

- Public media metadata was queried from AniList using the dataset's
  MyAnimeList-compatible anime identifiers.
- The active manifest references the 187 unique anime that can appear in the 20
  users' five-item history preview or current Top-10. Unreferenced local cache
  files from earlier safe preparation attempts are ignored by the application.
- Anonymous showcase users were deterministically selected from model-known
  users while excluding profiles with adult-category observed history. This is
  a classroom-display choice, not part of model training or evaluation.
- Poster files and their index remain inside the ignored local demo bundle.
  Git contains only the preparation code, manifest template, and fallback SVG.
- The Demo makes no network request at runtime and falls back to a neutral
  placeholder when a poster is unavailable.
- External poster availability, metadata matching, licensing, and future API
  stability are presentation-layer limitations and must be disclosed in the
  report. They do not affect the recorded recommendation metrics.

## Evaluation boundary

The panel ranks the remaining warm catalog after filtering each user's observed
anime. The published NDCG@10 and Hit Rate@10 values were measured under the
fixed one-positive-plus-99-negative sampled-candidate protocol; they must not be
presented as full-catalog accuracy for this interface.
