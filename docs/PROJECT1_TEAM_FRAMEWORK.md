# Project 1 team framework

This is the working integration guide for the recommender-system technical
review. The course's Project 1 technical-review slides govern assessment and
submission; [`tasks.txt`](../tasks.txt) governs ownership. A method's presence
in Git does not by itself mean that its experiments or report contribution are
finished.

## What the project must demonstrate

| Course requirement | Team evidence to assemble |
|---|---|
| Recommender-system topic; at least two algorithms | Explain the selected methods and compare results produced by the team. |
| At least two datasets | Name both datasets, their preparation, and which methods ran on each. |
| Experimental analysis | Give parameter effects, strengths and weaknesses, performance factors, and component ablations where applicable. |
| Cases and presentation | Discuss representative successes and failures where applicable, with clear figures and conclusions. |
| Teamwork | Record each member's actual contribution and coordinate integration. |

The report later needs Abstract, Introduction, Methods, Experiments and
Conclusion, at least four pages in 11 or 12 pt type. The course lists three
submission files: `group_XX_report.pdf`, `group_XX_contribution.pdf`, and
`group_XX_code.zip`; the code ZIP excludes datasets and trained models, and
the submitted files must be self-contained rather than pointing to externally
hosted files. The slide deck gives 18 October 2026 as the submission deadline.
This framework does not create those files yet.

## Workstream ownership and current Git-visible state

| Workstream (`tasks.txt`) | Owner | Current entry point | Next integration evidence |
|---|---|---|---|
| Content-based | Siyuan | No implementation visible on the reviewed remote branches | Code and a filled handoff record |
| User-user neighbourhood | Zhewen | `Zhewen`: `src/models/user_user_knn.py`, `test_knn.py`, `test_knn_k.py` | Integrate code through its own PR; add evaluated results and handoff record |
| Item-item neighbourhood | Michael | `item_item_collab/item_item_collab.ipynb` on `main` | Make the notebook reproducible and add evaluated results and handoff record |
| Model-based | Lijie | `src/`, `configs/`, `evidence/`, `demo/` | Anime and MovieLens results are archived; integrate the workstream through the draft PR |

This table describes what was visible in Git on 28 September 2026. It does
not assess work done outside the repository. Keep the original authors and
workstream ownership when bringing code into `main`.

## Repository roles

- `tasks.txt`: named ownership and method descriptions; do not replace this
  with a new assignment system.
- `item_item_collab/` and `src/models/`: method implementations. A contributor
  may add a self-contained module or notebook without rewriting another
  member's method.
- `configs/`, `src/`, and `ops/`: reproducible preparation, training,
  evaluation, and analysis code where applicable. Not every method must use
  the model-based command-line pipeline.
- `evidence/`: small, versionable results and provenance. Keep raw datasets,
  predictions, checkpoints, and generated bulk outputs outside Git.
- `docs/`: method explanations and workstream handoffs. Use
  [`WORKSTREAM_HANDOFF_TEMPLATE.md`](WORKSTREAM_HANDOFF_TEMPLATE.md) for a
  concise, reviewable record from each owner.

## Integration sequence

1. Each owner supplies their code entry point and a filled handoff record on
   their own branch. A notebook is acceptable when its cells run in order and
   its data acquisition or input path is documented.
2. Run that workstream's reproducibility command and record its actual output.
   Review it in a PR against `main`. Keep prototype-only work labelled as such.
3. Assemble a comparison table only for methods measured under compatible
   dataset, split, candidate set, and metric definitions. If protocols differ,
   discuss behavior qualitatively and label the numbers separately. Do not
   infer that one dataset is easier from raw NDCG or HR across protocols.
4. After the method and evidence PRs land, choose the results and cases for
   the group report and record actual contributions. Package the source-only
   ZIP and check the three required filenames at submission time.

The model-based workstream already compares several algorithms on Anime and
MovieLens under its own frozen protocols. That protects the numeric minimum
of two methods and two datasets, but it does not replace the team's assigned
work or establish a fair cross-workstream ranking by itself.
