# Anime report figures and interpretation — 2026-09-26

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: archived-result analysis and visualization
- Verification Status: final aggregates cross-checked against 19 archived run
  metrics; alpha scan checked against local summary and historical configuration
- Version Label: anime_report_figures_v2
- Owner: Lijie model-based recommender workstream

## Deliverables and reproduction

Three figures are available under `results/anime_report_figures_20260926_v2/`, each
as a 300-dpi PNG and vector SVG/PDF. English figure labels are ready for the
course report. That directory also contains full-precision CSV tables and a
SHA-256 manifest. The earlier v1 images are preserved in their original directory.
The v2 layouts are 7.4 inches wide with 8–13 pt text; check final sizing when
inserting them into the report. Generated files remain outside Git; the sanitized plotting
input and plotting script are versionable and contain no server coordinates.

```powershell
python -m pip install -e ".[analysis]"
python ops/build_anime_report_figures.py --output results/anime_report_figures_reproduced
```

Use a new output directory: the CLI refuses to overwrite an existing non-empty
directory. It loads only the sanitized aggregate/seed records, not datasets,
checkpoints or prediction files, and never trains or tunes a model.

| Figure | Supporting table | Purpose |
|---|---|---|
| `model_comparison` | `model_summary.csv`, `baseline_differences.csv` | Full-range scores plus paired differences from BPR |
| `component_ablation` | `paired_component_gains.csv` | All 12 paired gains across two components and both metrics |
| `parameter_analysis` | `alpha_scan.csv` | Controlled validation-only confidence-weight scan |

The component filename follows the earlier runbook; the v2 plot compares the
fixed fusion with each of its two actual components on NDCG and HR. It is not
a claim that the final NeuMF/Weighted NeuMF comparison isolates weighting.

The accompanying [English results/discussion draft](31_LIJIE_ANIME_RESULTS_DISCUSSION_DRAFT.md)
provides report-ready paragraphs and a claim-to-evidence table.

## Figure 1: final model comparison

**Suggested English caption.** Final Anime sampled-candidate test results for
60,384 warm users. Each user is evaluated with one held-out positive and 99
fixed unseen negatives. Bars show means and whiskers show sample standard
deviations across training seeds 42, 43 and 44; Popular is one deterministic
run. The fixed rank ensemble uses weights 0.7 for its BPR component and 0.3 for
Weighted NeuMF. Panels A/B show the full 0–1 metric range. Panels C/D show
model-minus-standalone-BPR differences: markers are the mean and whiskers the
sample SD of three paired seed differences, not confidence intervals. Popular
is omitted from paired panels because it has one deterministic run. These
scores do not measure full-catalog or online recommendation accuracy.

**中文解读。** 融合模型的 test NDCG@10 为 0.799658，HR@10 为 0.956335，
在当前比较中均最高。Standalone BPR 的 NDCG 为 0.770767，高于普通 NeuMF 的
0.765100，因此不能笼统声称深度模型一定优于传统模型。Weighted NeuMF 的 NDCG
高于普通 NeuMF，但其 HR 略低（0.946862 对 0.947094），两个指标并非始终同向。

## Figure 2: components and paired fusion gains

**Suggested English caption.** Paired gains from the fixed percentile-rank
ensemble over each actual component. Panel A shows NDCG@10 and panel B HR@10;
both use the same numerical delta scale. Point shapes identify seeds 42, 43
and 44, with a short horizontal line denoting the mean. All twelve paired
gains are positive. Points are training runs, not users; no significance
test is implied.
The BPR component uses a different preselected configuration from the standalone
BPR baseline in Figure 1. NeuMF and Weighted NeuMF also differ in embedding
size, learning rate and negative-sampling ratio; their final score difference
therefore cannot be attributed solely to confidence weighting.

**中文解读。** Ensemble 相对实际 BPR component 的平均 NDCG 增益为
0.009804，相对 Weighted NeuMF 为 0.015832；HR 平均增益分别为 0.006282 和
0.009473。两个指标、两个组件、三个 seed 的 12 个差值均为正。
这支持“固定融合在本次评估中改善两个组件的排名表现”。仅凭聚合指标，尚不能解释
它具体修复了哪些用户或电影的推荐错误，也不能把三次训练的 SD 当成显著性检验。

普通 NeuMF 与最终 Weighted NeuMF 的 embedding dimension 分别为 32 和 64，
learning rate 为 0.001 和 0.002，negatives per positive 为 8 和 24。
这些配置差异使最终测试比较不是单因素消融。真正只扫描 alpha 的结果在图 3。

## Figure 3: confidence-weight parameter analysis

**Suggested English caption.** Validation-only confidence-weight sweep from
Phase D, using seed 42 and 10,000 development users with 100 sampled candidates
per user. All configured hyperparameters other than alpha are held fixed;
validation-based early stopping can produce different realized epoch counts.
Both panels plot changes relative to alpha = 0, whose NDCG@10 and HR@10 are
0.767247 and 0.933700. Alpha = 0.5 gives the largest NDCG gain (+0.001352), while
alpha = 1.5 gives the largest HR gain (+0.006000). Selection used NDCG. No
error bars are shown because this scan used one seed per alpha. Green diamonds
and a shaded band identify the alpha=0.5 choice based on NDCG; they do not
indicate an uncertainty interval.

**中文解读。** 在匹配配置的 validation 扫描中，alpha=0.5 相比 alpha=0 的
NDCG 提升约 0.00135。继续增大 alpha 并未带来持续提升，alpha=1.5 和 2.0 的
NDCG 低于不加权基线。不过，alpha=1.5 的 HR 最高。这可以用于讨论权重强度与
排名质量之间的权衡，但单 seed 扫描不足以支持跨随机种子稳定有效的结论。

不要把这里的 0.768599 与全用户最终 test 的 0.783825 直接比较来归因改进：
用户范围和 evaluation split 不同。参数图也不能混入 9 月 1 日旧配置的 alpha
扫描；当前图只使用 9 月 2 日 Phase D 的五个匹配配置 trial。

## Provenance and verification limits

- Final source commit: `b2f4d6b8222f9f5a9afd0633f54a235f50e52c69`.
- Final summary SHA-256:
  `b9f34504da2adf564e23ed17b3215a2d582e2b164f13fc436393ee6ea8da2ddb`.
- Metadata archive SHA-256:
  `f8fbe19c4b8bacb4d8ef728155f1d5717c5e9c0e2680c922a60a3e22b7e6c74a`.
- All 19 individual archived model metrics were cross-checked against the final
  summary; seed means and sample SDs were independently recomputed. Ensemble
  values come from the hash-verified final summary; predictions were not rerun.
- Phase D summary SHA-256 and recorded release commit are stored in
  `evidence/anime_report_figure_data_20260926.json`. Its grid config matches the
  historical commit `06ffe48fa75cf3027e1651c2927a6d1c3fb66aa9`.
- Individual Phase D run metric files were not available in this local input;
  the parameter graph relies on the archived local sweep summary and matching
  configuration. It is not a new independent reproduction of training.
- Images were visually inspected; no overlapping labels or cropped text remain.
  The parameter chart uses explicitly labeled delta axes with distinct ticks.
- `python -m pytest`: 62 passed. Analysis tests verify sample SD, reject
  duplicate/missing training seeds, and check that baseline/fusion gains pair
  by seed rather than row order or the wrong BPR model. Baseline difference
  SD is computed from the paired differences, not independent error propagation.
  `git diff --check` passed.

## Remaining case analysis

The local final-test metadata archive contains no per-user prediction parquet
files. Success/failure/disagreement cases need those frozen predictions (and
history metadata) to choose examples using a deterministic rule. The existing
Demo output cannot substitute for formal test examples. No such cases are
claimed in this figure package, and no remote server was accessed for this work.
