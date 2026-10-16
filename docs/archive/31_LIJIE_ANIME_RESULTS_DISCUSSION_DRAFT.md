# Anime results and discussion draft

## Material Passport

- Origin Skill: academic-research-suite / academic-paper
- Scope: results interpretation from the existing frozen Anime experiment
- Version: v1, 2026-09-26
- Status: source-grounded draft for team editing; not a completed group report
- Inputs: `evidence/anime_report_figure_data_20260926.json`, final configs,
  and report figures v2

This draft covers the model-based Anime results. Figure numbers refer to the
three Anime figures. Protocol differences between the four methods are listed
in the current [result summary](../RESULTS.md).

Companion material: [Methods and experimental setup](32_LIJIE_METHODS_AND_EXPERIMENTAL_SETUP.md)
and [verified parameter tables](33_LIJIE_FINAL_PARAMETER_TABLES.md).

## Model comparison

We evaluated recommendation quality on 60,384 warm users, with one held-out
positive and 99 fixed unseen negatives per user. We report NDCG@10 as the primary
metric and Hit Rate@10 as the secondary metric. Trainable systems were evaluated
at seeds 42, 43 and 44; the reported dispersion is the sample standard deviation
across these three training runs. Popular is deterministic and has one run.

The fixed ensemble achieved the highest mean scores among the compared systems:
0.799658 NDCG@10 and 0.956335 HR@10. Its absolute improvements over the standalone
BPR baseline were 0.028891 and 0.003632, respectively (Figure 1). However, neural
models did not uniformly outperform BPR. GMF, MLP and ordinary NeuMF all had lower
mean NDCG and HR than standalone BPR under their locked configurations. Weighted
NeuMF improved NDCG over BPR by 0.013058 but reduced HR by 0.005840.

NDCG rewards a higher rank, while HR counts any top-ten hit. Weighted NeuMF
improved the former and reduced the latter. These comparisons apply to the
tested configurations and do not establish an ordering between neural and
non-neural model families.

## Ensemble and component comparison

The ensemble combines per-user percentile ranks with fixed weights of 0.7 for
the BPR component and 0.3 for Weighted NeuMF. The BPR component is a separately
selected configuration and must be distinguished from the standalone BPR
baseline. Against this stronger component, the ensemble improved mean NDCG by
0.009804 and HR by 0.006282. Against Weighted NeuMF, the corresponding gains were
0.015832 and 0.009473. All twelve paired differences across two components, two
metrics and three training seeds were positive (Figure 2).

The aggregate gains show that blending improved on both components in these
runs, but do not identify which users or items benefited. The ensemble requires
loading and scoring two models, then normalizing and combining their ranks.
Production serving latency was not measured.

## Parameter analysis: confidence strength and metric choice

Figure 3 isolates alpha in the Phase D validation scan. The scan used 10,000
development users, seed 42 and the same 100-candidate evaluation design.
Configured hyperparameters other than alpha were fixed, although early stopping
produced different realized training lengths. Relative to alpha = 0, alpha = 0.5
increased NDCG from 0.767247 to 0.768599, an absolute gain of 0.001352. This was
the highest NDCG among the five tested values and supported selection by the
declared primary metric.

Larger weights did not produce a monotonic improvement. Alpha = 1.5 had the
highest HR, 0.939700, but its NDCG was 0.000729 below the alpha = 0 baseline.
This trade-off explains why optimizing HR alone would have selected a different
configuration. The scan provides single-seed evidence for the choice made on
validation; it does not demonstrate that the weighting benefit repeats across
seeds. Moreover, the final Weighted NeuMF and ordinary NeuMF configurations also
differ in embedding size, learning rate and negative-sampling ratio. Their final
test-score gap cannot be attributed solely to confidence weighting.

## Limitations and cross-dataset results

Our conclusions apply to the warm-user, warm-item sampled-candidate task. The
high test HR does not mean that the system achieves the same rate against the
full catalog or in online use. Three training seeds describe run-to-run
variation on one fixed split; they do not measure uncertainty over alternative
splits, user populations or deployments, and no statistical-significance claim
is made from these standard deviations. Pairing by seed supports transparent
run comparisons but does not make different architectures share identical
initialization or optimization trajectories.

Deterministic Anime success, failure and disagreement cases still require the
frozen per-user prediction files; the available local Anime metadata archive is
insufficient. MovieLens transfer results have now been completed under a locked
configuration. The common-model NDCG ordering and positive gains from the fixed
ensemble were observed on both datasets, but the direction of the Weighted
NeuMF-versus-BPR HR difference changed. See the separate
[MovieLens results record](../../evidence/movielens_transfer_results_20260928.md).
The Anime test remains frozen, and neither dataset's final test was used to
select new hyperparameters.

## Evidence for the reported conclusions

| Claim | Evidence location | Required qualification |
|---|---|---|
| Ensemble has the highest mean NDCG and HR in the compared systems | Figure 1 A/B; `model_summary.csv` | This sampled-candidate protocol only |
| Neural alternatives do not all improve on standalone BPR | Figure 1 C/D; `baseline_differences.csv` | Locked system/configuration comparison |
| Fusion improves both actual components on both metrics for every recorded seed | Figure 2; `paired_component_gains.csv` | Three seeds; no user-level significance test |
| Alpha=0.5 maximizes NDCG within the tested grid | Figure 3; `alpha_scan.csv` | Single-seed validation evidence |
| Confidence weighting alone explains the final NeuMF gap | Not supported | Do not claim; multiple hyperparameters differ |
| Fusion corrects particular user-level failure modes | Not established locally | Requires frozen per-user predictions |
| Common-model NDCG order and positive ensemble gains repeat on MovieLens | [MovieLens results](../../evidence/movielens_transfer_results_20260928.md) | Two tested datasets; broader generalization remains untested |
