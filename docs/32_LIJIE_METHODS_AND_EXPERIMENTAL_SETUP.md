# Methods and experimental setup: Anime recommendation

## Material Passport

- Origin Skill: academic-research-suite / academic-paper
- Scope: model-based workstream, Methods and experimental setup draft
- Version: v1, 2026-09-26
- Status: implementation and parameters checked against the final-run source and
  19 archived run configurations; ready for team editing
- Final-run source: `b2f4d6b8222f9f5a9afd0633f54a235f50e52c69`
- Evidence: `evidence/anime_methods_audit_20260926.json`

The English sections below describe the implementation actually evaluated.
They accompany the [parameter tables](33_LIJIE_FINAL_PARAMETER_TABLES.md) and
[results/discussion draft](31_LIJIE_ANIME_RESULTS_DISCUSSION_DRAFT.md). They cover
Lijie's experiments only; they do not impose the same protocol on teammates.
MovieLens uses a separate locked transfer experiment; its completed results are
in the [transfer record](../evidence/movielens_transfer_results_20260928.md).

## Methods

### Task and notation

We formulate the task as Top-N recommendation from rating-derived positive
feedback. For user \(u\), let \(O_u\) contain all retained observed items,
including low-rated and unrated interactions. Let \(P_u\) denote positive
interactions, defined by an Anime rating of at least 7. Models learn a score
\(s(u,i)\) used to rank candidate items. Scores are ranking signals rather than
predictions on the original 1–10 rating scale. The models use user and item IDs;
titles, genres and posters are not training features.

### Popular and BPR matrix factorization

The non-personalized Popular baseline assigns each item the number of distinct
users with a positive training interaction:

\[
s_{\mathrm{pop}}(i)=|\{u:(u,i)\in P_{\mathrm{train}}\}|.
\]

BPR learns user and item embeddings and an item bias,
\(s_{\mathrm{BPR}}(u,i)=\mathbf p_u^\top\mathbf q_i+b_i\).
Following the pairwise ranking approach introduced by Rendle et al. [1], the
implemented data loss for a batch of triples \((u,i,j)\) is

\[
\mathcal L_{\mathrm{BPR}}=-\frac{1}{|B|}
\sum_{(u,i,j)\in B}\log\sigma\bigl(s(u,i)-s(u,j)\bigr),
\]

where \(i\) is a training positive and \(j\notin O_u\) is sampled from the
warm item catalog. Training resamples negatives each epoch. Rejection sampling
excludes every retained observed user-item pair, including validation and test
positives, so they cannot become training negatives. Repeated negative draws
are allowed during training. The implementation uses Adam, and both final BPR
configurations set optimizer weight decay to zero. The standalone BPR baseline
and the BPR component used in fusion have different locked configurations.

### Neural collaborative filtering

We implement GMF, MLP and NeuMF architectures based on the neural collaborative
filtering framework of He et al. [2]. The equations below specify our code,
rather than asserting an exact reproduction of the original paper's training
procedure. GMF maps the elementwise product of user and item embeddings to a
scalar logit:

\[
s_{\mathrm{GMF}}(u,i)=\mathbf h^\top
(\mathbf p_u\odot\mathbf q_i)+b.
\]

MLP concatenates the embeddings and passes them through fully connected ReLU
layers, followed by a scalar linear output:

\[
\mathbf z_0=[\mathbf p_u;\mathbf q_i],\qquad
\mathbf z_\ell=\operatorname{ReLU}(W_\ell\mathbf z_{\ell-1}+\mathbf b_\ell),
\qquad s_{\mathrm{MLP}}=\mathbf h^\top\mathbf z_L+b.
\]

NeuMF concatenates a GMF interaction vector and an MLP representation before its
linear output. Its two branches have independent user/item embedding tables:

\[
s_{\mathrm{NeuMF}}(u,i)=\mathbf h^\top
[(\mathbf p_u^G\odot\mathbf q_i^G);\mathbf z_L^M]+b.
\]

All neural models are initialized from scratch. Embeddings use a zero-mean
normal initialization with standard deviation 0.01, linear weights use Xavier
uniform initialization, and biases start at zero. Final MLP/NeuMF configurations
use no dropout. Training applies binary cross-entropy with logits to training
positives and sampled unseen negatives. Evaluation ranks the raw logits.

### Normalized rating-confidence weighting

Weighted NeuMF retains the NeuMF architecture and changes the positive-example
loss weights. For training rating \(r_{ui}\), threshold \(\tau=7\), maximum
rating \(R=10\), and confidence strength \(\alpha\ge0\), define

\[
a_{ui}=1+\alpha\frac{r_{ui}-\tau}{R-\tau},\qquad
w_{ui}=\frac{a_{ui}}{\frac{1}{|P_{\mathrm{train}}|}
\sum_{(v,k)\in P_{\mathrm{train}}}a_{vk}}.
\]

Normalization is over all positive training examples used by that run, not over
each user or minibatch. Thus the average positive weight is one; lower-rated
positives may receive a weight below one after normalization. Sampled negatives
retain weight one. The implemented minibatch loss is

\[
\mathcal L_{\mathrm{point}}=\frac{1}{|B|}\sum_{(u,i,y)\in B}
\widetilde w_{ui}\left[-y\log\sigma(s(u,i))
-(1-y)\log(1-\sigma(s(u,i)))\right],
\]

with \(\widetilde w_{ui}=w_{ui}\) for positives and 1 for negatives. The
denominator is the number of examples, not the sum of weights. Alpha = 0
recovers unweighted pointwise training for the same architecture and settings.
The final Weighted NeuMF uses alpha = 0.5. The final ordinary NeuMF differs in
other hyperparameters too, so the controlled weighting comparison is the
matched Phase D alpha scan, not the final NeuMF-versus-Weighted-NeuMF gap.

### Fixed percentile-rank ensemble

To combine scores with different scales, we convert each component's scores
into per-user percentile ranks. For \(n_u>1\) candidates and descending rank
\(r_m(u,i)\), define

\[
\pi_m(u,i)=\frac{n_u-r_m(u,i)}{n_u-1},\qquad
s_{\mathrm{ens}}(u,i)=0.7\pi_{\mathrm{BPR}}(u,i)
+0.3\pi_{\mathrm{Weighted\ NeuMF}}(u,i).
\]

The top and bottom candidates receive percentiles 1 and 0. Component ties and
final ensemble ties are broken by ascending item ID. We require identical
user-item-label candidate tuples for the two components and pair their runs by
training seed. The blend weight was selected on validation and fixed before
the final test. Fusion trains no additional predictive model.

## Experimental setup

### Dataset, filtering and split construction

The primary dataset is the Anime Recommendations Database [3]. The local raw
rating file contains 7,813,737 rows. We remove every row belonging to a duplicated
user-item pair: seven ambiguous pairs account for fourteen removed rows. Ratings
of 7–10 are positive; ratings below 7 and the unrated marker -1 remain observed
interactions for exclusion from negatives.

We iteratively remove users and items with fewer than five positive interactions
until a positive 5-core is obtained. This filtering occurs before holdout
construction and yields 60,384 users, 7,223 items and 5,208,162 positives. For
each user, a seed-42 shuffle selects one validation positive and one test
positive, with all remaining positives assigned to training. If a held-out item
has no training interaction, deterministic within-user swaps repair the split
while retaining one validation and one test positive per user. The training
set contains 5,087,394 positive interactions.

| Dataset quantity | Recorded value |
|---|---:|
| Raw rating rows | 7,813,737 |
| Duplicate rows removed | 14, from 7 user-item pairs |
| Positive interactions before 5-core | 5,231,106 |
| Positive interactions after 5-core | 5,208,162 |
| Eligible users / warm items | 60,384 / 7,223 |
| Training positives | 5,087,394 |
| Validation / test positives | 60,384 / 60,384 |
| Retained observed pairs in warm user/item universe | 7,181,148 |
| Validation / test candidate rows | 6,038,400 / 6,038,400 |

This is a random warm-start protocol. Eligibility and exclusion masks use the
retained interaction record, whereas the loss uses only training positives.
It is not a chronological future-interaction or cold-start evaluation.

### Evaluation candidates and metrics

For each eligible user and each split, we combine the held-out positive with
99 distinct unseen warm items sampled without replacement. Validation and test
negative sampling use seeds 42 and 43, respectively. Candidate files are frozen
and reused across methods and training seeds. Validation and test negative sets
may overlap. Candidates are ordered by descending score, with exact ties broken
by ascending item ID.

Let \(r_u\) be the positive item's rank among that user's 100 candidates. With
one relevant item per user, the evaluated metrics simplify to

\[
\operatorname{HR@10}=\frac{1}{|U|}\sum_u\mathbb 1[r_u\le10],\qquad
\operatorname{NDCG@10}=\frac{1}{|U|}\sum_u
\frac{\mathbb 1[r_u\le10]}{\log_2(r_u+1)}.
\]

NDCG@10 is the primary selection metric and HR@10 is the secondary metric.
Results are averaged equally over users. For trainable systems, we then report
the mean and sample SD over seeds 42, 43 and 44, using divisor \(3-1\) for the
variance. These SDs are descriptive variation over training runs on one fixed
split, not confidence intervals over users or datasets.

### Training, selection and test separation

Development tuning uses a fixed 10,000-user subset. In those runs, training
positives are filtered to the selected users while the frozen warm-item mapping
is retained. Final runs use all 60,384 eligible users. Each epoch resamples
training negatives and evaluates the fixed validation candidates. Adam optimizes
the pairwise or pointwise objective; the configuration's `l2` field is passed as
Adam weight decay. It is zero except for GMF (\(10^{-6}\)) and MLP (\(10^{-5}\)).

The best checkpoint is the first to achieve the largest validation NDCG. A
strict improvement resets patience; five consecutive epochs without improvement
end training, subject to the configured maximum epoch count. The best checkpoint
is restored for reported validation and test scoring. Test evaluation was
disabled during tuning and enabled for the locked final-run configuration. In
final runs, test candidates are loaded at startup but are scored only after
validation-based checkpoint selection; test metrics do not select checkpoints.

There are 19 archived model runs: one Popular run plus three seeds for each of
six trainable configurations, counting standalone BPR and the BPR ensemble
component separately. Three ensemble results are subsequently derived from
matching component predictions. All trainable final runs use CUDA and GPU
negative sampling. The recorded server had two NVIDIA RTX A6000 GPUs, Python
3.11.15 and PyTorch 2.2.0+cu121. PyTorch's CUDA build is 12.1; the driver's
reported CUDA compatibility value of 12.4 is a different field.

Full settings, batch-size semantics and realized checkpoint epochs appear in
the [parameter tables](33_LIJIE_FINAL_PARAMETER_TABLES.md). Reproducibility is
supported by source commits, configurations, data hashes and archived metrics.
Seeding alone does not promise bitwise-identical CUDA results across platforms.

## References

1. S. Rendle, C. Freudenthaler, Z. Gantner and L. Schmidt-Thieme,
   “BPR: Bayesian Personalized Ranking from Implicit Feedback,” UAI, 2009,
   pp. 452–461. [Author-deposited paper and bibliographic record](https://arxiv.org/abs/1205.2618).
   The arXiv deposit is dated 2012; the conference paper is from 2009.
2. X. He, L. Liao, H. Zhang, L. Nie, X. Hu and T.-S. Chua,
   “Neural Collaborative Filtering,” 2017.
   [Author-deposited paper](https://arxiv.org/abs/1708.05031).
3. Cooper Union, “Anime Recommendations Database,” Kaggle.
   [Dataset source](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database).

Reference metadata was checked against the linked primary pages on 2026-09-26.
The dataset page did not expose its description through the browsing tool, so
the counts above come from the local manifest and archive, not a freshly read
web description. This manuscript describes the project's implementation and
does not claim novel invention of BPR/NCF or a complete reproduction of their
published experiments.

## Integration notes

- Place the Methods text before the existing results draft; retain the exact
  weighted-loss normalization and separate standalone/component BPR rows.
- The source map and detailed audit are supporting material, not report prose.
  Refer to `src/models/` for architecture, `src/training.py` for weights and
  sampling, `src/train.py` for optimization/selection, `src/data.py` and the
  frozen `src/prepare.py` for splits, and `src/evaluate.py`/`src/ensemble.py` for
  metrics and fusion. The current preparation code also has a MovieLens adapter;
  the draft was checked against the historical final-run source.
- Anime user- or item-level cases still require frozen per-user predictions.
  MovieLens has a separate anonymous case record and transfer summary. The
  full-catalog Demo output does not substitute for formal test evaluation.
