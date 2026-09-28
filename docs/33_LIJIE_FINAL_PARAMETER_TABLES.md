# Final Anime parameter tables

Generated from the 19 archived run configurations and checked against both the
archived YAML files and current `configs/final/`. These tables describe the
completed Anime experiment, not MovieLens preparation.

Source commit: `b2f4d6b8222f9f5a9afd0633f54a235f50e52c69`.
Audit input: `evidence/anime_methods_audit_20260926.json`.

## Architecture and optimization

| System | Embedding dimension | Hidden widths | Adam learning rate | Adam weight decay | Confidence alpha |
|---|---:|---|---:|---:|---:|
| Popular | — | — | — | — | — |
| BPR (standalone) | 64 | — | 0.001 | 0 | — |
| GMF | 16 | — | 0.001 | 1e-06 | 0 |
| MLP | 64 | 128 → 64 → 32 | 0.001 | 1e-05 | 0 |
| NeuMF | 32 | 64 → 32 → 16 → 8 | 0.001 | 0 | 0 |
| Weighted NeuMF | 64 | 64 → 32 → 16 → 8 | 0.002 | 0 | 0.5 |
| BPR component | 128 | — | 0.00075 | 0 | — |

For NeuMF, the dimension applies separately to each of its four embedding
tables (GMF user/item and MLP user/item). The two branches do not share
embeddings. All final MLP/NeuMF dropout values are zero. GMF has an affine
output on elementwise products and no MLP hidden layers. Alpha is inactive for
BPR and zero for the unweighted pointwise models. Models are initialized from
scratch; pretrained GMF/MLP weights are not loaded into NeuMF.

## Training batch semantics and stopping

| Trainable configuration | Batch definition | Full-batch examples or triples | Negatives per positive | Max epochs | Patience |
|---|---|---:|---:|---:|---:|
| BPR (standalone) | 4,096 pairwise triples | 4,096 | 8 | 30 | 5 |
| GMF | 2,048 positive anchors | 18,432 | 8 | 20 | 5 |
| MLP | 2,048 positive anchors | 18,432 | 8 | 20 | 5 |
| NeuMF | 2,048 positive anchors | 18,432 | 8 | 30 | 5 |
| Weighted NeuMF | 2,048 positive anchors | 51,200 | 24 | 30 | 5 |
| BPR component | 4,096 pairwise triples | 4,096 | 24 | 30 | 5 |

BPR iterates over the positive set once for each requested negative draw per
epoch; its batch size counts triples, not a positive-plus-negatives pointwise
batch. Pointwise batches include each positive anchor and its sampled negatives
before shuffling. Final partial batches can be smaller than the counts above.
Evaluation batches contain at most 65,536 user-item pairs for every trainable
configuration. All 18 trainable runs specify `device: cuda` and
`gpu_sampling: true`. Popular has no gradient training.

## Realized epochs and restored checkpoints

| Configuration | Epochs completed (42 / 43 / 44) | Best validation epoch (42 / 43 / 44) |
|---|---|---|
| BPR (standalone) | 30 / 30 / 30 | 30 / 28 / 27 |
| GMF | 20 / 20 / 20 | 20 / 19 / 17 |
| MLP | 20 / 20 / 20 | 20 / 20 / 20 |
| NeuMF | 22 / 29 / 30 | 17 / 24 / 26 |
| Weighted NeuMF | 13 / 15 / 11 | 8 / 10 / 6 |
| BPR component | 22 / 19 / 18 | 17 / 14 / 13 |

The best checkpoint maximizes validation NDCG@10, with the earlier epoch kept
on exact ties. Test predictions use this restored checkpoint. A run reaching
the epoch cap does not necessarily use its final epoch for inference.

## Evaluation and fusion settings

| Setting | Value |
|---|---|
| Positive threshold / maximum rating | 7 / 10 |
| Positive core size | 5, before holdout construction |
| Split seed | 42 |
| Candidate seeds, validation / test | 42 / 43 |
| Training seeds | 42, 43, 44; Popular once at seed 42 |
| Evaluation K | 10 |
| Candidates per user per split | 1 positive + 99 distinct unseen negatives |
| Tie-break rule | Descending score, then ascending item ID |
| Final user count | 60,384 |
| Development tuning user count | 10,000 |
| Early-stop/checkpoint metric | Validation NDCG@10 |
| Ensemble | 0.7 × BPR-component percentile + 0.3 × Weighted-NeuMF percentile |
| Test configuration | Enabled only for the locked final runs |

## Recorded runtime environment

| Field | Value |
|---|---|
| python | 3.11.15 |
| torch | 2.2.0+cu121 |
| pytorch_cuda_build | 12.1 |
| driver_version | 550.54.15 |
| nvidia_smi_cuda_capability | 12.4 |
| numpy | 1.26.4 |
| pandas | 2.2.3 |
| pyarrow | 17.0.0 |
| GPUs | 2 × NVIDIA RTX A6000 |

This is the archived server environment, not the current local Python runtime.
CUDA build and driver compatibility version are distinct fields.

## Traceability and report placement

Use the architecture and training tables in Methods/Experiments. The per-seed
checkpoint and environment tables can be placed in an appendix if space is
limited. Keep standalone BPR and the fusion component as separate rows.

The audit JSON records all 19 configuration paths, full values, selected user
counts, training-positive counts and checkpoint epochs. It also includes the
raw-rating checksum check, local manifest hash, archived environment hash and
hashes of the historical source files. No dataset, checkpoint, prediction or
server coordinate is embedded in these tables.
