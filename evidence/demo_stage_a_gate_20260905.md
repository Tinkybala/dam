# Demo Stage A Gate — 2026-09-05

## Scope

The visual demo bundle was exported from the frozen final source commit
`b2f4d6b8222f9f5a9afd0633f54a235f50e52c69`.  Only two seed-42 inference
checkpoints, the finalized user/item mappings, the selected anonymous users'
observed history, and curated warm-catalog metadata are used.  Test candidates,
labels, predictions, other checkpoints, logs, and connection credentials are
excluded.

## Gate A result

```text
GATE_A=PASS
hash_files=7
checkpoint_load=PASS
mapping_shape=PASS users=60384 items=7223
demo_users=PASS count=12 observed_rows=2172
sensitive_scan=PASS matches=0
```

The BPR checkpoint loads with `(user_count=60384, item_count=7223,
embedding_dim=128)`.  The Weighted NeuMF checkpoint loads with
`embedding_dim=64` and hidden layers `[64, 32, 16, 8]`.  Both use CPU inference
in the demo process.

## Manifest file hashes

| Relative file | SHA-256 |
| --- | --- |
| `demo_bundle/bpr_seed42_model.pt` | `b6377b9946bdd4e833543afe5323eb190641e951969db3574738f926cf0f53aa` |
| `demo_bundle/weighted_neumf_seed42_model.pt` | `47aa13c98e583797f25c0ca69b9214635291325c3ada368e936f77ed69847fde` |
| `demo_bundle/user_mapping.parquet` | `2b133c1dd0efc3ddfa0298c1fd8bc2f16e368a8b7d4f5895454276060769bcd2` |
| `demo_bundle/item_mapping.parquet` | `83b1c797201906b5595e054046a6158e0fe895c8873371d8572530a8058f8d2f` |
| `demo_bundle/observed_by_user.parquet` | `13143dd4881b44ce578a853e705eba3b9b4493556084b4112bcd8e46f391ba5c` |
| `demo_bundle/demo_users.parquet` | `db00a1e3b12912e15ab72a00a623d9bf44ba926305ee79d4f6086690e814cd4f` |
| `demo/assets/anime_metadata.parquet` | `f4bc08293bacbd6c97a2f55c17ad4a7114db906748c35b18638d5f9d52b49994` |

The bundle is ignored by Git.  The one-time access key used for export was
revoked after hash verification and its local files were deleted.

