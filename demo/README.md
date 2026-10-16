# Offline recommendation demo

The Streamlit demo shows viewing histories and full-catalogue Top-10 recommendations for 20 anonymous existing Anime users.
It uses the frozen seed-42 BPR ensemble component and Weighted NeuMF checkpoints, with a 0.7/0.3 percentile-rank blend.
The interface supports English and Chinese.

## Run

From the repository root:

```bash
python -m pip install -e ".[demo]"
python -m streamlit run demo/app.py
```

The app reads `demo_bundle/` by default. You can select another bundle:

```bash
# Bash
DAM_DEMO_BUNDLE=/path/to/demo_bundle python -m streamlit run demo/app.py
```

```powershell
$env:DAM_DEMO_BUNDLE = "C:/path/to/demo_bundle"
python -m streamlit run demo/app.py
```

## Local bundle

A source-only clone does not include the trained bundle.
It needs the two checkpoints, ID mappings, observed histories and metadata named in [manifest.example.json](manifest.example.json).
The example manifest describes paths and settings; it is not a replacement for those files.

Place the existing trained bundle in `demo_bundle/` before launching the app.
There is no standalone bundle-export command in this repository.
If assembling it from the original training outputs, use these filenames:

| Bundle file | Original source |
| --- | --- |
| `bpr_seed42_model.pt` | Ensemble-component BPR seed-42 `model.pt` |
| `weighted_neumf_seed42_model.pt` | Weighted NeuMF seed-42 `model.pt` |
| `user_mapping.parquet`, `item_mapping.parquet` | The matching Anime preparation artifacts |
| `demo_users.parquet` | Anonymous labels with model-known `user_id` values |
| `observed_by_user.parquet` | Observed `user_id` / `anime_id` pairs for those demo users |
| `manifest.json` | Filled-in dimensions, paths and settings from the example manifest |

Checkpoint embedding dimensions must match the manifest and mappings.
The original Anime checkpoints use BPR dimension 128 and Weighted NeuMF dimension 64, with hidden layers `[64, 32, 16, 8]`.
The catalogue metadata is already included at `demo/assets/anime_metadata.parquet`.

Once the bundle is in place, refresh the demo users and poster cache:

```bash
python -m demo.prepare_showcase --ratings data/rating.csv --bundle demo_bundle --user-count 20
```

`DAM_DEMO_BUNDLE` must point to the resulting bundle directory.
The preparation step downloads posters from AniList; the running app uses only local files.
Missing posters use the included placeholder.
Posters do not affect model scoring.

## Evaluation scope

The app ranks unseen items from the 7,223-item warm Anime catalogue.
It does not run the sampled-candidate test or use test predictions to produce recommendations.
The report's NDCG@10 and HR@10 therefore do not measure demo accuracy.

Keep checkpoints, original user identifiers, interaction histories and poster caches in the ignored `demo_bundle/` directory.
