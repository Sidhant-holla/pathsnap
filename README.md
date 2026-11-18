# PathSnap: Map-Matching & Route Reconstruction

PES University UE24CS352A (Machine Learning) hackathon project.

PathSnap matches noisy GPS trajectories to a road network. Each GPS observation
gets a set of candidate road segments, which a trained classifier scores. HMM/Viterbi
decoding then reconstructs a connected route. A FastAPI backend serves the
pipeline, and a separate UI lets an evaluator choose a trip and view the result.

> Work in progress.

## Repository layout

| Folder | Purpose |
|--------|---------|
| `data/raw/` | Raw trip files (`.track`, `.nodes`, `.arcs`, `.route`). **Not committed.** |
| `src/pathsnap/` | Core library: data loading, preprocessing, candidate generation, features, HMM/Viterbi |
| `notebooks/` | EDA and experiment notebooks |
| `scripts/` | Entry-point scripts (training, evaluation, HPO, map generation) |
| `experiments/results/` | CSVs, experiment logs, HPO trial records |
| `models/` | Saved model weights and inference config |
| `api/` | FastAPI backend (Laptop 1) |
| `ui/` | Evaluator-facing UI (Laptop 2) |
| `reports/` | Figures, Folium maps, technical report |

## Setup

Requires Python 3.11+ (pinned numpy/pandas versions need it; developed on 3.11).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                      # makes `import pathsnap` work everywhere
```

The dataset is not in the repository. Place it like this:

```
data/raw/
├── train/<trip_id>/<trip_id>.{track,nodes,arcs,route}          (80 trips)
├── student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}   (10 trips)
└── evaluator_test/<trip_id>/...                                 (10 trips, API machine only)
```

To read the data from somewhere else, set the `PATHSNAP_DATA` environment variable to the folder
that contains the partition folders. The evaluator folder may also be named `evaluator`,
`evaluator_testing`, `eval` or `test` (any case); see `PARTITION_ALIASES` in `src/pathsnap/io.py`.

Then check everything works:

```bash
python -m scripts.verify_setup        # should end with "Setup OK"
```

See [docs/PLAN.md](docs/PLAN.md) for the approach, module contracts and team responsibilities.

## Reproducing the results

Run from the repository root, in this order. Every number and figure in the report comes from
these commands. Seeds are fixed (42), and the evaluator partition is never read.

| Step | Command | Output |
|------|---------|--------|
| 1. Exploratory analysis | open `notebooks/01_eda.ipynb` and run all | `experiments/results/eda/`, `reports/figures/eda/`, `reports/maps/eda_*.html` |
| 2. Candidate radius choice | `python -m scripts.radius_analysis` | `experiments/results/candidates/`, radius figures |
| 3. Feature tables | `python -m scripts.build_features` | `data/processed/*.parquet` (~15 s, not committed) |
| 4. Models (CV + student-test) | `python -m scripts.train_models --models logreg rf lgbm gnb knn` | `experiments/results/<model>.json`, `models/<model>.joblib` |
| 5. MLP | `python -m scripts.train_mlp` | same, plus the per-epoch loss curve |
| 6. Bayesian HPO | `python -m scripts.hpo --model lgbm --trials 50` | `experiments/hpo/` (Optuna study, trials, tuned vs untuned) |
| 7. Model comparison | `python -m scripts.plot_model_comparison` | `experiments/results/model_comparison.csv`, `reports/figures/comparison/` |
| 8. Over/underfitting check | `python -m scripts.overfitting_check` | `experiments/results/overfitting_check.csv` + figure |
| Tests | `python -m pytest -q tests` | metric definitions, labels, candidates, features, leakage guard |

## Data partitions

| Partition | Trips | Use |
|-----------|-------|-----|
| Training | 80 | Fitting, group-aware CV, Bayesian HPO |
| Student testing | 10 | Development-stage model comparison |
| Evaluator testing | 10 | Final evaluation only. Never used for training, tuning or selection |
