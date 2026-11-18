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

Requires Python 3.10+ (developed on 3.11).

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
that contains the partition folders.

Then check everything works:

```bash
python -m scripts.verify_setup        # should end with "Setup OK"
```

See [docs/PLAN.md](docs/PLAN.md) for the approach, module contracts and team responsibilities.

## Data partitions

| Partition | Trips | Use |
|-----------|-------|-----|
| Training | 80 | Fitting, group-aware CV, Bayesian HPO |
| Student testing | 10 | Development-stage model comparison |
| Evaluator testing | 10 | Final evaluation only. Never used for training, tuning or selection |
