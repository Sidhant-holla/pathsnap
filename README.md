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

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Put the raw dataset under `data/raw/`. The folder structure will be documented once
the dataset has been explored.

## Data partitions

| Partition | Trips | Use |
|-----------|-------|-----|
| Training | 80 | Fitting, group-aware CV, Bayesian HPO |
| Student testing | 10 | Development-stage model comparison |
| Evaluator testing | 10 | Final evaluation only. Never used for training, tuning or selection |
