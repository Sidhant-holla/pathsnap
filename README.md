# PathSnap: Map-Matching & Route Reconstruction

PES University UE24CS352A (Machine Learning) hackathon project.

PathSnap matches noisy GPS trajectories to a road network and reconstructs the route that was driven.

1. Each GPS point gets a set of **candidate road segments** (arcs within 50 m).
2. Each (point, candidate) pair is described by 18 route-free features.
3. A **classifier** (LightGBM) scores the probability that each candidate is the correct road.
4. An **HMM decoded with Viterbi** turns those probabilities into a connected route, using
   road-network distances for the transition probabilities.
5. A **FastAPI** backend (Laptop 1) serves the pipeline, and a **Streamlit** UI (Laptop 2) lets an
   evaluator pick a trip and see the predicted route and its accuracy.

## Results

Student-test partition (10 trips), confident points.

| Method | Top-1 accuracy | Route continuity | Route F1 |
|--------|---------------|------------------|----------|
| Nearest road | 0.540 | 0.289 | 0.591 |
| HMM with distance-only emissions | 0.906 | 0.902 | 0.953 |
| LightGBM classifier only | 0.896 | 0.809 | 0.839 |
| **LightGBM + HMM (deployed)** | **0.928** | **0.904** | **0.974** |

Classifier comparison: GroupKFold(5) by trip on the 80 training trips, then refit and scored
on student-test. The full table is in `experiments/results/model_comparison.csv`.

| Model | Family | CV Top-1 | Student-test Top-1 | Log loss | Inference (ms / 1k rows) |
|-------|--------|----------|--------------------|----------|--------------------------|
| LightGBM (deployed) | Boosting | 0.881 | 0.896 | 0.060 | 2.6 |
| LightGBM, Bayesian-tuned | Boosting | 0.882 | 0.900 | 0.060 | 26.7 |
| LightGBM ranker | Learning to rank | 0.882 | 0.896 | 0.055 | 2.5 |
| Random Forest | Bagging | 0.878 | 0.890 | 0.063 | 1.8 |
| MLP | Neural network | 0.876 | 0.893 | 0.063 | 0.5 |
| KNN (distance-weighted) | Instance-based | 0.867 | 0.872 | 0.108 | 174.8 |
| Logistic Regression | Linear | 0.861 | 0.860 | 0.147 | 0.3 |
| Gaussian Naive Bayes | Probabilistic | 0.854 | 0.861 | 0.293 | 0.5 |

## Repository layout

| Folder | Purpose |
|--------|---------|
| `src/pathsnap/` | Core library: loading (`io`), projection (`geo`), labels (`labels`), candidates, features, preprocessing, models, evaluation, road graph, HMM/Viterbi (`hmm`), prediction pipeline, route metrics, Folium maps (`viz`) |
| `scripts/` | Entry points: feature building, training, HPO, ablations, HMM tuning, maps, API smoke test |
| `notebooks/` | Exploratory data analysis (`01_eda.ipynb`) |
| `api/` | FastAPI backend (Laptop 1) and the server-side scorer |
| `ui/` | Streamlit evaluator interface (Laptop 2) |
| `models/` | Saved models (`*.joblib`) and the HMM decoder configuration (`hmm_params.json`) |
| `experiments/` | Result CSV/JSON files, HPO study (`hpo/`), HMM results (`results/hmm/`), run logs |
| `reports/` | Figures (`figures/`), map screenshots (`figures/maps/`), interactive Folium maps (`maps/`) |
| `docs/` | Plan, deployment guide, data-quality log, report notes |
| `tests/` | Tests for metrics, the data pipeline, the leakage guard and the HMM |
| `data/raw/` | Raw trip files. **Not committed.** |

## Setup

Requires Python 3.11+ (the pinned numpy/pandas versions need it).

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

To keep the data elsewhere, set `PATHSNAP_DATA` to the folder that contains the partition folders.
The evaluator folder may also be named `evaluator`, `evaluator_testing`, `eval` or `test` (any case).

Check that everything works:

```bash
python -m scripts.verify_setup        # should end with "Setup OK"
python -m pytest -q tests             # metrics, pipeline, leakage guard and HMM tests
```

## Running the demo (two laptops)

Full guide, including the firewall rule and finding the IP address: [docs/DEPLOY.md](docs/DEPLOY.md).

```powershell
# Laptop 1: API (keep this console visible; it logs every request)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Laptop 2: UI
$env:API_URL = "http://<laptop-1-ip>:8000"
streamlit run ui/app.py
```

The prediction pipeline (`pathsnap.pipeline.predict_trip`) never reads the reference route. Only
`api/scorer.py` reads it, after the prediction, to compute accuracy on the server. Reference routes
are never sent to the UI.

## Reproducing the results

Run from the repository root, in this order. Seeds are fixed (42). The evaluator partition is
never used for training, tuning or selection.

| Step | Command | Output |
|------|---------|--------|
| 1. Exploratory analysis | open `notebooks/01_eda.ipynb` and run all | `experiments/results/eda/`, `reports/figures/eda/`, `reports/maps/eda_*.html` |
| 2. Candidate radius choice | `python -m scripts.radius_analysis` | `experiments/results/candidates/` and radius figures |
| 3. Feature tables | `python -m scripts.build_features` | `data/processed/*.parquet` (~15 s, not committed) |
| 4. Classifiers (CV + student-test) | `python -m scripts.train_models --models logreg rf lgbm gnb knn lgbm_ranker` | `experiments/results/<model>.json`, `models/<model>.joblib` |
| 5. MLP | `python -m scripts.train_mlp` | same, plus the per-epoch loss curve |
| 6. Bayesian HPO (LightGBM) | `python -m scripts.hpo --model lgbm --trials 50` | `experiments/hpo/`: Optuna study, trials, tuned vs untuned |
| 7. Model comparison | `python -m scripts.plot_model_comparison` | `experiments/results/model_comparison.csv`, `reports/figures/comparison/` |
| 8. Over/underfitting check | `python -m scripts.overfitting_check` | `experiments/results/overfitting_check.csv` and figure |
| 9. Feature-group ablation | `python -m scripts.feature_ablation` | `experiments/results/feature_ablation.csv` and figure |
| 10. HMM: tune on train, 4-way ablation on student-test | `python -m scripts.hmm_ablation all` | `models/hmm_params.json`, `experiments/results/hmm/` |
| 11. HMM fix / no-help maps | `python -m scripts.hmm_maps` | `reports/maps/hmm_*.html` |
| 12. Full-trip route maps and screenshots | `python -m scripts.report_maps` | `reports/maps/st_route_*.html`, `reports/figures/maps/*.png` |

See [docs/PLAN.md](docs/PLAN.md) for the design and module contracts, and
[docs/data_quality.md](docs/data_quality.md) for the data issues found and how they were handled.

## Data partitions

| Partition | Trips | Use |
|-----------|-------|-----|
| Training | 80 | Fitting, group-aware CV, Bayesian HPO, HMM decoder tuning |
| Student testing | 10 | Development-stage model comparison and the HMM ablation |
| Evaluator testing | 10 | Final evaluation only, served by the API. Never used for training, tuning or selection |
