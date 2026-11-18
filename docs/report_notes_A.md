# Report notes, Member A: preprocessing, labels, candidates, features (report §2.2) and the MLP (§3.2)

These are **notes, not report text**. Write the sections in your own words; each of us must be
able to explain them in the viva. Every number comes from a file in the repo, named next to it.

Code: `src/pathsnap/{io,geo,labels,candidates,features,preprocessing,mlp}.py`,
`scripts/{verify_setup,radius_analysis,build_features,train_mlp,overfitting_check,feature_ablation}.py`,
tests in `tests/test_pipeline.py`.

---

## §2.2 Q1: How did we preprocess the data? (partition: train unless stated)

**What we were given:** per trip, `.track` (lon, lat, t), `.nodes` (lon, lat; row = node id),
`.arcs` (from, to; row = arc id, directed) and `.route` (ordered arc ids driven). There were no
features and no per-point labels, so both had to be built.

1. **Projection to metres** (`geo.project_trip`). Trips span 25 UTM zones (latitude -34.6 to 61.7), so
   each trip is projected to its own UTM zone. Check: ≤ 0.04% error vs exact WGS84 geodesic distance;
   haversine is ~0.3% off because it assumes a sphere (`scripts/verify_setup.py`).
2. **Per-point labels** (`labels.align_route`): a route-constrained dynamic programme. The states are
   positions in the route list; between consecutive points the car can only stay or move forward;
   the cost is the point-to-arc distance (not squared, to be robust to wild fixes).
   - Median distance from a point to its labelled arc: 4.9 m (confident train points).
   - Naive nearest-route-arc labelling disagrees with ours on up to 14.7% of a trip (trip 00000090),
     because of two-way streets and roads driven twice.
   - Points > 50 m from their arc are flagged `low_conf`: 11.2% of train points, almost all from
     off-network episodes (trip 54: 4.3 h inside a ~1 km² area with no mapped road).
     Models are trained on confident rows only; metrics are reported on confident (main) and all points.
3. **Segmentation at time gaps**: the track is split wherever Δt > 10 s (42 of 90 trips have such gaps).
   Smoothing and neighbour features never cross a gap.
4. **Candidate generation** (`candidates.py`): all arcs within 50 m of a point, at most the 16 closest,
   found with a spatial index (shapely STRtree) because networks have up to 307k arcs. Two-way twins
   share a rank so the cut-off never drops one direction.
   - Radius and cap chosen **on train only** (`scripts/radius_analysis.py`,
     `experiments/results/candidates/radius_analysis.csv`, `reports/figures/candidate_radius_tradeoff.png`):
     99% of confident points are within 34.5 m of their true arc; at r = 50 m, k = 16 the miss rate is
     0.46% with 7.3 candidates per point (`feature_build_summary.csv`). k = 24 would only gain 0.3 points.
5. **Scaling** (`preprocessing.make_scaler`, for LogReg/NB/KNN/MLP): log1p on the 10 heavy-tailed
   features, then StandardScaler, always fitted on the training fold only (inside the Pipeline).
6. **Result**: `data/processed/{train,student_test}_candidates.parquet` (1.52M train rows, 12.1%
   positives) and `*_points.parquet` (one row per GPS point, the denominator for top-1 and miss rate).

**Known label limitation (decided, documented in docs/PLAN.md):** at a U-turn on a two-way street both
directions are equidistant, so the alignment can pick the wrong direction on that segment. The optional
heading term (`heading_weight_m=10`) would change 0.95% of labels; an independent ±5 s heading check
agrees with the new label 874 vs 151 times. We kept version-1 labels (under 1% cannot change the model
ranking, and switching meant rerunning everything). This is a limitation of the reference labels (brief §6).

## §2.2 Q2: What patterns did we see, and what features did we extract, and why?

Key patterns (details and figures in the EDA notebook, B's notes):
- **Two-way ties**: for ~64% of points the nearest arc is tied with its reverse twin (same road, opposite
  direction). Nearest-road baseline: **55.9%** top-1; nearest + heading tie-break: **82.6%** (confident train points).
- GPS noise is mostly < 35 m; time gaps and off-network episodes exist (see above).

18 route-free features per (point, candidate), in 5 groups (`features.FEATURE_COLUMNS`):

| Group | Features | Why |
|---|---|---|
| Relative distance | `f_dist_gap` (distance minus the point's nearest candidate), `f_dist_rank`, `f_dist`, `f_n_cands` | the decision is which candidate wins *for this point*, so closeness relative to the alternatives matters |
| Direction | `f_heading_diff` (heading smoothed over ±2 s vs arc bearing), `f_heading_gap`, `f_heading_reliable` | the only signal separating the two directions of a two-way street |
| Neighbourhood | `f_ctx_dist` (mean distance of points t±1..3 to the arc), `f_ctx_gap` | one noisy fix can sit nearer the wrong road; its neighbours usually don't |
| Topology / geometry | `f_endpoint_dist` (junction proximity), `f_arc_len`, `f_proj_frac`, `f_in_deg`, `f_out_deg` | mistakes concentrate at junctions |
| GPS quality / motion | `f_speed`, `f_jitter`, `f_dt_prev`, `f_dt_next` | flag unreliable fixes |

No feature uses the route or labels; `tests/test_pipeline.py::test_features_do_not_depend_on_the_route`
checks that features are identical with and without the route loaded.

**Evidence the features matter** (all 5-fold CV on train, LightGBM):
- **Importance** (`experiments/results/lgbm_feature_importance.csv`, gain = total loss reduction of the
  splits using each feature): `f_dist_gap` 35.5%, `f_ctx_gap` 18.4%, `f_heading_diff` 14.2%,
  `f_heading_gap` 12.4%. Raw `f_dist` 0.4% (its information is already in `f_dist_gap`: correlated
  features share credit). Time-gap features ≈ 0%.
- **Group ablation** (`scripts/feature_ablation.py`, `experiments/results/feature_ablation.csv`,
  `reports/figures/feature_ablation.png`): baseline 0.881; removing direction → **0.591 (−29.0 pts)**,
  neighbourhood −1.2, topology −0.9, relative distance −0.4, GPS quality −0.2. **Every removal is
  worse in all 5 folds.** The best single group alone reaches only 0.57.
- Lesson: importance ≠ necessity. Relative distance has the largest gain share but neighbourhood features
  also encode distance; direction is the group the model cannot replace.
- We did **feature engineering, not feature selection**: all models use all 18 features.

---

## §3.2 MLP (`src/pathsnap/mlp.py`, `scripts/train_mlp.py`)

**A1. Why suitable?** Learns non-linear combinations of the features (e.g. heading only matters when the
distances are tied) without hand-written interactions, gives probabilities for the HMM, and is cheap at
inference. Different family (neural network) from the trees.

**A2. Strengths / limitations seen**
- Strengths: ST top-1 0.893 (second-best untuned model), **best classifier-only route continuity 0.819**,
  fast inference 0.45 ms per 1k rows (~6x faster than LightGBM).
- Limitations: needs scaling; slightly below LightGBM on CV (0.876 vs 0.881); the best stopping epoch
  varies a lot by fold (4 to 30), so training length is data-dependent; less interpretable than trees.

**B5. Hyperparameters and initial values** (`experiments/results/mlp.json`): hidden layers (64, 32), ReLU,
Adam, learning rate 1e-3, L2 alpha 1e-4, batch 1024, max 60 epochs, early-stopping patience 6,
validation = last 10% of training rows (trip-contiguous, because tables are sorted by trip, so whole trips
are held out). Seed 42.

**B6. Loss / objective**: binary cross-entropy (log loss) of "is this candidate the true arc", optimised by
Adam. No class weighting, to keep probabilities calibrated for the HMM.

**C7/C8. Student-test results** (10 trips): Top-1 0.8930 (confident) / 0.8917 (all points); candidate
macro F1 0.9343, observation macro F1 0.8413; precision 0.878, recall 0.889; log loss 0.0631; confusion
matrix [[TN 149,232, FP 2,354], [FN 2,131, TP 16,997]]. CV (5 folds): top-1 0.8758 ± 0.0256, candidate
macro F1 0.9254, log loss 0.0771. Per-fold top-1: 0.861, 0.912, 0.867, 0.891, 0.848.

**C9. Over/underfitting?** Mild, healthy fit.
- Training-data top-1 0.895 vs CV 0.876 vs ST 0.893: a gap of ~2 points (`overfitting_check.csv`).
- Real per-epoch curves (`reports/figures/mlp_loss_curve.png`, `mlp_loss_curve.csv`): validation loss
  flattens by epochs 4-12 and never rises; early stopping kept the epoch-12 weights.
- Validation loss is *below* training loss because the held-out block of trips happens to be cleaner than
  the training mix (training loss is measured on a sample that includes harder trips). Not leakage:
  those trips are never trained on.

**Train time** 48.7 s (80 trips). **Epoch curve allowed?** Yes: the MLP genuinely trains in epochs.

---

## Things to be ready to explain in the viva
- Why UTM per trip, not degrees or haversine.
- How the labels are built, and why monotone alignment beats nearest-arc labelling.
- Why 50 m / 16 candidates, and what "candidate miss rate" means.
- Why heading is essential (64% two-way ties; −29 points without it).
- Importance vs ablation: what "gain" means and why it differs from necessity.
- Why GroupKFold by trip and a trip-contiguous validation split for the MLP (1 s points are near-duplicates).
- Why the MLP's validation loss is below its training loss.
