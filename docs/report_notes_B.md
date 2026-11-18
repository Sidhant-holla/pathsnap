# Report notes, Member B: dataset exploration (report §2.2), Naive Bayes and KNN (§3.2), comparison plots (§3.3), maps (§6)

These are **notes, not report text**. Write the sections in your own words; each of us must be
able to explain them in the viva. Every number comes from a file in the repo, named next to it.

Code: `notebooks/01_eda.ipynb`, `src/pathsnap/viz.py`, `src/pathsnap/models_b.py`,
`scripts/{plot_model_comparison,report_maps}.py`. Data-quality log: `docs/data_quality.md`.

---

## §2.2 Dataset understanding and EDA

**State the partition for every analysis.** The notebook uses train + student-test (90 trips) and never opens
the evaluator partition. Each section in the notebook names its partition.

**Organisation:** `data/raw/<partition>/<trip_id>/<trip_id>.{track,nodes,arcs,route}`. 80 train trips,
10 student-test trips, 227,577 GPS points (208,417 / 19,160), ~90 MB (`experiments/results/eda/partition_summary.csv`,
`trip_inventory.csv`). Every trip has all 4 files. Files are tab-separated with no header; `.track` uses Windows line endings.
Longitude comes first. Node and arc ids are 0-based row numbers. **Each trip has its own road network.**

**Missing-value analysis** (`missing_invalid_summary.csv`, figure `reports/figures/eda/02_missing_seconds_per_trip.png`):
- 0 NaN values, 0 impossible coordinates, 0 duplicate or decreasing timestamps in any file.
- 3 trips (02, 71, 97) have fractional timestamps (9,659 points), treated as continuous seconds.
- "Missing" in time is the real issue: 96.7% of steps are 1 s, but 42 of 90 trips have a gap > 10 s and
  23 have a gap > 60 s. The median trip has no fix for 3.3% of its duration; trip 90 has a 42.5 h gap
  (99.6% of its time span), trip 78 has a 14.7 h gap (`03_sampling_gaps.png`).
- Decision: **no imputation**. Interpolating GPS would invent evidence. Instead the track is split into
  segments at gaps > 10 s.

**Other patterns:**
- 151 implausible jumps (> 252 km/h between fixes) in 11 trips, mostly trip 13 (95) and trip 32 (43). Kept.
- **Geographic extent**: 25 UTM zones, latitude −34.6 to 61.7, so each trip is projected separately
  (`04_trip_locations.png`, interactive `reports/maps/eda_trip_overview.html`).
- **Trajectories** (`05_trajectory_samples.png`): 12 training trips; speeds median 40 km/h, 99th percentile 167 km/h (`06_speed_stationary.png`).
- **Road networks** (`network_summary.csv`, `07_network_structure.png`): 649 to 307,392 arcs per trip.
  About 95% of arcs (median per trip) have a reverse twin (two-way streets). No isolated nodes. The largest
  strongly connected component holds 96% of nodes (median), but there are many small components
  (median 127 per train trip) from clipping at the edge. 1,903 duplicate arcs, 106 zero-length, 24 self-loops.
- **Reference routes** (`route_stats.csv`): 59 to 2,266 arcs; 79/90 fully connected, 11 trips have 1–3 breaks;
  10 trips drive some arc twice. So the reference is imperfect "reference", not ground truth.
- **GPS vs route** (`08_gps_vs_route.png`): median 5.7 m from the aligned route arc (all train points); 11.2% of
  train points are > 50 m away, mostly **off-network episodes**: trip 54 spends 4.3 h (15,584 points) inside
  a ~1 km² area with no mapped road, ending within 6 m of where it started (`09_off_network_trip54.png`).
- **Two-way ties**: for ~64% of points (mean per trip) the nearest arc is tied with its reverse twin, the key
  reason heading features were built.
- **Anomaly flags** per trip: `anomaly_flags.csv` (gaps, jumps, off-route, route breaks, fractional time).

**Required visuals:** GPS trajectory plots (`05_*`), road-network visualisation (`07_*` and the network layer of
every map), missing-value analysis (`02_*`, `03_*`), Folium maps (`reports/maps/eda_trip_00000001.html`,
`eda_trip_00000054.html`, `eda_trip_00000090.html`, `eda_trip_overview.html`; screenshots in
`reports/figures/maps/eda_trip_*.png`).

---

## §3.2 Gaussian Naive Bayes (`models_b.make_gnb`, ZOO name `gnb`)

**A1. Why suitable?** The fastest probabilistic baseline: a closed-form generative model giving class
probabilities. It tests whether the features are useful *independently* of each other.

**A2. Strengths / limitations seen**
- Strengths: trains in 0.8 s, inference 0.48 ms per 1k rows; ranks candidates reasonably (ST top-1 0.861).
- Limitation: it assumes the features are independent given the class, but four features all measure distance
  (`f_dist`, `f_dist_rank`, `f_dist_gap`, `f_ctx_dist`). It counts the same evidence several times, so its
  probabilities are **over-confident**: the worst log loss (ST 0.293, CV 0.399 ± 0.086), and high recall
  (0.909) but low precision (0.748): 5,854 false positives on student-test.

**B5. Hyperparameters:** `var_smoothing = 1e-9` (sklearn default), with the shared scaler (log1p + StandardScaler) first.

**B6. Objective:** maximum-likelihood estimates of each feature's per-class mean and variance (no
iterative loss); prediction = Bayes' rule with the class priors.

**C7/C8. Student-test:** top-1 0.8611 (all points 0.8599); candidate macro F1 0.8976; observation macro F1 0.7290;
log loss 0.2933; continuity 0.746; confusion matrix [[145,732, 5,854], [1,744, 17,384]].
CV: top-1 0.8537 ± 0.0265, candidate macro F1 0.8865, per fold 0.835, 0.896, 0.846, 0.861, 0.831.

**C9. Over/underfitting:** **underfitting** (high bias). Training-data top-1 0.859 ≈ CV 0.854 ≈ ST 0.861
(`overfitting_check.csv`): no gap, but the lowest accuracy of all models. The model is too simple for
correlated features. No loss curve (no epochs).

## §3.2 KNN, distance-weighted (`models_b.SubsampledKNN`, ZOO name `knn`)

**A1. Why suitable?** Instance-based, so it makes no assumption about the shape of the decision boundary:
"a candidate whose features look like known correct candidates is probably correct".

**A2. Strengths / limitations seen**
- Strengths: CV top-1 0.867, ST 0.872 (better than the linear and probabilistic models); essentially no training (0.2 s).
- Limitations: the **slowest at inference**, 174.8 ms per 1k rows (~70x LightGBM), because every prediction searches
  the stored training set; the saved model contains the data (15 MB); needs scaling; lower recall (0.848).
- Adaptation: fitted on **every 5th GPS point** (all candidates of the kept points) to keep it practical; 1 s
  points are near-duplicates, so little is lost.

**B5. Hyperparameters:** k = 50 neighbours, distance weighting, Euclidean distance on the scaled features,
`point_step = 5`.

**B6. Objective:** none (lazy learner). Prediction = distance-weighted share of positives among the 50 nearest
training candidates.

**C7/C8. Student-test:** top-1 0.8721 (all 0.8709); candidate macro F1 0.9228; observation macro F1 0.8015;
log loss 0.1078; continuity 0.788; confusion matrix [[149,331, 2,255], [2,912, 16,216]].
CV: top-1 0.8668 ± 0.0280, candidate macro F1 0.9201, per fold 0.857, 0.910, 0.852, 0.877, 0.838.

**C9. Over/underfitting:** training-data top-1 0.907 vs CV 0.867 (+4.0 pts), but this is **memorisation by design**:
with distance weighting a stored point is its own nearest neighbour at distance 0. CV and ST agree (0.867 / 0.872),
so it generalises consistently; the train score just isn't meaningful for KNN. No loss curve.

---

## §3.3 Comparison plots (built by `scripts/plot_model_comparison.py` from the logged results)

All in `reports/figures/comparison/`, covering the 8 model variants:
- `cv_macro_f1.png`: CV macro F1 (candidate and observation level, mean ± fold std).
- `student_test_top1_f1.png`: student-test top-1 and macro F1.
- `log_loss.png`: CV and student-test log loss.
- `confusion_matrices.png`: candidate-level confusion matrices on student-test.
- `fold_stability.png`: top-1 per fold; LightGBM is ahead in every fold.
- `accuracy_vs_speed.png`: accuracy vs inference cost.
- `overfitting_check.png`: training data vs CV vs student-test.
Loss curves only for models that train iteratively: `reports/figures/{mlp,lgbm,lgbm_tuned,lgbm_ranker}_loss_curve.png`.
Table 3.1: `experiments/results/model_comparison.csv`.

## §6 Folium maps (screenshots for the PDF in `reports/figures/maps/`)

- **Full-trip predicted routes**, all 10 student-test trips: `st_route_<trip>.png` (interactive `reports/maps/st_route_*.html`).
  Layers: GPS observations, nearby road network, reference route (green), predicted route (red, classifier + HMM),
  start/end markers, legend and layer control. Built by `scripts/report_maps.py` with the same `predict_trip` the API uses.
- **HMM fixed it / didn't help** (D's windows): `hmm_fix_*.png`, `hmm_nohelp_00000077.png`, with the classifier-only
  route (orange) added.
- **EDA maps**: `eda_trip_00000001.png` (typical trip), `eda_trip_00000054.png` (off-network episode), `eda_trip_00000090.png` (42 h gap).
- Map design: routes are drawn above the GPS points so a dense trace can't hide them; reference routes appear only in
  student-test report maps, never in the UI.

---

## Things to be ready to explain in the viva
- How the dataset is organised and what each file means (row number = id; longitude first).
- Why there is no imputation, and what "missing" means for GPS (time gaps).
- What the off-network episodes are and why those points are flagged, not deleted.
- Why Naive Bayes has good top-1 but terrible log loss (independence assumption, correlated distance features).
- Why KNN is slow, why it was subsampled, and why its training score is meaningless.
- How to read each comparison plot, and what each map layer is.
