# Report notes: models, selection and HPO (Member C)

Working notes for report sections 3 (models) and 4 (Bayesian HPO). Every number here comes
from a file in the repo (listed next to it), so it can be checked and re-generated. These
are notes and facts, not final report text: write the report sections in your own words.

Reproduce: `python -m scripts.build_features` → `python -m scripts.train_models`
→ `python -m scripts.hpo --model lgbm --trials 50` → `python -m scripts.train_models --models lgbm_ranker`
→ `python -m scripts.plot_model_comparison` → `python -m scripts.overfitting_check`.

---

## 1. Evaluation protocol (applies to every model)

- **Data:** one row per (GPS point, candidate arc); 18 route-free features (`features.FEATURE_COLUMNS`);
  label = candidate is the arc the route-constrained Viterbi alignment assigns to that point.
  Train: 80 trips, 1,523,093 candidate rows, 208,417 points, 12.1% positives.
  Student-test: 10 trips, 170,774 rows, 19,160 points.
- **CV:** `GroupKFold(5, shuffle=True, random_state=42)` grouped by `trip_id`, so no trip is ever
  split between training and validation (1-second neighbours of the same trip would leak).
  Same folds for every model and for every HPO trial.
- **Training rows:** confident rows only (point ≤ 50 m from the reference route; 1,487,898 of
  1,523,093 rows). Low-confidence points are mostly off-network episodes (trips 54, 33) whose label
  is unreliable. Validation trips are always scored on **all** rows.
- **Preprocessing inside the model** (`models.py`): scale-sensitive models get the shared
  `preprocessing.make_scaler` (log1p on skewed features → StandardScaler) as the first Pipeline
  step, so CV refits it on each training fold. Trees use raw features.
- **Student-test** is used only for development-stage comparison after CV. The evaluator
  partition is never loaded for training, tuning or selection.
- **Timing:** train time = fit on all 80 trips; inference = `predict_proba` on the whole
  student-test table, reported per 1,000 candidate rows (a typical trip has ~17k rows).
- **Seed 42** everywhere. Results: `experiments/results/<model>.json`, per-fold `<model>_cv_folds.csv`,
  one row per model in `summary.csv`.

### Metric definitions (`src/pathsnap/evaluate.py`)

| Metric | Definition |
|---|---|
| **Top-1 (confident)** | **Primary.** Per confident GPS point, argmax candidate == labelled arc. Denominator = all confident points from the points table, so points whose labelled arc is not a candidate (or that have no candidates) count as wrong. |
| Top-1 (all) | Same over all points. |
| Candidate macro F1 | Binary "is this candidate the labelled arc" at threshold 0.5, F1 of both classes averaged. Confident rows. |
| Observation macro F1 | Predicted arc vs labelled arc per point; F1 per arc class (class = trip_id:arc_id), macro-averaged. |
| Log loss | Binary cross-entropy of the candidate probabilities, confident rows. |
| Continuity (argmax) | Share of consecutive distinct predicted arcs that are connected (a.v == b.u), classifier-only route, all points. HMM continuity is in D's results. |
| Miss rate | Share of points whose labelled arc is not among the candidates: 0.46% of confident train points (`experiments/results/candidates/feature_build_summary.csv`). An upper bound no classifier can beat. |

**Why Top-1 is the primary metric (points to make):**
- The route is built from each point's best-scoring candidate (and the HMM re-ranks candidates);
  nothing in the system applies a 0.5 threshold, so threshold metrics don't describe what the
  system does.
- Candidate-level accuracy is dominated by easy negatives (88% of rows are negatives); a model that
  says "no" everywhere gets 88% accuracy and 0% Top-1.
- Log loss is the secondary criterion because the HMM uses the probabilities as emissions
  (normalised per point), so calibration matters for decoding.

---

## 2. Consolidated table (C's models; full 7-model table in `experiments/results/model_comparison.csv`)

| Model | Family | Key configuration | CV cand. macro F1 | CV Top-1 (conf) | Continuity (ST, argmax) | ST cand. macro F1 | ST log loss | Inference (ms / 1k rows) |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear | C=1, class_weight=balanced, lbfgs, shared log1p+standard scaler | 0.8793 | 0.8608 ± 0.0267 | 0.7733 | 0.8839 | 0.1473 | 0.34 |
| Random Forest | Bagging ensemble | 200 trees, max_features=sqrt, min_samples_leaf=25, max_samples=0.2 | 0.9281 | 0.8777 ± 0.0276 | 0.8031 | 0.9360 | 0.0630 | 1.79 |
| **LightGBM** | Boosting | lr 0.05, 63 leaves, min_child_samples 50, subsample/colsample 0.8, λ=1, early stopping → 328 trees | **0.9298** | **0.8811 ± 0.0280** | 0.8085 | **0.9387** | 0.0600 | 2.58 |
| LightGBM (tuned) | Boosting | best HPO trial #24 → 976 trees (see §4) | 0.9299 | 0.8822 ± 0.0292 | 0.8100 | 0.9394 | 0.0595 | 26.74 |
| LightGBM ranker (stretch) | Learning to rank | LambdaRank, query = one GPS point's candidates, same tree params, NDCG@1 early stopping → 202 trees | 0.9197 | 0.8816 ± 0.0279 | 0.8022 | 0.9430 | 0.0545 | 2.53 |

Student-test point-level results (`<model>.json → student_test`):

| Model | ST Top-1 (conf) | ST Top-1 (all) | ST obs. macro F1 | Precision | Recall | Confusion [[TN, FP], [FN, TP]] | Train time |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.8600 | 0.8588 | 0.7701 | 0.689 | 0.951 | [[143355, 8231], [932, 18196]] | 2.7 s |
| Random Forest | 0.8903 | 0.8890 | 0.8380 | 0.896 | 0.877 | [[149631, 1955], [2353, 16775]] | 50.1 s |
| LightGBM | 0.8964 | 0.8951 | 0.8485 | 0.895 | 0.887 | [[149589, 1997], [2153, 16975]] | 22.7 s |
| LightGBM (tuned) | 0.8997 | 0.8984 | 0.8501 | 0.897 | 0.888 | [[149630, 1956], [2145, 16983]] | 92.6 s |
| LightGBM ranker | 0.8957 | 0.8944 | 0.8524 | 0.929 | 0.870 | [[150317, 1269], [2497, 16631]] | 18.5 s |

Baselines for context (A, confident train points, `docs/PLAN.md`): nearest arc 55.9% Top-1
(71% of points have a two-way tie at the nearest distance); nearest + heading tie-break 82.6%.

Figures: `reports/figures/comparison/` (CV macro F1, ST Top-1/F1, log loss, fold stability,
accuracy vs speed, confusion matrices, overfitting check), `reports/figures/lgbm_loss_curve.png`,
`lgbm_tuned_loss_curve.png`, `lgbm_ranker_loss_curve.png`.

---

## 3. Per-model notes (answers to the compulsory questions, as evidence)

### 3.1 Logistic Regression (`models.py: "logreg"`)

- **Why considered:** linear baseline. Several features are already "relative" (distance gap to the
  best candidate, heading gap, context gap), so a weighted sum of them should already rank candidates
  reasonably. Gives interpretable coefficients and tells us how much non-linearity is worth.
- **Strengths seen:** fastest (0.34 ms / 1k rows, 2.7 s to train on 1.49M rows); coefficients agree with
  the geometry: largest standardised weights are `f_ctx_gap` −1.35, `f_heading_diff` −1.20,
  `f_heading_gap` −1.10, `f_endpoint_dist` +1.05, `f_dist_rank` −0.98 (`logreg_coefficients.csv`).
- **Limitations seen:** lowest Top-1 of C's models (CV 0.861, ST 0.860), 0.8–2.9 points behind LightGBM
  in every fold, so the decision boundary is not linear in these features (e.g. heading matters a lot
  only when two candidates are at the same distance: an interaction a linear model can't express).
- **Hyperparameters (initial = final):** C = 1.0 (L2), `class_weight="balanced"`, lbfgs, max_iter 2000.
- **Loss:** L2-regularised binary cross-entropy, with balanced class weights (12.4% positives on
  confident rows, so each positive row weighs ~7× a negative one).
- **Class weighting effect:** recall 0.951 but precision 0.689 at threshold 0.5, and log loss 0.147
  (2.5× LightGBM's). Balanced weights shift every probability upward, so the probabilities are
  poorly calibrated. This matters for the HMM emissions; it doesn't change the per-point ranking much.
- **Over/underfitting:** train Top-1 0.866 vs CV 0.861 vs ST 0.860: gap 0.005. Train and held-out
  agree but all are lower than the trees, so this is **underfitting** (too little capacity), not
  overfitting (`overfitting_check.csv`).

### 3.2 Random Forest (`"rf"`)

- **Why considered:** non-linear, handles feature interactions (distance × heading) and mixed scales
  without preprocessing; bagging gives stable probabilities; robust to the noisy low-confidence labels.
- **Strengths seen:** second-best of C's models (CV 0.878, ST 0.890); log loss 0.063 (well calibrated);
  same top features as LightGBM: `f_heading_diff` 0.21, `f_heading_gap` 0.18, `f_dist_gap` 0.18
  (`rf_feature_importance.csv`).
- **Limitations seen:** slowest training (50 s vs 23 s for LightGBM); behind LightGBM in all 5 folds;
  a full-size forest on 1.5M rows would be hundreds of MB, so trees were limited (below).
- **Hyperparameters:** 200 trees, `max_features="sqrt"`, `min_samples_leaf=25`, `max_samples=0.2`
  (each tree sees a 20% bootstrap: faster training and a 17 MB model that fits in the repo),
  no class weights.
- **Loss / objective:** each tree greedily minimises Gini impurity; probabilities = mean of leaf
  class fractions over trees.
- **Over/underfitting:** train 0.898 vs CV 0.878 (gap 0.020), ST 0.890. Mild; leaf size 25 and 20%
  subsampling keep it in check.

### 3.3 LightGBM (`"lgbm"`): selected model

- **Why considered:** gradient boosting is usually the strongest family on tabular features like
  these; handles interactions, missing-free numeric features and 1.5M rows quickly; gives real
  per-iteration loss curves; produces probabilities the HMM can use.
- **Strengths seen:** best CV Top-1 (0.8811) and best ST Top-1 (0.8964) of the untuned models; best
  candidate macro F1 (CV 0.930 / ST 0.939); lowest log loss among classifiers (CV 0.073 / ST 0.060);
  beats Random Forest and Logistic Regression in **every** fold (`*_cv_folds.csv`):

  | Fold | LogReg | RF | LightGBM |
  |---|---|---|---|
  | 0 | 0.8441 | 0.8632 | 0.8692 |
  | 1 | 0.9049 | 0.9182 | 0.9209 |
  | 2 | 0.8484 | 0.8690 | 0.8725 |
  | 3 | 0.8667 | 0.8912 | 0.8957 |
  | 4 | 0.8398 | 0.8471 | 0.8474 |

  The ±0.028 std is mostly *between-trip difficulty* (the same folds are hard for every model), so
  per-fold comparison is the fairer test than overlapping error bars.
- **Limitations seen:** most capacity → largest train/CV gap (below); inference 2.6 ms / 1k rows
  (slower than linear/MLP, still ~45 ms per trip).
- **Hyperparameters (initial):** learning_rate 0.05, num_leaves 63, min_child_samples 50,
  subsample 0.8 (every iteration), colsample_bytree 0.8, reg_lambda 1, reg_alpha 0, up to 3000 trees
  with early stopping (100 rounds) on **15% of the training trips held out by trip** inside each
  fit, so the outer CV fold is never used for stopping. Final model keeps 328 trees.
- **Loss:** binary log loss (cross-entropy), gradient boosting with second-order (Newton) steps.
- **Loss curve** (`lgbm_loss_curve.png`, real values from `evals_result_`): both losses fall fast until
  ~100 iterations (val 0.061 at iteration 100); validation then creeps down to its minimum 0.0585 at
  iteration 328 while training keeps falling to 0.048 → the onset of
  overfitting; early stopping picks 328. The validation curve sits slightly *below* training at the
  start because the held-out trips happen to be easier than the training mix (not leakage: they are
  whole unseen trips).
- **Over/underfitting:** train 0.922 vs CV 0.881 (gap 0.041), ST 0.896. Some overfitting to the
  training trips, but held-out performance is still the best of all models, so the extra capacity
  is paying off.
- **Feature importance (gain):** `f_dist_gap`, `f_ctx_gap`, `f_heading_diff`, `f_heading_gap`,
  `f_dist_rank`, `f_speed` (`lgbm_feature_importance.csv`): relative distance, neighbourhood
  agreement and direction, matching the design idea that heading separates two-way twins.

### 3.4 LightGBM ranker (stretch, `"lgbm_ranker"`)

- **Why considered:** the real decision is "which candidate of this point wins", which is a ranking
  problem. LambdaRank with one query group per GPS point optimises NDCG@1 = per-point Top-1 directly.
- **Adaptation:** a ranker outputs scores, not probabilities. We convert them with a softmax over each
  point's candidates, which gives a per-point distribution (exactly what the HMM emission needs).
- **Result:** CV Top-1 0.8816 vs 0.8811 for the classifier (better in 3 of 5 folds), ST 0.8957 vs 0.8964.
  A tie within noise. Optimising Top-1 directly bought nothing measurable: the binary classifier
  already orders each point's candidates almost as well. Highest ST precision (0.929) and lowest ST
  log loss (0.0545), but worse CV log loss (0.082), driven by fold 2 (0.131 vs 0.078 for the
  classifier), the fold that contains the off-network trips.
- **Over/underfitting:** train Top-1 0.937 vs CV 0.882 (gap 0.055, the largest of all 8 models) and
  ST 0.896; train log loss 0.047 vs CV 0.082. It fits the training trips' rankings most tightly but
  generalises no better than the classifier: clear overfitting (`overfitting_check.csv`).

---

## 4. Model selection (all 7 models, `model_comparison.csv`)

| Model | Owner | CV Top-1 | ST Top-1 | CV log loss | Inference ms / 1k |
|---|---|---|---|---|---|
| LightGBM | C | 0.8811 | 0.8964 | 0.0733 | 2.6 |
| Random Forest | C | 0.8777 | 0.8903 | 0.0750 | 1.8 |
| MLP | A | 0.8758 | 0.8930 | 0.0771 | 0.5 |
| KNN | B | 0.8668 | 0.8721 | 0.1373 | 174.8 |
| Logistic Regression | C | 0.8608 | 0.8600 | 0.1604 | 0.3 |
| Gaussian NB | B | 0.8537 | 0.8611 | 0.3990 | 0.5 |

Reasoning points for choosing LightGBM:
1. **Primary metric:** highest CV Top-1, and higher than RF/LogReg in every fold.
2. **Generalises to unseen trips:** also highest ST Top-1 of the untuned models.
3. **Probability quality:** lowest log loss of the classifiers, which matters for HMM emissions.
4. **Efficiency:** 23 s to train, ~45 ms per trip at inference, fine for the live API.
5. **Compatibility with the decoder:** outputs calibrated per-candidate probabilities directly
   (the ranker needs a per-point softmax; KNN is 70× slower at inference).
- Worth noting: MLP is close (0.876 / 0.893) with 5× faster inference and the best argmax continuity
  (0.819); a reasonable fallback if inference time ever mattered.

---

## 5. Bayesian hyperparameter optimisation (`scripts/hpo.py`, `experiments/hpo/`)

- **Model:** LightGBM (selected above).
- **Sampler:** Optuna TPE (`TPESampler(seed=42, multivariate=True, n_startup_trials=10)`): a Bayesian
  method that models good vs bad regions of the search space from past trials.
- **Pruner:** `MedianPruner(n_startup_trials=8, n_warmup_steps=1)`: a trial is stopped after a fold if
  its running mean is below the median of earlier trials at that fold.
- **Objective:** maximise mean GroupKFold(5) Top-1 on confident points of the 80 train trips: same
  folds, same confident-only fitting and same metric as the model comparison. Mean log loss and
  macro F1s were recorded per trial as user attributes. Student-test was not used during the search.
- **Search space** (log scale where the effect is multiplicative):

  | Hyperparameter | Range | Untuned | Best (trial 24) |
  |---|---|---|---|
  | learning_rate | 0.01–0.2, log | 0.05 | 0.0103 |
  | num_leaves | 15–255, log | 63 | 202 |
  | min_child_samples | 10–500, log | 50 | 24 |
  | colsample_bytree | 0.4–1.0 | 0.8 | 0.789 |
  | subsample | 0.5–1.0 | 0.8 | 0.659 |
  | reg_lambda | 1e-3–30, log | 1.0 | 3.50 |
  | reg_alpha | 1e-3–10, log | 0.0 | 6.38 |

  Number of trees is not searched: early stopping on held-out train trips chooses it per fit.
- **Trials:** 50 finished (35 complete, 15 pruned) plus 1 left in RUNNING state: the run was
  interrupted once and resumed from the SQLite study, and the interrupted trial was never closed.
  It has no value and does not affect the result. Total trial time ≈ 135 min on 16 CPU cores.
- **Importance (fANOVA, `hpo_param_importance.png`):** reg_alpha 0.54, learning_rate 0.12,
  num_leaves 0.11, reg_lambda 0.09, colsample 0.08, min_child_samples 0.05, subsample 0.02.
- **History (`hpo_history.png`):** all 35 completed trials land between 0.8800 and 0.8822 (median
  0.8818), against 0.8811 for the defaults: the objective surface is flat around the default configuration.

### Tuned vs untuned (`experiments/hpo/tuned_vs_untuned.csv`)

| | Untuned | Tuned |
|---|---|---|
| CV Top-1 (conf), mean ± std | 0.8811 ± 0.0280 | 0.8822 ± 0.0292 |
| CV candidate macro F1 | 0.9298 | 0.9299 |
| CV observation macro F1 | 0.8368 | 0.8395 |
| CV log loss | 0.0733 | 0.0737 |
| ST Top-1 (conf) | 0.8964 | 0.8997 |
| ST candidate macro F1 | 0.9387 | 0.9394 |
| ST observation macro F1 | 0.8485 | 0.8501 |
| ST log loss | 0.0600 | 0.0595 |
| Inference (ms / 1k rows) | 2.6 | 26.7 |
| Train time (80 trips) | 22.7 s | 92.6 s |
| Trees | 328 | 976 |
| Model file | 1.0 MB | 9.2 MB |

Per fold (Top-1, conf): tuned − untuned = −0.0007, +0.0019, +0.0016, +0.0029, −0.0005 → better in 3 of 5.

**Did tuning help?** Slightly on every Top-1/F1 number, but:
- the CV gain (+0.0011) is far smaller than the fold-to-fold spread and not consistent across folds;
- CV log loss got marginally worse;
- train − CV gap grew from 0.041 to 0.046 (more overfitting);
- inference is ~10× slower and the model 9× larger (low learning rate + many leaves → 976 trees).
The default configuration was already close to the optimum for this feature set, which the flat
optimisation history also shows. Deployment uses the untuned `lgbm` (API default `PATHSNAP_MODEL=lgbm`);
the tuned model is saved as `models/lgbm_tuned.joblib` if the team prefers it.

---

## 6. Things to be ready to explain in the viva

- Why GroupKFold by trip and not random rows (neighbouring 1 s points are near-duplicates).
- Why train on confident rows but score on all, and why confident Top-1 is the headline.
- Why a model can't exceed 1 − miss rate on Top-1 (0.46% of confident points have no correct candidate).
- Why balanced class weights hurt Logistic Regression's log loss but not much its Top-1.
- Why early stopping uses held-out *train trips*, never the CV validation fold.
- What TPE does differently from grid/random search, and what the pruner does.
- Why the ranker needs a softmax before the HMM can use it.
