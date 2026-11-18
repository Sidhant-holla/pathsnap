# Technical report checklist

The report is built by `python -m scripts.build_report` into `reports/PathSnap_Technical_Report.pdf`.
Ticked items are covered there. The open items need the physical two-laptop demo or the team.

Each line below is something the brief (Deliverables PDF, section 2) requires, with the file that
holds the evidence and whose notes cover it. Write the report text in your own words from the
notes; every number must be traceable to a file in the repo.

Notes: A = `docs/report_notes_A.md`, B = `docs/report_notes_B.md`, C = `docs/report_notes_C.md`,
D = `docs/report_notes_D.md`. Data quality: `docs/data_quality.md`. Final decisions: `docs/PLAN.md` §12.

## 2.2 Dataset understanding and EDA
- [x] Q1 How did you preprocess the data? (A §2.2 Q1)
- [x] Q2 Patterns observed; features extracted and why. (A §2.2 Q2, B §2.2)
- [x] Missing-value analysis: `reports/figures/eda/02_missing_seconds_per_trip.png`, `03_sampling_gaps.png`, `experiments/results/eda/missing_invalid_summary.csv`
- [x] GPS trajectory plots: `reports/figures/eda/05_trajectory_samples.png`
- [x] Road-network visualisation: `reports/figures/eda/07_network_structure.png`
- [x] At least one Folium map: `reports/figures/maps/eda_trip_00000001.png`
- [x] State the partition used for each analysis (train + student-test; evaluator never opened)
- [x] Supporting evidence for the features: `reports/figures/feature_ablation.png`, `experiments/results/lgbm_feature_importance.csv`

## 3.1 Overall model comparison table
- [x] Table with Model | Family | Key configuration | CV Macro F1 | Top-1 | Route continuity | ST Macro F1 | Log loss | Inference time
  → `experiments/results/model_comparison.csv` (8 variants). State which macro F1 is used (candidate level; observation level as a secondary column) and what "Top-1" means.

## 3.2 Per model (A1, A2, B5, B6, C7, C8, C9 for every model)
- [x] Logistic Regression (C)
- [x] Random Forest (C)
- [x] LightGBM, untuned (C) and tuned (C §5)
- [x] LightGBM ranker (C)
- [x] MLP (A)
- [x] Gaussian Naive Bayes (B)
- [x] KNN (B)
- [x] Over/underfitting evidence for each: `reports/figures/comparison/overfitting_check.png`, `experiments/results/overfitting_check.csv`
- [x] No textbook-only answers: tie each answer to our numbers

## 3.3 Visualisations
- [x] CV macro F1 across models: `reports/figures/comparison/cv_macro_f1.png`
- [x] Student-test top-1 and macro F1: `reports/figures/comparison/student_test_top1_f1.png`
- [x] Log loss: `reports/figures/comparison/log_loss.png`
- [x] Confusion matrices (optional): `reports/figures/comparison/confusion_matrices.png`
- [x] Training/validation loss curves ONLY for iterative models: `reports/figures/{mlp,lgbm,lgbm_tuned,lgbm_ranker}_loss_curve.png`
- [x] Plots justifying model selection: `reports/figures/comparison/fold_stability.png`, `accuracy_vs_speed.png`

## 4. Bayesian hyperparameter optimisation (C §5)
- [x] Hyperparameters searched, ranges/distributions, objective, number of trials (51), best values: `experiments/hpo/best_params.json`, `trials.csv`
- [x] Tuned vs untuned (CV macro F1, ST top-1, ST macro F1, log loss, inference time): `experiments/hpo/tuned_vs_untuned.csv`
- [x] Figures: `reports/figures/hpo_history.png`, `hpo_param_importance.png`
- [x] Conclusion: tuning gave a marginal gain at 10x inference cost; the untuned model is deployed (PLAN §12)

## 5. HMM and Viterbi (D §5)
- [x] Hidden states, emission and transition probabilities, gaps and missing candidates, disconnected segments
- [x] Classifier-only vs classifier + HMM: `experiments/results/hmm/ablation_student_test.csv` (and the 4-way table with nearest road and distance-only HMM)
- [x] Where the HMM helped / didn't: `reports/figures/maps/hmm_fix_*.png`, `hmm_nohelp_00000077.png`
- [x] Decoder tuning on train only, and the tuned vs default decision (PLAN §12)

## 6. Folium maps (B §6)
- [x] At least one representative map in the PDF showing GPS, road network, reference route and predicted route, with legend and layer control: `reports/figures/maps/st_route_*.png`
- [x] Map-generation code is in the repo: `src/pathsnap/viz.py`, `scripts/report_maps.py`, `scripts/hmm_maps.py`

## 7. API, UI and deployment (D §7, docs/DEPLOY.md)
- [x] Screenshots of the working UI: `reports/figures/ui/` (student-test). Optional: add demo-day ones to `reports/figures/deployment/`
- [ ] Screenshots of successful API responses **on the CMD of the receiving laptop** (Laptop 1 console log lines): save to `reports/figures/deployment/`, then rerun `python -m scripts.build_report`
- [x] One sentence on the leakage design: the prediction never reads the route; only the server-side scorer does

## Also required
- [ ] Include the disclosures required by the event's policy (Deliverables §8, rule 5)
- [x] One consolidated PDF; every number verifiable in the repo
- [ ] Repository made public at submission
