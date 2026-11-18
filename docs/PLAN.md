# PathSnap: Team Plan

Read this whole file before writing any code. Every one of us must be able to
explain **every** section in the viva, not just our own part.

---

## 1. Our approach in one paragraph

For each GPS point we collect the nearby road segments as **candidates**. We describe
each (point, candidate) pair with geometric, directional and *neighbourhood* features.
A classifier then scores the probability that each candidate is the true road. Next,
an **HMM decoded with Viterbi** picks the most likely *drivable* sequence of roads,
using those probabilities as emissions and road-network distances as transitions.
We also use the **same Viterbi idea to build our training labels**, by aligning GPS
points to the known reference route. So one algorithm, explained once, does two jobs.

```
.track ─► project to metres ─► candidates (radius r) ─► features ─► classifier P(correct)
                                                                         │
.nodes/.arcs ─► directed road graph ─► network distances ─► HMM transitions
                                                                         ▼
                                                  Viterbi ─► one arc per GPS point ─► route
.route ─► route-constrained Viterbi ─► per-point labels (training + scoring only)
```

## 2. What makes our approach better than the default (and still easy to explain)

| # | Idea | Why it helps | How to explain it in one line |
|---|------|--------------|-------------------------------|
| 1 | **Labels from route-constrained Viterbi** | `.route` lists arcs but doesn't say which point belongs to which arc. Nearest-arc labelling breaks at junctions and U-turns. | "We run Viterbi where the only allowed states are the route arcs in order, so each point gets the route arc that best explains it while moving forward." |
| 2 | **Label confidence flag** | Points far from their labelled arc reveal limits of the reference route and GPS (the brief asks for this). | "A point more than X m from its aligned arc is marked low-confidence, and we report how many there are." |
| 3 | **Data-driven candidate radius** | Too small a radius means a high miss rate; too large means slow and noisy. | "We picked r as the 99th percentile of point-to-true-arc distance on the **training** trips." |
| 4 | **Direction-aware features** | Arcs are *directed* (from→to). A two-way street is two arcs at exactly the same distance, so only heading can tell them apart. | "The angle between the car's smoothed heading and the arc's bearing separates the two directions of the same road." |
| 5 | **Neighbourhood features** | One noisy point may be closer to the wrong road, but its neighbours aren't. | "We measure how well this arc also explains the points 2–3 seconds before and after." |
| 6 | **Per-point normalised emissions** | Classifier probabilities aren't a distribution over one point's candidates. The HMM needs one. | "Divide each candidate's probability by the sum over that point's candidates." |
| 7 | **Soft breaks + gap segmentation** | Disconnected transitions or long GPS gaps would otherwise give –∞ and kill the whole path. | "Impossible moves get a large penalty instead of –∞, and gaps over G seconds split the trip into segments that are decoded separately." |
| 8 | **4-way ablation** | Shows exactly where accuracy comes from. | "Nearest road → distance-only HMM → classifier only → classifier + HMM." |
| 9 | **Leakage-proof code structure** | Rules say ground truth must never reach the model or the UI. | "The prediction pipeline *can't* read `.route`, because only the evaluation module and the API scorer load it." |

## 3. Key facts and rules (from the official brief)

- **Files (no headers, whitespace-separated):** `.track` = `lon lat t` (≈1 Hz);
  `.nodes` = `lon lat` (row index = node_id); `.arcs` = `from_node to_node` (row index = arc_id);
  `.route` = ordered arc_ids actually driven.
- **Longitude comes first.** Folium wants `[lat, lon]`.
- Road-network files come **per trip**, so `arc_id` is only unique *within a trip*. Always key by `(trip_id, arc_id)`.
- Never compute distances in degrees. Project each trip to UTM metres (`pyproj`, `estimate_utm_crs`).
- **Partitions:** 80 train / 10 student-test / 10 evaluator-test.
  - CV must be **GroupKFold by trip**. Fit scalers etc. *inside* the CV pipeline, on training folds only.
  - Student-test: model comparison only.
  - **Evaluator-test: we don't train on it, tune on it, look at its metrics or plot its routes. Ever.**
    Its only use is the live API demo.
- No hardcoded predictions, and no reading ground truth inside prediction code.
- Training/validation loss curves only for models that really train in epochs (MLP, LightGBM). Never fake one.
- At least 5 models from different families, Bayesian HPO with saved trial records, and the HMM compared against classifier-only.

## 4. Shared code contracts (follow these exactly, so everything plugs together)

Everything lives in `src/pathsnap/`. Run scripts from the repo root with `python -m scripts.<name>`.

| Module | Owner | Main function(s) |
|--------|-------|------------------|
| `io.py` | B | `list_trips(partition) -> list[str]`; `load_trip(trip_id, with_route=False) -> Trip` (dataclass: `track[lon,lat,t]`, `nodes[lon,lat]`, `arcs[u,v]`, `route: list[int] or None`) |
| `geo.py` | A | `project(trip) -> Trip` adds `x,y` (metres) to track and nodes; `point_to_segment(...)` |
| `graph.py` | D | `build_graph(trip) -> nx.DiGraph`; `network_distance(arc_a, pos_a, arc_b, pos_b, cutoff)` with caching |
| `labels.py` | A | `align_route(trip) -> DataFrame[obs_idx, label_arc, label_dist_m, low_conf]` (route-constrained Viterbi) |
| `candidates.py` | A | `generate_candidates(trip, radius_m, max_k) -> DataFrame[obs_idx, arc_id, dist_m, proj_frac, ...]` |
| `features.py` | A | `build_features(trip, cands) -> DataFrame`; constant `FEATURE_COLUMNS` (all start with `f_`) |
| `evaluate.py` | C | `group_cv(make_model, df, n_splits=5, seed=42)`; `score(df) -> dict`; `log_result(name, config, metrics)` |
| `hmm.py` | D | `viterbi(cands_with_proba, graph, params) -> DataFrame[obs_idx, arc_id]` |
| `pipeline.py` | D+A | `predict_trip(trip_id, model, params) -> dict`. **Never loads the route.** |
| `viz.py` | B | `trip_map(trip, layers={...}) -> folium.Map` (GPS, network, reference*, predicted) |

\*The reference-route layer is only used for report maps of train/student-test trips. The UI never shows it.

**The feature table** is the one dataset every model trains on, so preprocessing stays consistent:
`data/processed/{train,student_test}_candidates.parquet`. It has one row per (trip_id, obs_idx, arc_id) with
columns `trip_id, obs_idx, t, arc_id, label (0/1), label_low_conf, f_*`.
Points whose true arc isn't among their candidates keep all-zero labels and count toward the **candidate miss rate**.

**Every experiment** writes `experiments/results/<model_name>.json` (config, seed, all metrics, train/infer time)
and appends one row to `experiments/results/summary.csv`. Models are saved to `models/<name>.joblib`.

## 5. Features (v1, built by A; anyone can propose more)

| Group | Features |
|-------|----------|
| Geometric | perpendicular distance; distance rank among the point's candidates; distance − min distance; #candidates; position along arc (0–1); arc length |
| Directional | \|heading − arc bearing\| using heading smoothed over ±2 s; speed (m/s) |
| Neighbourhood | mean distance from points t−3…t+3 to this arc; whether the arc (or a directly connected arc) is nearest for t±1 |
| Topology | distance from projection to the nearest arc endpoint (junction proximity); out-degree of the arc's end node |
| GPS quality | time gap to previous and next point; local jitter (distance of the point from its smoothed track) |

Feature rule: **no feature may use `.route` or labels.** Anything computed from the route is leakage.

## 6. Metrics (computed by `evaluate.score`, identical for every model)

- **Observation Top-1 accuracy**: for each point, is the argmax candidate the labelled arc? *This is our primary
  selection metric*, because the route uses the argmax and not a 0.5 threshold.
- **Macro F1**, reported two ways and both defined in the report:
  (a) candidate-level binary (correct vs incorrect candidate, threshold 0.5);
  (b) observation-level, with predicted arc vs labelled arc macro-averaged over arcs.
- **Log loss** (candidate-level binary). It matters because the HMM uses the probabilities, so calibration matters.
- **Candidate miss rate**, **confusion matrix** (binary), precision, recall, **train and inference time**.
- After decoding: **route continuity** (% of consecutive distinct predicted arcs that are actually connected,
  i.e. `arc_a.v == arc_b.u`), plus **route precision/recall/F1** of the predicted arc sequence vs `.route`.

## 7. Models (each from a different family)

| Model | Family | Owner | Notes |
|-------|--------|-------|-------|
| Logistic Regression | Linear | C | Baseline; StandardScaler in Pipeline; `class_weight="balanced"` |
| Gaussian Naive Bayes | Probabilistic | B | Quick; shows what independence assumptions cost us |
| KNN (distance-weighted) | Instance-based | B | Scaling is essential; slow at inference, so note that |
| Random Forest | Ensemble (bagging) | C | Robust baseline, feature importances |
| LightGBM | Boosting | C | Likely winner; has epoch-wise train/val loss curves |
| MLP | Neural network | A | Scaled inputs, early stopping; has loss curves |
| *(stretch)* LightGBM Ranker | Learning-to-rank | C | Treats each point's candidates as a group, which optimises top-1 directly |

**Bayesian HPO (C):** Optuna TPE on the chosen model with GroupKFold(5) on the 80 train trips.
The objective is mean CV top-1, with log loss logged too. Run about 50 trials.
Save the study (`experiments/hpo/optuna.db` + trials CSV) and the history and param-importance plots.
Then compare tuned vs untuned on CV macro F1, student-test top-1, student-test macro F1, log loss and inference time.

## 8. HMM / Viterbi (D)

- **Hidden state** at time t = one of point t's candidate arcs (we keep the `arc_id` for each candidate).
- **Emission:** `log e = λ · log(p_i / Σ_j p_j)`, where p comes from the classifier.
  The distance-only baseline uses a Gaussian on the distance instead: `e ∝ exp(−d²/2σ²)`.
- **Transition:** `log a = −|d_network − d_straight| / β`. Here d_network is the driving distance from the
  projected position on arc A to the projected position on arc B, following arc directions (bounded Dijkstra,
  cached), and d_straight is the distance between the two GPS points.
  - If the implied speed is above v_max, or the arcs aren't connected, the transition gets a large finite
    penalty (soft break, not −∞).
- **Gaps:** if Δt > G seconds, split the trip and decode each segment independently. A point with no candidates
  falls back to its nearest arc and is flagged.
- **Decoder parameters (λ, β, σ, G):** tuned on train + student-test only.
- **Output:** one arc per point. Consecutive duplicates are collapsed into the route, and gaps between
  non-adjacent arcs are filled with the shortest path for display.
- **Report:** the 4-way ablation (nearest road / distance-HMM / classifier-only / classifier+HMM) on student-test,
  plus 2–3 maps where the HMM fixed something and one where it didn't help.

## 9. API and UI (D)

- **API** (`api/main.py`, FastAPI, Laptop 1, `uvicorn api.main:app --host 0.0.0.0 --port 8000`):
  - `GET /trips` returns the 10 evaluator trip IDs.
  - `POST /predict {trip_id}` runs `predict_trip`, then a **separate** server-side scorer loads the route and
    computes the metrics. It returns GPS points, the predicted route geometry and the metrics. It never returns
    reference arcs.
  - Logging middleware prints every request with timing (the evaluators want to see these logs).
- **UI** (`ui/app.py`, Streamlit + streamlit-folium, Laptop 2): pick a trip → call the API (`API_URL` env var)
  → show a map of GPS points and the predicted route, plus a metrics table.
- **Network:** both laptops on the same hotspot. Allow port 8000 through the Windows firewall on Laptop 1.
  Test this early.
- **Before the demo:** a smoke test on the evaluator trips may only check that nothing crashes.
  We don't read or tune on those numbers.

## 10. Work split

The data pipeline is the bottleneck, because nobody can train until the feature table exists. So the
split isn't one model each: early on, everyone builds something that doesn't depend on the features.

| | **A: Data & features** | **B: EDA, viz, report** | **C: Models & HPO** | **D: HMM & deployment** |
|---|---|---|---|---|
| **Phase 0** (first 30 min) | `geo.py`, agree on the `Trip` format with B | `io.py` loader (everyone depends on it, so push it fast); confirm the folder layout | `evaluate.py` harness tested on a fake table with the agreed schema | `graph.py` + `network_distance` |
| **Phase 1** (→ first evaluation) | `labels.py` (route-constrained Viterbi), `candidates.py`, radius analysis, `features.py`, build the parquet tables | **EDA notebook** (all §4 checklist items), Folium maps, data-quality log. *This is what gets evaluated first.* | Plug in LogReg the moment the table lands; set up the results CSV and plots | `hmm.py` with distance-only emissions (no classifier needed yet); nearest-road baseline |
| **Phase 2** | MLP (+ loss curves); help D with `pipeline.py` | GNB + KNN; model comparison plots | RF + LightGBM; pick the model with evidence | Swap in classifier emissions; tune λ, β, G; start API |
| **Phase 3** | Feature ablations if time allows; review everyone's numbers | Ablation maps (classifier vs HMM); report skeleton filling up | Bayesian HPO, tuned vs untuned | API + UI, two-laptop test |
| **Phase 4** | Report: data, labels, features | **Report owner**: assembles the PDF | Report: models, HPO | Report: HMM, API/UI screenshots |

**Sync points (5 min each):** after Phase 0 (contracts work?), when the feature table lands, after model
selection, and before the demo. At each one, every owner explains their part to the other three. That's our viva prep.

## 11. Git workflow

- Each person works on a branch (`a/features`, `b/eda`, `c/models`, `d/hmm-api`), with small commits and frequent merges to `main`.
- Shared contract files (`io.py`, `evaluate.py`, `FEATURE_COLUMNS`): change them only after telling the team.
- Never commit `data/raw/` or anything secret. Do commit result JSON/CSVs, plots, maps and the final model.
- Fixed seed `42` everywhere.
- You must understand every line you push.
