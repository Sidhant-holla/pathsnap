# Report notes, Member D: HMM/Viterbi (report §5), maps (§6), API/UI/deployment (§7)

These are **notes, not report text**. Write the sections in your own words (brief §8: report
content is checked for AI-generated text, and each of us must be able to explain it in the viva).
Every number below comes from a file in the repo; the file is named next to it so the report
stays verifiable.

Partitions: decoder parameters chosen on **train** only; every comparison below is on
**student_test** (10 trips, 19,160 GPS points). Evaluator trips: never scored or looked at.

---

## §5 HMM and Viterbi route reconstruction

Code: `src/pathsnap/graph.py`, `src/pathsnap/hmm.py`, `src/pathsnap/route_metrics.py`.
Tests: `tests/test_hmm.py` (12 tests on a synthetic two-way network with a parallel distractor road).

### Q1. Hidden states
- State at GPS point t = one of that point's **candidate arcs** (A's `candidates.py`: arcs within
  r = 50 m, at most 16 per point). Each state keeps its `arc_id` and `proj_frac` (where on the arc
  the point projects), so the candidate-to-road-segment correspondence is preserved end to end.
- Student-test: 8.9 candidates per point on average, candidate miss rate 0.03% (confident points),
  0.17% (all) → `experiments/results/candidates/feature_build_summary.csv`.
- Arcs are **directed**: a two-way street is two states at the same distance. Only heading (in the
  classifier) and the transitions (driving direction) can separate them.

### Q2. Emissions
- Classifier: `log e_i = λ · log(p_i / Σ_j p_j)`. p = LightGBM P(candidate is correct). Normalising
  per point turns independent binary probabilities into a distribution over the point's candidates.
  λ weighs emissions against transitions.
- Distance-only baseline: Gaussian GPS error, `log e_i = −d_i² / 2σ²` (no classifier at all).
- Probabilities are floored at 1e-9 before the log (no −∞).

### Q3. Transitions
- `log a = −|d_network − d_straight| / β`.
  - d_network: **driving** distance from the projected position on arc A to the projected position
    on arc B, following arc directions: rest of A + shortest path (end node of A → start node of B)
    + offset into B. Dijkstra (networkx) bounded by a cutoff, node-to-node results cached per trip
    (consecutive points share most candidates).
  - d_straight: distance between the two GPS fixes (UTM metres).
  - Idea: if the car really went A → B, it drove about as far as the GPS moved.
- Cutoff per step = min(v_max·Δt + slack, d_straight + β·break_penalty): anything longer is either
  impossible or would get the break penalty anyway, so Dijkstra never searches further than needed.
- **Design detail worth explaining:** small *backward* moves on the same arc (GPS jitter of a stopped
  car) return a **negative** distance (signed step) instead of "drive round the block". A stopped car
  costs ~0, while a car moving 10 m forward on the wrong-direction twin costs |−10 − 10| = 20 m.
  (First version used +|step| and that made the wrong twin look perfect for a moving car. We caught
  it before testing; `test_backward_jitter_is_signed` covers it.)
- Decode in **log space** (sums, not products), so a 19k-point trip never underflows.

### Q4. Missing candidates and large GPS gaps
- No candidate within 50 m → the point's only state is its **nearest arc at any distance**
  (STRtree nearest query), flagged `fallback`. So the decoder always returns exactly one arc per point.
- Δt > G seconds → the track is split into **segments** decoded independently (no transition across
  a gap). Tuned G = 175 s.
- Stress test on train trips (`predict_trip`, not tuned on):
  trip 90 (42 h gap) → 2 segments, 0.35 s; trip 54 (4.3 h off-network) → 16,519 fallback points,
  1,908 soft breaks, still decoded in 5.4 s; trip 13 (largest network, 307k arcs) → 3.3 s.

### Q5. Disconnected road segments
- Unreachable within the cutoff, or implied speed > v_max (50 m/s) + 30 m slack →
  `log a = −break_penalty` (50), a large but **finite** value. Ordinary transitions are capped at the
  same value, so an impossible move is never better than a possible one.
- Why finite: with −∞ one bad fix (or a missing road in the map) makes every path −∞ and the
  decode collapses. With a soft break the path continues and the break is counted (`soft_break` flag).
- For the displayed route, consecutive decoded arcs that are not adjacent are joined with the
  shortest path (`route_metrics.fill_route`). If none exists within 2 km, the route is split into
  separate pieces instead of drawing a fake line.

### Decoder parameter tuning (train only)
- Optuna TPE (seed 42) on 24 random train trips, objective = pooled confident-point top-1 of the
  decoded path. File: `scripts/hmm_ablation.py`, `models/hmm_params.json`.
- **Out-of-fold probabilities**: the saved LightGBM was fitted on all 80 train trips, so its
  probabilities on them are over-confident. We used 5-fold GroupKFold OOF probabilities
  (pooled OOF top-1 0.8849, close to C's CV 0.8811) so λ is tuned on "unseen-trip" probabilities.
- Run 1: 25 trials (distance), 40 trials (classifier), 867 s. Best G (175 of max 180) and β (48.8 of
  max 50) sat at the range edges → **run 2** widened ranges (β ≤ 200, G ≤ 600 s, λ ≥ 0.05), 1,227 s.
  Run 2's optimum was inside the ranges and no better on train (0.9173 vs 0.9174) → objective is flat
  there. Kept the run with the best **train** score (run 1). Both runs:
  `experiments/results/hmm/hmm_params_run{1,2}.json`, `tune_*_trials_run{1,2}.csv`, `experiments/logs/hmm_tune_run{1,2}.log`.
- Final: classifier-HMM λ = 0.276, β = 13.2 m, G = 175 s; distance-HMM σ = 14.8 m, β = 48.8 m, G = 60 s.
  Fixed (not tuned): v_max = 50 m/s, speed slack 30 m, break penalty 50, backward tolerance 15 m.

### Q6. Did the HMM improve the route? (`experiments/results/hmm/ablation_student_test.csv`)

| Method | Top-1 (confident) | Top-1 (all) | Continuity | Route P | Route R | Route F1 | Route F1 (length-w.) | Decode ms/point |
|---|---|---|---|---|---|---|---|---|
| Nearest road | 54.03% | 53.96% | 28.88% | 0.430 | 0.992 | 0.591 | 0.657 | 0.006 |
| Distance-only HMM | 90.58% | 90.45% | 90.20% | 0.923 | 0.988 | 0.953 | 0.981 | 0.35 |
| Classifier only (LightGBM) | 89.64% | 89.51% | 80.85% | 0.736 | 0.993 | 0.839 | 0.893 | 0.006 |
| **Classifier + HMM** | **92.79%** | **92.66%** | **90.42%** | **0.958** | **0.992** | **0.974** | **0.987** | 0.30 |

- Classifier-only numbers match C's `model_comparison.csv` exactly (89.64% / 80.85%): same metric code.
- HMM improves top-1 on **all 10** trips (+0.8 to +6.3 points; biggest: trip 77 75.4% → 81.7%,
  trip 76 92.6% → 96.7%) and route F1 on all 10 (e.g. trip 65 0.709 → 0.986, trip 76 0.727 → 0.965)
  → `ablation_per_trip.csv`.
- Where the gain comes from: route **precision** 0.736 → 0.958. Classifier-only routes zig-zag
  between neighbouring/twin arcs, so the route contains many arcs never driven; recall is ~0.99 for
  every method.
- Nearest road: 54% because 71% of points have a two-way tie at the nearest distance (A's finding).
  Its precision 0.43 shows it hops between both directions.
- Distance-only HMM (no ML) already beats the classifier on continuity and route F1. The road network
  alone carries a lot of information. The classifier adds +2.2 points top-1 on top of it.
- Continuity is 90.4%, not 100%: between two fixes the car can pass a short arc without any point
  landing on it, so consecutive decoded arcs are reachable but not adjacent. `fill_route` inserts
  those arcs for the displayed route.
- Metric definitions (write them out in the report): `src/pathsnap/route_metrics.py` docstring.
  Route P/R/F1 = predicted arc set vs `.route` arc set (gap-filled route; raw also reported);
  length-weighted versions weight arcs by metres.

### Cases where the HMM did not help (be honest here; brief asks for it)
1. **Emission weight too low (trip 77, points 142–201, map `hmm_nohelp_00000077.html`).**
   The decoder moves to the next arc (8339) about 20 points early. Those points are 3.5 m from the
   true arc 8343 and 26.7 m from 8339, and the classifier is confident (p 0.68 vs 0.02). With the
   tuned λ = 0.28 transition smoothness outweighs the classifier: window 40/60 correct.
   Same params with λ = 1: 60/60, and the whole trip goes 81.7% → 85.4%.
   This matches the overall picture: **untuned defaults (λ = 1, β = 5, G = 30) score 93.58% top-1 on
   student-test vs 92.79% tuned** (route F1 0.9749 vs 0.9743; `ablation_student_test_defaults.csv`
   vs `ablation_student_test.csv`, from `python -m scripts.hmm_ablation ablation --defaults`).
   Likely reason (hypothesis, not proven): the train subset contains off-network trips (e.g. 54,
   33) where the classifier is unreliable, which pushes the train optimum to a low λ.
   **Team decision pending:** keep the train-tuned values (strict: student-test stays a clean test)
   or pick defaults using student-test (allowed by our plan, but then say so in the report).
2. **Stationary car at the end of the trip (trip 77, points 3785–3844).** The car is parked: GPS wanders
   ~200 m but net movement is 3–12 m. The HMM picks the southbound twin (jittery heading ~190–227°)
   while the label stays on the northbound last route arc, because the route-constrained labels can only move
   forward. Same physical road, 0 m spatial error. It's a label limitation (PLAN §4, U-turn/stationary
   note), not a decoder error. Counted as wrong by top-1, though.
3. **Gap filling helps the HMM route but hurts the classifier-only route:** classifier-only route F1
   is 0.839 filled vs 0.901 raw (filling between zig-zag arcs adds wrong arcs). Classifier + HMM is
   0.974 filled vs 0.930 raw. Gap filling only makes sense on a drivable decode.

---

## §6 Folium maps (D's part)

- `reports/maps/hmm_fix_1_00000076.html`: trip 76 points 1394–1453, classifier 30/60 → HMM 60/60
  (classifier jumps to the parallel carriageway at an interchange).
- `reports/maps/hmm_fix_2_00000056.html`: trip 56 points 169–228, 28/60 → 42/60.
- `reports/maps/hmm_fix_3_00000000.html`: trip 0 points 1829–1888, 48/60 → 60/60.
- `reports/maps/hmm_nohelp_00000077.html`: case 1 above, 53/60 → 40/60.
- Layers: GPS points (blue), nearby road network (grey), reference route (green; report maps of
  student_test only), classifier-only arcs (orange), classifier + HMM route (red dashed); layer
  control + legend + title. Generated by `scripts/hmm_maps.py` (window choice rules in its docstring:
  car must move ≥150 m net, twin-direction flips ignored because they are invisible on a map).
- Screenshot them from a normal browser (the app's preview pane opens them at world zoom).

---

## §7 API, UI and deployment

Architecture (brief §3): Laptop 1 = FastAPI (`api/main.py`) with the model, HMM, road graphs and
server-side scorer. Laptop 2 = Streamlit UI (`ui/app.py`). Guide: `docs/DEPLOY.md`.

- **Endpoints:** `GET /trips` (the served partition's trip IDs; evaluator by default via
  `PATHSNAP_PARTITION`), `POST /predict {"trip_id"}`, `GET /health`, `GET /` and auto docs at `/docs`.
- **Pipeline per request** (`src/pathsnap/pipeline.py::predict_trip`): load trip (no route) → UTM →
  candidates + features (same function that built the training table) → LightGBM → Viterbi →
  collapse + gap-fill → lat/lon polylines.
- **Leakage design (good viva point):** `predict_trip` never loads `.route` (`load_trip` default
  `with_route=False`). `api/scorer.py` is the only API module that reads it, and only *after*
  prediction. The response has GPS points, the predicted route, flags, timings and metrics, never
  reference arcs (only their count). The UI has no data or model at all.
- **Crash-only checks on evaluator trips:** `{"trip_id": ..., "score": false}` skips scoring
  entirely; the route isn't even read, so no evaluator metric is computed or logged.
  `scripts/smoke_api.py` uses it (added by Sidhant).
- **Logging** (brief: show logs on the receiving laptop): middleware prints one line per request:
  `time | POST /predict | trip=00000000 | status=200 | 672 ms | client=<ip>`, plus a stage-timing
  line (load / features / classify / graph / decode / route / score).
- **Timings** (student-test, one laptop): 0.3–1.8 s per trip end to end; decode ~0.3 ms per point;
  LightGBM scores the whole student-test table (170,774 candidates) in 0.70 s.
- **Errors:** unknown trip → 404, bad body → 422, pipeline exception → 500 with the message (and the
  traceback in the API console). The UI shows each as a readable message; API unreachable → hint
  about hotspot / IP / firewall.
- **Network:** both on one hotspot; inbound TCP 8000 allowed on Laptop 1
  (`New-NetFirewallRule ... -LocalPort 8000 -Profile Private,Public`, hotspots are usually "Public");
  uvicorn bound to 0.0.0.0.

**Screenshots still to take (two-laptop test):**
1. UI on Laptop 2: trip list from the API, metrics table and map for one trip.
2. Laptop 1 console: startup line + `POST /predict` lines with trip id, status and timing (the brief
   asks for exactly this, "on the CMD of the receiving laptop").
3. `http://<laptop-1-ip>:8000/health` opened from Laptop 2.
4. Optional: `/docs` page, `smoke_api` output (crash-only, no metrics).

## Reproduce
```
python -m scripts.build_features
python -m scripts.hmm_ablation oof        # OOF probabilities on train
python -m scripts.hmm_ablation tune       # decoder parameters (train only)
python -m scripts.hmm_ablation ablation   # 4-way table on student_test
python -m scripts.hmm_maps                # report maps
python -m pytest tests/test_hmm.py
```
