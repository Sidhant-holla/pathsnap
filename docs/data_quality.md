# Data Quality Log

Record of data issues found during exploration, the evidence for each, and what we decided.
**Partitions examined:** `train` (80 trips) and `student_test` (10 trips). The `evaluator_test` partition was not
opened. Evidence comes from `notebooks/01_eda.ipynb`, whose tables are in `experiments/results/eda/` and
figures in `reports/figures/eda/`.

## Structure and format

| Item | Evidence | Decision / assumption |
|------|----------|-----------------------|
| File layout | `data/raw/<partition>/<trip_id>/<trip_id>.{track,nodes,arcs,route}`. All 4 files exist for all 90 trips. | Loader (`pathsnap.io`) finds trips by partition folder. Trip ids are unique across partitions. |
| Separators and line endings | Tab-separated, no header. `.track` uses `\r\n`, the others `\n`. | Read with `sep=r"\s+"`, which handles both. |
| Column order | `.track` = lon, lat, t. `.nodes` = lon, lat. | Longitude first. Folium maps flip to `[lat, lon]` in `viz.py` only. |
| Implicit ids | `node_id` and `arc_id` are 0-based row numbers. | DataFrame index = id. Never reorder or drop rows of `.nodes` / `.arcs`. |
| Per-trip networks | Every trip has its own network (649 to 307,392 arcs). The same arc id means different roads in different trips. | Always key arcs by `(trip_id, arc_id)`. |

## Missing and invalid values

| Item | Evidence | Decision |
|------|----------|----------|
| NaN / unparseable values | 0 in all files of all 90 trips. | No imputation. |
| Impossible coordinates | 0 points outside ±180/±90 and no (0, 0) points. | None needed. |
| Timestamp order | Strictly increasing in every trip (no duplicates, no reversals). | Sort not required, but the loader output is still checked in `scripts/verify_setup.py`. |
| Fractional timestamps | 9,659 points in trips `00000002`, `00000071`, `00000097`. | Treated as continuous seconds. Nothing assumes integer t. |
| Missing fixes in time | 96.7% of steps are exactly 1 s. 42/90 trips have a gap > 10 s and 23 have a gap > 60 s. Median trip is missing 3.3% of its duration; trip `00000090` has a 42.5 h gap (99.6% of its time span has no fix), trip `00000078` 14.7 h. | For features, the track is split into **segments at gaps > 10 s**, so smoothing and neighbourhood features never cross a gap. The HMM uses its own tuned limit of 175 s: it splits the trip only at longer gaps and bridges gaps of 10–175 s with road-network driving distances. Missing fixes are **not** interpolated, because inventing GPS points would invent evidence. |
| Implausible jumps | 151 consecutive-fix speeds > 252 km/h in 11 trips (95 in `00000013`, 43 in `00000032`). | Kept. Neighbourhood features and HMM transition costs make an isolated jump unlikely to change the decoded road. |

## Road network

| Item | Evidence | Decision |
|------|----------|----------|
| Two-way streets | About 95% of arcs (median per trip) have a reverse twin v→u. For ~64% of GPS points the nearest arc is tied with its twin. | The heading-vs-bearing feature is essential. Candidate ranking gives twins the same rank so neither is cut off. |
| Duplicate arcs | 1,903 arcs repeat an existing u→v pair; 106 zero-length arcs; 24 self-loops. | Kept, since removing them would change arc ids. Geometrically identical duplicates are indistinguishable, which is a small ceiling on achievable accuracy. |
| Connectivity | No isolated nodes. The largest strongly connected component holds 96% of nodes (median), but many small components exist (median 127 per train trip), mostly network clipped at the edge. | HMM transitions between unreachable arcs get a large **finite** penalty so decoding never breaks. |
| Projection | Trips cover 25 UTM zones, latitude −34.6 to 61.7. | Per-trip UTM projection. Verified ≤ 0.04% distance error vs WGS84 geodesic. |

## Reference routes (labels)

| Item | Evidence | Decision |
|------|----------|----------|
| Route is per-arc, not per-point | `.route` lists 59 to 2,266 arcs; tracks have 262 to 19,313 points. | Per-point labels come from a monotone route alignment (`labels.align_route`). |
| Route continuity | 79/90 routes fully connected; 11 trips have 1–3 breaks between consecutive arcs. | The reference is treated as imperfect. Route-level metrics are computed against it as given, and the breaks are noted. |
| Repeated arcs | 10 trips drive some arc more than once. | Alignment uses route *positions*, so repeated arcs are separate states. |
| Off-route GPS | 11.2% of training points are > 50 m from their aligned arc. Most come from **off-network episodes**: in `00000054` the vehicle spends 4.3 h (15,584 points) inside a ~1 km² area with no mapped road, ending within 6 m of where it started. `00000033` is similar. | Points > 50 m are flagged `low_conf`. Training uses confident points. Metrics are reported on confident points (main) and all points (secondary). These points have no correct candidate by construction. |

## Assumptions that shape the approach
1. The reference route is the best available label but not error-free ground truth.
2. The GPS error is mostly < 35 m (99% of confident training points), which justifies a 50 m candidate radius.
3. A vehicle moves along connected arcs in their direction of travel, which is the basis for the HMM transitions.
4. Segments separated by > 10 s are decoded as independent pieces.
