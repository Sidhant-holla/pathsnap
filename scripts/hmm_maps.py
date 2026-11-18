"""Report maps: where HMM decoding fixed the classifier, and where it did not help (PLAN section 8).

Run after scripts.hmm_ablation (uses the tuned models/hmm_params.json):
    python -m scripts.hmm_maps

On every student_test trip we decode with the classifier alone and with classifier + HMM,
mark each GPS point correct/incorrect against its label, and slide a 60-point window
along the trip. Only windows where the car actually moves are considered: the straight-line
distance between the window's first and last GPS point must be >= MIN_MOVE_M (summed step
lengths are not used, because GPS jitter on a parked car adds up to hundreds of metres).
A fix or a loss only counts when the wrong arc is a DIFFERENT road, not the
opposite-direction twin of the right one (twins overlap on a map, so the difference would
be invisible, and on a near-stationary car the label's direction is itself uncertain). The windows with the largest gain become "fix" maps, and the
window with the largest loss (classifier right, HMM wrong) the "did not help" map.

Each map shows GPS points, nearby road network, the reference route (green; allowed here
because these are student_test report maps, never UI maps), the classifier-only arcs
(orange) and the classifier + HMM route (red, dashed).

Outputs: reports/maps/hmm_fix_<k>_<trip>.html, reports/maps/hmm_nohelp_<trip>.html and
experiments/results/hmm/map_windows.csv.
"""
import json
from dataclasses import replace

import folium
import numpy as np
import pandas as pd

from pathsnap.config import MAPS_DIR, STUDENT_TEST
from pathsnap.evaluate import KEY, load_model, load_points, load_table, predict_proba
from pathsnap.hmm import HMMParams
from pathsnap.viz import _arc_lines, _dedupe_consecutive, trip_map
from scripts.hmm_ablation import OUT, PARAMS_PATH, TripCache, decode

WINDOW = 60
MIN_MOVE_M = 150.0
N_FIX = 3
CLF_COLOR = "#f28e2b"


def windows(diff: np.ndarray, w: int) -> np.ndarray:
    """Sum of `diff` over every window of w points (index = window start)."""
    c = np.r_[0, np.cumsum(diff)]
    return c[w:] - c[:-w]


def window_map(item: dict, start: int, clf_arcs, hmm_arcs, title: str) -> folium.Map:
    trip = item["trip"]
    sl = slice(start, start + WINDOW)
    sub = replace(trip, track=trip.track.iloc[sl].reset_index(drop=True))
    # reference arcs of this stretch = labels of the points in the window, in order
    m = trip_map(sub, reference_route=item["label"][sl], predicted_route=hmm_arcs[sl],
                 network_buffer_m=120, title=title)
    # trip_map already added its LayerControl; a layer added after it would be referenced by
    # the control before it exists in the page script, so remove the control and re-add it last.
    for key in [k for k, c in m._children.items() if isinstance(c, folium.LayerControl)]:
        del m._children[key]
    fg = folium.FeatureGroup(name="Classifier only (no HMM)")
    folium.PolyLine(_arc_lines(trip, _dedupe_consecutive(clf_arcs[sl])), color=CLF_COLOR, weight=4,
                    opacity=0.85).add_to(fg)
    fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:24px;left:200px;z-index:9999;background:white;padding:6px 10px;'
        f'border:1px solid #888;border-radius:4px;font:13px sans-serif"><span style="display:inline-block;'
        f'width:14px;height:4px;background:{CLF_COLOR};margin-right:6px;vertical-align:middle"></span>'
        'Classifier only (no HMM)<br>Predicted route = classifier + HMM</div>'))
    return m


def main():
    params = HMMParams.from_dict(json.loads(PARAMS_PATH.read_text())["classifier_hmm"]) if PARAMS_PATH.exists() else HMMParams()
    bundle = load_model(json.loads(PARAMS_PATH.read_text()).get("model", "lgbm") if PARAMS_PATH.exists() else "lgbm")
    df = load_table(STUDENT_TEST)
    df["proba"] = predict_proba(bundle["model"], df[bundle["feature_columns"]], df[KEY])
    cache = TripCache(STUDENT_TEST, df, load_points(STUDENT_TEST))

    rows, decoded = [], {}
    for tid in sorted(df["trip_id"].unique()):
        it = cache.get(tid)
        clf, _ = decode(it, "clf", params)
        hmm, _ = decode(it, "clf_hmm", params)
        decoded[tid] = (clf, hmm)
        lab = it["label"]
        ok_c, ok_h = clf == lab, hmm == lab
        if len(clf) < WINDOW:
            continue
        u, v = it["G"].graph["u"], it["G"].graph["v"]
        twin = (u[clf] == v[lab]) & (v[clf] == u[lab])
        visible_fix = ok_h & ~ok_c & ~twin
        twin_h = (u[hmm] == v[lab]) & (v[hmm] == u[lab])
        loss = ok_c & ~ok_h & ~twin_h
        x, y = it["trip"].track["x"].to_numpy(), it["trip"].track["y"].to_numpy()
        n_w = len(x) - WINDOW + 1
        moved = np.hypot(x[WINDOW - 1:] - x[:n_w], y[WINDOW - 1:] - y[:n_w])
        gain = windows(visible_fix.astype(int) - loss.astype(int), WINDOW).astype(float)
        gain[moved < MIN_MOVE_M] = np.nan
        if np.isnan(gain).all():
            continue
        for kind, s in (("fix", int(np.nanargmax(gain))), ("nohelp", int(np.nanargmin(gain)))):
            rows.append(dict(trip_id=tid, kind=kind, start=s, gain_points=int(gain[s]),
                             clf_correct=int(ok_c[s:s + WINDOW].sum()), hmm_correct=int(ok_h[s:s + WINDOW].sum())))
    win = pd.DataFrame(rows)
    fixes = win[win["kind"] == "fix"].sort_values("gain_points", ascending=False).head(N_FIX)
    nohelp = win[win["kind"] == "nohelp"].sort_values("gain_points").head(1)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat([fixes, nohelp]).to_csv(OUT / "map_windows.csv", index=False)

    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    for k, r in enumerate(fixes.itertuples(), 1):
        clf, hmm = decoded[r.trip_id]
        title = (f"Trip {r.trip_id}, points {r.start}-{r.start + WINDOW - 1}: HMM fixed "
                 f"{r.gain_points} points (classifier {r.clf_correct}/{WINDOW}, +HMM {r.hmm_correct}/{WINDOW})")
        path = MAPS_DIR / f"hmm_fix_{k}_{r.trip_id}.html"
        window_map(cache.get(r.trip_id), r.start, clf, hmm, title).save(path)
        print(f"saved {path.name}: {title}")
    for r in nohelp.itertuples():
        clf, hmm = decoded[r.trip_id]
        title = (f"Trip {r.trip_id}, points {r.start}-{r.start + WINDOW - 1}: HMM did not help "
                 f"(classifier {r.clf_correct}/{WINDOW}, +HMM {r.hmm_correct}/{WINDOW})")
        path = MAPS_DIR / f"hmm_nohelp_{r.trip_id}.html"
        window_map(cache.get(r.trip_id), r.start, clf, hmm, title).save(path)
        print(f"saved {path.name}: {title}")


if __name__ == "__main__":
    main()
