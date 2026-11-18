"""Tests for the evaluation harness. Run:  python -m tests.test_evaluate   (or pytest)."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

from pathsnap import evaluate as ev
from pathsnap import models
from pathsnap.testing import make_fake_table


def tiny():
    """Hand-checkable trip A with 5 GPS points.

    obs 0: arc 1 (label, p=.9) | arc 2 (p=.2)        -> argmax 1, correct          confident
    obs 1: arc 1 (p=.6)        | arc 3 (label, p=.4) -> argmax 1, wrong            LOW-CONF
    obs 2: arc 4 (p=.7)        | arc 5 (p=.1)        -> labelled arc 6 not a candidate (missed)
    obs 3: no candidates, labelled arc 7                                           confident
    obs 4: arc 8 (label, p=.8)                       -> correct                    confident
    """
    df = pd.DataFrame({
        "trip_id": ["A"] * 7,
        "obs_idx": [0, 0, 1, 1, 2, 2, 4],
        "arc_id": [1, 2, 1, 3, 4, 5, 8],
        "label": [1, 0, 0, 1, 0, 0, 1],
        "proba": [.9, .2, .6, .4, .7, .1, .8],
        "label_low_conf": [False, False, True, True, False, False, False],
    })
    points = pd.DataFrame({"trip_id": ["A"] * 5, "obs_idx": range(5), "label_arc": [1, 3, 6, 7, 8],
                           "low_conf": [False, True, False, False, False]})
    return df, points


def test_score_hand_computed():
    df, points = tiny()
    m = ev.score(df, points=points)
    assert m["n_points"] == 5 and m["n_points_conf"] == 4 and m["n_points_no_candidates"] == 1
    assert np.isclose(m["top1_all"], 2 / 5)          # obs 0, 4 correct of 5
    assert np.isclose(m["top1_conf"], 2 / 4)         # confident: obs 0, 2, 3, 4 -> 0 and 4 correct
    assert np.isclose(m["top1_covered"], 2 / 2)      # confident AND covered: obs 0, 4
    assert np.isclose(m["miss_rate_all"], 2 / 5)     # obs 2, 3
    assert np.isclose(m["miss_rate_conf"], 2 / 4)
    assert np.isclose(m["top1_conf"], m["top1_covered"] * (1 - m["miss_rate_conf"]))
    # candidate level on confident rows (obs 0, 2, 4): labels 1,0,0,0,1 preds(.5) 1,0,1,0,1
    # TP=2 FP=1 FN=0 TN=2
    assert m["confusion_matrix"] == [[2, 1], [0, 2]]
    assert np.isclose(m["cand_precision"], 2 / 3) and np.isclose(m["cand_recall"], 1.0)
    crow = ~df["label_low_conf"]
    assert np.isclose(m["cand_macro_f1"], f1_score(df.loc[crow, "label"], df.loc[crow, "proba"] >= .5, average="macro"))
    assert np.isclose(m["log_loss"], log_loss(df.loc[crow, "label"], df.loc[crow, "proba"]))
    assert np.isclose(m["log_loss_all"], log_loss(df["label"], df["proba"]))
    # obs-level, all points. true: A1 A3 A6 A7 A8, pred: A1 A1 A4 <none> A8
    # A1: TP1 FP1 -> 2/3; A3: 0; A6: 0; A7: 0 (FN only, <none> not a class); A4: 0; A8: 1
    # classes {A1,A3,A4,A6,A7,A8} -> (2/3 + 1) / 6
    assert np.isclose(m["obs_macro_f1_all"], (2 / 3 + 1) / 6)
    # confident: true A1 A6 A7 A8, pred A1 A4 <none> A8 -> classes A1 A4 A6 A7 A8 -> 2/5
    assert np.isclose(m["obs_macro_f1_conf"], 2 / 5)


def test_without_points_table_falls_back_to_candidate_points():
    df, _ = tiny()
    m = ev.score(df)
    assert m["n_points"] == 4                           # obs 3 is invisible without the points table
    assert np.isclose(m["top1_all"], 2 / 4)


def test_points_table_must_agree_with_labels():
    df, points = tiny()
    points.loc[0, "label_arc"] = 2
    try:
        ev.score(df, points=points)
    except ValueError:
        return
    raise AssertionError("mismatched label_arc was not detected")


def test_ties_break_to_lower_arc_id_regardless_of_row_order():
    df = pd.DataFrame({"trip_id": ["A", "A"], "obs_idx": [0, 0], "arc_id": [7, 6],
                       "label": [1, 0], "proba": [.5, .5]})
    a = ev.top1_predictions(df)
    b = ev.top1_predictions(df.iloc[::-1])
    assert a["pred_arc"].item() == b["pred_arc"].item() == 6


def test_obs_classes_are_per_trip():
    # Same arc_id 1 in two different trips must be two different classes.
    df = pd.DataFrame({"trip_id": ["A", "A", "B", "B"], "obs_idx": [0, 0, 0, 0],
                       "arc_id": [1, 2, 1, 2], "label": [1, 0, 0, 1], "proba": [.9, .1, .9, .1]})
    m = ev.score(df)
    # A: true A:1 pred A:1 (F1 1). B: true B:2 pred B:1 -> classes B:1, B:2 both F1 0.
    assert np.isclose(m["obs_macro_f1_all"], 1 / 3)


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(1)
    t = pd.Series(rng.integers(0, 30, 500).astype(str))
    p = pd.Series(np.where(rng.random(500) < .6, t, rng.integers(0, 40, 500).astype(str)))
    assert np.isclose(ev.macro_f1_classes(t, p), f1_score(t, p, average="macro", zero_division=0))


def test_continuity():
    arcs = pd.DataFrame({"u": [0, 1, 5], "v": [1, 2, 6]})        # arc0: 0->1, arc1: 1->2, arc2: 5->6
    pts = pd.DataFrame({"trip_id": ["A"] * 6, "obs_idx": range(6),
                        "pred_arc": [0, 0, np.nan, 1, 2, 2]})
    # NaN skipped; distinct sequence 0,1,2: 0->1 connected, 1->2 not -> 1/2
    assert np.isclose(ev.route_continuity(pts, {"A": arcs}), 0.5)


def test_fake_tables_are_consistent():
    df, points, arcs = make_fake_table(n_trips=6, seed=2)
    m = ev.score(df.assign(proba=df["label"].astype(float)), points=points, arcs_by_trip=arcs)
    assert np.isclose(m["top1_covered"], 1.0)                  # an oracle is right on every covered point
    assert np.isclose(m["miss_rate_all"], 1 - points["true_in_cands"].mean())
    assert m["n_points_no_candidates"] == int((points["n_cands"] == 0).sum())
    # On covered points the oracle follows the true arcs, which form a connected chain.
    # (Missed points still pick some distractor, so continuity over all points is < 1.)
    pts = ev.top1_predictions(df.assign(proba=df["label"].astype(float)), points)
    assert np.isclose(ev.route_continuity(pts[pts["covered"]], arcs), 1.0)
    assert m["continuity"] < 1.0


def test_group_cv_on_fake_table():
    df, points, arcs = make_fake_table(n_trips=15, seed=3)
    seen = []

    def make():
        m = models.make("logreg")
        orig = m.fit

        def spy(X, y):
            seen.append(len(X))
            return orig(X, y)
        m.fit = spy
        return m

    cv = ev.group_cv(make, df, n_splits=5, seed=42, points=points, verbose=False)
    oof = cv.oof
    assert len(oof) == len(df) and oof["proba"].notna().all()       # every row scored, incl. low-conf
    assert (oof.groupby("trip_id")["fold"].nunique() == 1).all()    # a trip is never split
    assert oof["fold"].nunique() == 5
    assert (cv.folds["n_train_trips"] + cv.folds["n_val_trips"] == 15).all()
    # low-confidence rows are left out of fitting
    for k, n_fit in enumerate(seen):
        tr = oof["fold"] != k
        assert n_fit == int((tr & ~oof["label_low_conf"]).sum())
    s = cv.summary
    assert s["top1_covered_mean"] > 0.6, s                           # beats nearest-arc (~0.34 here)
    cv2 = ev.group_cv(lambda: models.make("logreg"), df, seed=42, points=points, verbose=False)
    assert (cv2.oof["fold"].to_numpy() == oof["fold"].to_numpy()).all()   # same seed, same folds


def test_fit_and_score_and_logging():
    df, points, arcs = make_fake_table(n_trips=12, seed=4)
    trips = sorted(df["trip_id"].unique())
    train, test = df[df["trip_id"].isin(trips[:9])], df[df["trip_id"].isin(trips[9:])]
    model, m, pred = ev.fit_and_score(lambda: models.make("logreg"), train, test, points=points, arcs_by_trip=arcs)
    assert 0 <= m["continuity"] <= 1 and m["infer_time_s"] > 0
    assert m["n_fit_rows"] == int((~train["label_low_conf"]).sum())
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ev.log_result("lr_test", {"family": "linear", "C": 1.0}, {"student_test": m}, results_dir=d)
        ev.log_result("lr_test", {"family": "linear", "C": 1.0}, {"student_test": m}, results_dir=d)
        ev.log_result("other", {"family": "x"}, {"student_test": m}, results_dir=d)
        rec = json.loads((d / "lr_test.json").read_text())
        assert rec["metrics"]["student_test"]["confusion_matrix"] == m["confusion_matrix"]
        summ = pd.read_csv(d / "summary.csv")
        assert list(summ["name"]) == ["lr_test", "other"]          # re-run replaced, not duplicated
        assert "student_test_top1_conf" in summ and "student_test_confusion_matrix" not in summ
        path = ev.save_model("lr_test", model, ev.feature_columns(), {"C": 1.0}, models_dir=d)
        bundle = ev.load_model(path)
        assert bundle["feature_columns"] == ev.feature_columns()
        assert np.allclose(ev.predict_proba(bundle["model"], test[bundle["feature_columns"]]), pred["proba"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all evaluate tests passed")
