"""Build the technical report PDF from the logged results.

Run:  python -m scripts.build_report

Every table is read from the result files in experiments/ and every figure comes from
reports/figures/, so the report always matches the repository. Re-run it after any
experiment is re-run.

Deployment screenshots: put the photos/screenshots taken during the two-laptop demo
(Laptop 1 console with the API logs, the UI on Laptop 2, /health from Laptop 2) into
reports/figures/deployment/ as PNG or JPG and re-run. Until then the report shows a
placeholder box in section 7.

Output: reports/PathSnap_Technical_Report.pdf
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph,  # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from pathsnap.config import FIGURES_DIR, REPO_ROOT, RESULTS_DIR  # noqa: E402

OUT_PDF = REPO_ROOT / "reports" / "PathSnap_Technical_Report.pdf"
GEN = FIGURES_DIR / "report"            # charts generated only for the report
DEPLOY_SHOTS = FIGURES_DIR / "deployment"
REPO_URL = "https://github.com/Sidhant-holla/pathsnap"

# ============================================================================ data
mc = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
over = pd.read_csv(RESULTS_DIR / "overfitting_check.csv").set_index("model")
abl = pd.read_csv(RESULTS_DIR / "feature_ablation.csv")
imp = pd.read_csv(RESULTS_DIR / "lgbm_feature_importance.csv", index_col=0)["importance"]
imp = 100 * imp / imp.sum()
build = pd.read_csv(RESULTS_DIR / "candidates" / "feature_build_summary.csv").set_index("partition")
inv = pd.read_csv(RESULTS_DIR / "eda" / "trip_inventory.csv", dtype={"trip_id": str})
tvu = pd.read_csv(REPO_ROOT / "experiments" / "hpo" / "tuned_vs_untuned.csv", index_col=0)
best = json.loads((REPO_ROOT / "experiments" / "hpo" / "best_params.json").read_text())
hmm = pd.read_csv(RESULTS_DIR / "hmm" / "ablation_student_test.csv").set_index("method")
hmm_def = pd.read_csv(RESULTS_DIR / "hmm" / "ablation_student_test_defaults.csv").set_index("method")
hmm_trip = pd.read_csv(RESULTS_DIR / "hmm" / "ablation_per_trip.csv")
hmm_params = json.loads((REPO_ROOT / "models" / "hmm_params.json").read_text())
runs = {n: json.loads((RESULTS_DIR / f"{n}.json").read_text())
        for n in ["logreg", "gnb", "knn", "rf", "lgbm", "lgbm_tuned", "lgbm_ranker", "mlp"]}


def st(n, k):
    return runs[n]["metrics"]["student_test"][k]


def cv(n, k):
    return runs[n]["metrics"]["cv"][k]


# ============================================================================ generated charts
GEN.mkdir(parents=True, exist_ok=True)


def pipeline_diagram(path):
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.axis("off")
    boxes = [("Raw trip files\n.track .nodes .arcs", "#dfe7f2"), ("Project to metres\n(UTM per trip)", "#dfe7f2"),
             ("Candidate roads\nwithin 50 m (max 16)", "#dfe7f2"), ("18 features per\n(point, road) pair", "#dfe7f2"),
             ("LightGBM\nP(correct road)", "#fde2c8"), ("HMM + Viterbi\nconnected route", "#fde2c8"),
             ("API returns route\n+ accuracy", "#d9efd9")]
    w, gap = 1.36, 0.22
    for i, (text, col) in enumerate(boxes):
        x = i * (w + gap)
        ax.add_patch(plt.Rectangle((x, 0.3), w, 1.0, facecolor=col, edgecolor="#555", lw=1))
        ax.text(x + w / 2, 0.8, text, ha="center", va="center", fontsize=8.5)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + w + gap, 0.8), xytext=(x + w, 0.8), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(3 * (w + gap) + w / 2, 0.05, "route labels (.route) are only used to make training labels and to score",
            ha="center", fontsize=7.5, color="#a33")
    ax.set_xlim(-0.05, len(boxes) * (w + gap))
    ax.set_ylim(0, 1.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


GROUPS = {
    "Relative distance": ["f_dist_gap", "f_dist_rank", "f_dist", "f_n_cands"],
    "Direction": ["f_heading_diff", "f_heading_gap", "f_heading_reliable"],
    "Neighbourhood": ["f_ctx_gap", "f_ctx_dist"],
    "Topology / geometry": ["f_endpoint_dist", "f_arc_len", "f_proj_frac", "f_in_deg", "f_out_deg"],
    "GPS quality / motion": ["f_speed", "f_jitter", "f_dt_prev", "f_dt_next"],
}
GCOL = {"Relative distance": "#3b6ea5", "Direction": "#e08214", "Neighbourhood": "#5aa469",
        "Topology / geometry": "#8c6bb1", "GPS quality / motion": "#9a9a9a"}
group_of = {f: g for g, fs in GROUPS.items() for f in fs}


def importance_chart(path):
    s = imp.sort_values()
    s = s[s >= 0.3]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(s.index, s.values, color=[GCOL[group_of[f]] for f in s.index])
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=7)
    ax.set_xlabel("share of LightGBM split gain (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in GCOL.values()], list(GCOL), fontsize=7,
              loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def hmm_trip_chart(path):
    d = hmm_trip.pivot_table(index="trip_id", columns="method", values="top1_conf")
    d = d.sort_values("clf")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = np.arange(len(d))
    ax.barh(y - 0.2, d["clf"], height=0.4, color="#f28e2b", label="classifier only")
    ax.barh(y + 0.2, d["clf_hmm"], height=0.4, color="#d7301f", label="classifier + HMM")
    ax.set_yticks(y, [f"{int(t):08d}" for t in d.index], fontsize=7)
    ax.set_xlim(0.7, 1.0)
    ax.set_xlabel("Top-1 accuracy per student-test trip", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


pipeline_diagram(GEN / "pipeline.png")
importance_chart(GEN / "feature_importance.png")
hmm_trip_chart(GEN / "hmm_per_trip.png")

# ============================================================================ layout helpers
ss = getSampleStyleSheet()
TITLE = ParagraphStyle("TITLE", parent=ss["Title"], fontSize=22, leading=27, spaceAfter=6)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, leading=19, spaceBefore=10, spaceAfter=6,
                    textColor=colors.HexColor("#1f3b63"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, leading=15, spaceBefore=8, spaceAfter=4,
                    textColor=colors.HexColor("#1f3b63"))
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=2)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.6, leading=13.2, spaceAfter=4)
BUL = ParagraphStyle("BUL", parent=B, leftIndent=12, bulletIndent=2, spaceAfter=2)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.2, leading=10.5, alignment=TA_CENTER,
                     textColor=colors.HexColor("#555555"), spaceAfter=8)
CELL = ParagraphStyle("CELL", parent=B, fontSize=7.8, leading=9.6, spaceAfter=0)
CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)
story = []
fig_no = [0]
tab_no = [0]


def h1(t):
    story.append(CondPageBreak(60 * mm))
    story.append(Paragraph(t, H1))


def h2(t):
    story.append(CondPageBreak(35 * mm))
    story.append(Paragraph(t, H2))


def h3(t):
    story.append(Paragraph(t, H3))


def p(t):
    story.append(Paragraph(t, B))


def bullets(items):
    for it in items:
        story.append(Paragraph(it, BUL, bulletText="•"))


FIG_LABELS = {}   # label -> figure number, filled while building
REFS = {}         # numbers from the previous pass, used in the text


def R(label):
    """Figure number for a label (two-pass build, so forward references work)."""
    return str(REFS.get(label, "?"))


def fig(path, width_mm, caption, label=None):
    path = Path(path)
    iw, ih = ImageReader(str(path)).getSize()
    w = width_mm * mm
    h = w * ih / iw
    max_h = 205 * mm
    if h > max_h:
        w, h = w * max_h / h, max_h
    fig_no[0] += 1
    if label:
        FIG_LABELS[label] = fig_no[0]
    story.append(KeepTogether([Image(str(path), width=w, height=h),
                               Paragraph(f"<b>Figure {fig_no[0]}.</b> {caption}", CAP)]))


def table(rows, widths, caption=None, highlight=None):
    data = [[Paragraph(str(c), CELLB if r == 0 else CELL) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b63")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b0b0b0")),
             ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2)]
    for r in range(2, len(rows), 2):
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f2f5f9")))
    for r in (highlight or []):
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fff3d6")))
    t.setStyle(TableStyle(style))
    items = []
    if caption:
        tab_no[0] += 1
        items.append(Paragraph(f"<b>Table {tab_no[0]}.</b> {caption}", CAP))
    items.append(t)
    story.append(KeepTogether(items))
    story.append(Spacer(1, 6))


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def f3(x):
    return f"{x:.3f}"


def build_story():
    global story
    story.clear()
    fig_no[0] = 0
    tab_no[0] = 0
    # ============================================================================ title page
    story += [Spacer(1, 30 * mm),
              Paragraph("PathSnap", TITLE),
              Paragraph("Map-Matching and Route Reconstruction from Noisy GPS", ParagraphStyle(
                  "sub", parent=B, fontSize=14, leading=18, alignment=TA_CENTER)),
              Spacer(1, 6 * mm),
              Paragraph("Technical Report — UE24CS352A Machine Learning Hackathon, PES University (RR Campus)",
                        ParagraphStyle("sub2", parent=B, alignment=TA_CENTER, fontSize=10.5)),
              Spacer(1, 14 * mm)]
    team = [["Member", "Name", "Main responsibility"],
            ["A", "Sidhant Holla", "Data pipeline: projection, labels, candidates, features; MLP model"],
            ["B", "Samarth Kulkarni", "Exploratory data analysis, visualisation and maps; Naive Bayes and KNN models"],
            ["C", "Greeshma Gangadhar", "Evaluation framework; Logistic Regression, Random Forest, LightGBM, ranker; Bayesian HPO"],
            ["D", "Pranav Prabhakar", "Road graph, HMM/Viterbi decoding, API, user interface and deployment"]]
    table(team, [16, 38, 120])
    story += [Spacer(1, 8 * mm),
              Paragraph(f"Code, results and models: <link href='{REPO_URL}'>{REPO_URL}</link>",
                        ParagraphStyle("link", parent=B, alignment=TA_CENTER)),
              Spacer(1, 4 * mm),
              Paragraph("Every number in this report is read from the result files in the repository "
                        "(<font face='Courier'>python -m scripts.build_report</font> rebuilds this PDF).",
                        ParagraphStyle("note", parent=CAP))]
    story.append(PageBreak())

    # ============================================================================ 1. introduction
    h1("1. Problem and overview of our approach")
    p("A GPS receiver in a car records its position about once per second, but the readings are noisy. Tall "
      "buildings, tunnels and signal loss can push a point tens of metres away from the road the car is "
      "actually on. The goal of PathSnap is to take these noisy points and work out which roads the car really "
      "drove on. This is called <b>map-matching</b>.")
    p("Our system works in two stages. First, for every GPS point we look at the nearby roads (we call them "
      "<i>candidates</i>) and a machine-learning classifier gives each candidate a probability of being the "
      "correct road. Second, because a car cannot jump between roads that are not connected, a Hidden Markov "
      "Model (HMM) decoded with the Viterbi algorithm picks the most likely <i>connected</i> sequence of roads "
      "for the whole trip. A FastAPI server runs this pipeline, and a separate user interface lets an evaluator "
      "choose a trip and see the predicted route and its accuracy.")
    fig(GEN / "pipeline.png", 175, "The PathSnap pipeline. The reference route is only used to create training "
        "labels and to score predictions; the prediction itself never reads it.")
    p(f"<b>Headline result.</b> On the 10 student-test trips, the final system (LightGBM + HMM) matches "
      f"<b>{pct(hmm.loc['clf_hmm', 'top1_conf'])}</b> of GPS points to the correct road and reconstructs the route "
      f"with an F1 score of <b>{f3(hmm.loc['clf_hmm', 'route_f1'])}</b>. Simply picking the nearest road gets only "
      f"{pct(hmm.loc['nearest', 'top1_conf'])}.")
    p("<b>Data partitions.</b> We used the 80 <b>training</b> trips for fitting, cross-validation and all tuning, and the "
      "10 <b>student-test</b> trips only for comparing models during development. The 10 <b>evaluator</b> trips were "
      "never used for training, tuning, feature selection or choosing models; they are only run through the API "
      "during the final evaluation. Each analysis below states which partition it uses.")

    # ============================================================================ 2. dataset & EDA
    h1("2. Dataset understanding and exploratory data analysis")
    p("<i>Partition used in this section: train + student-test (90 trips). The evaluator trips were not opened. "
      "Code: <font face='Courier'>notebooks/01_eda.ipynb</font>; tables in "
      "<font face='Courier'>experiments/results/eda/</font>.</i>")
    h2("2.1 What the raw data looks like")
    p("Each trip is a folder with four plain-text files with no header row:")
    bullets([
        "<b>.track</b>: the GPS readings, one per line: longitude, latitude and time in seconds.",
        "<b>.nodes</b>: points of the road map (junctions and bends): longitude and latitude. The line number is the node id.",
        "<b>.arcs</b>: road segments from one node to another. The line number is the arc id. Arcs are "
        "<i>one-way</i>, so a normal two-way street appears as two arcs pointing in opposite directions.",
        "<b>.route</b>: the arc ids the car actually drove, in order. This is the reference answer.",
    ])
    p(f"There are {len(inv[inv.partition == 'train'])} training and {len(inv[inv.partition == 'student_test'])} student-test trips "
      f"with {int(inv.n_points.sum()):,} GPS points in total. Trips range from {int(inv.n_points.min())} to "
      f"{int(inv.n_points.max()):,} points. Every trip has all four files (about 90 MB in total; the largest single "
      f"file is 3.9 MB), and all values parse cleanly as numbers. Two things surprised us. First, <b>every trip has its own road map</b>, "
      f"from {int(inv.n_arcs.min()):,} to {int(inv.n_arcs.max()):,} arcs, so an arc id only means something inside its own "
      f"trip. Second, <b>the dataset gives no features and no label per GPS point</b>: the route lists roads, not which "
      "point belongs to which road. We had to build both ourselves.")
    fig(FIGURES_DIR / "eda" / "01_trip_sizes.png", 165, "Number of GPS points, road-network arcs and route arcs per trip (log scale).")

    h2("2.2 Missing-value analysis")
    p("We checked every file of every trip. There are <b>no missing (NaN) values</b>, no impossible coordinates, and the "
      "timestamps always increase. Three trips have fractional timestamps, which we simply treat as seconds.")
    p("For GPS data the real \"missing values\" are <b>missing readings in time</b>. Most steps are exactly one second "
      f"apart ({pct(0.9668)}), but {int((inv.gaps_gt10s > 0).sum())} of 90 trips have at least one gap longer than 10 seconds, "
      f"and {int((inv.gaps_gt60s > 0).sum())} have a gap longer than a minute. In the extreme case, trip 90 has a 42-hour gap. "
      "We decided <b>not to fill these gaps</b>: inventing GPS points would invent evidence. Instead we split a trip into "
      "separate pieces wherever the gap is longer than 10 seconds, and nothing in our pipeline looks across a gap.")
    fig(FIGURES_DIR / "eda" / "02_missing_seconds_per_trip.png", 170, "Share of each trip's duration with no GPS reading (blue = train, orange = student-test).")
    fig(FIGURES_DIR / "eda" / "03_sampling_gaps.png", 165, "Left: time between consecutive readings. Right: the 15 trips with the largest single gap.")

    h2("2.3 Patterns we observed")
    bullets([
        "<b>The trips are all over the world</b> (latitude −34.6 to 61.7, 25 different UTM zones). Distances in degrees are "
        "meaningless here (one degree of longitude is 111 km at the equator but only 53 km at 61° north), so we convert "
        "every trip to metres using its own UTM projection. We checked this against exact distances on the Earth's "
        "ellipsoid: the error is at most 0.04%.",
        "<b>Most streets are two-way.</b> About 95% of arcs have a twin pointing the other way. For about 64% of GPS "
        "points, the nearest arc is tied with its reverse twin at exactly the same distance. Distance alone cannot "
        "tell the two directions apart, which is why direction (heading) became our most important feature.",
        "<b>The road networks are connected but cut off at the edges.</b> No node is isolated, and the largest "
        "strongly connected part holds 96% of nodes, but many small pieces appear where the map was clipped.",
        "<b>The reference routes are good but not perfect.</b> 79 of 90 routes are fully connected; 11 have 1–3 breaks, "
        "and 10 trips drive some road twice. We therefore treat the route as a reference, not as perfect truth.",
        "<b>Some GPS points are far from every road.</b> 151 readings imply impossible speeds (over 252 km/h). In trip 54 "
        "the car spends 4.3 hours inside an area with no mapped road, so those points cannot be matched to anything.",
    ])
    fig(FIGURES_DIR / "eda" / "04_trip_locations.png", 150, "Geographical extent: the centre of every train (blue) and "
        "student-test (orange) trip. The trips are spread across several continents.")
    fig(FIGURES_DIR / "eda" / "05_trajectory_samples.png", 160, "GPS trajectories of 12 training trips (colour shows time).")
    fig(FIGURES_DIR / "eda" / "07_network_structure.png", 170, "Road networks: node out-degree, arc length, and how much of each network is strongly connected.")
    fig(FIGURES_DIR / "eda" / "09_off_network_trip54.png", 95, "Trip 54: thousands of GPS points (red) far from any mapped road while the vehicle is off the network.")
    fig(FIGURES_DIR / "maps" / "eda_trip_00000001.png", 160, "Folium map of training trip 1: GPS points (blue), nearby road network (grey) and "
        "reference route (green). The layers can be switched on and off.")

    h2("2.4 How we preprocessed the data")
    bullets([
        "<b>Projection</b>: each trip is converted from longitude/latitude to metres in its own UTM zone.",
        "<b>Training labels</b>: we match every GPS point to one arc of the reference route. The car is only allowed to "
        "stay on its current route arc or move forward along the route, and we choose the matching that keeps the points "
        "closest to their arcs overall (a dynamic-programming method similar to Viterbi). This handles two-way streets "
        "and roads driven twice correctly; simply taking the nearest route arc disagrees with it on up to 15% of a trip. "
        "The median distance between a point and its labelled arc is 4.9 m.",
        f"<b>Unreliable points</b>: points more than 50 m from their labelled arc are marked <i>low-confidence</i> "
        f"({pct(build.loc['train', 'low_conf_pct'] / 100)} of training points, mostly the off-network episodes). We do not "
        "delete them; we train only on the confident points and report results both with and without them.",
        "<b>Candidates</b>: for each point we keep the arcs within 50 m, at most the 16 closest. We chose these numbers "
        "on the training trips: 99% of points are within 34.5 m of their correct arc, and with 50 m / 16 the correct arc is "
        f"missing for only {build.loc['train', 'miss_rate_confident_pct']:.2f}% of points, with "
        f"{build.loc['train', 'mean_cands_per_point']:.1f} candidates per point on average.",
        "<b>Scaling</b>: for models that are sensitive to scale (Logistic Regression, Naive Bayes, KNN, MLP) we apply "
        "log(1+x) to long-tailed features and then standardise them. The scaler is always fitted on the training part only.",
    ])
    fig(FIGURES_DIR / "candidate_radius_tradeoff.png", 165, "Choosing the candidate search radius on training trips: missed true roads (left) against the number of candidates per point (right).")

    h2("2.5 Features we extracted, and why")
    p("Each row of our training table is one (GPS point, candidate road) pair, described by 18 numbers. None of them "
      "uses the reference route, so exactly the same code runs inside the API. They fall into five groups:")
    table([["Group", "Main features", "Why we built it"],
           ["Relative distance", "distance to the road, distance minus the closest candidate's distance, rank by distance, number of candidates",
            "The real question is <i>which</i> candidate wins for this point, so being closer than the alternatives matters more than the raw distance."],
           ["Direction", "difference between the car's heading (smoothed over ±2 s) and the road's direction",
            "The only way to tell the two directions of a two-way street apart."],
           ["Neighbourhood", "how well the road fits the GPS points 1–3 seconds before and after",
            "A single noisy reading can sit nearer the wrong road, but its neighbours usually don't."],
           ["Topology / geometry", "distance to the nearest junction, road length, position along the road, number of connecting roads",
            "Most mistakes happen near junctions, where several roads meet."],
           ["GPS quality / motion", "speed, jitter, time since previous / until next reading",
            "To flag readings that should be trusted less."]],
          [30, 64, 80], "The five feature groups.")
    p("To check that the features really matter we did two experiments with LightGBM (5-fold cross-validation on the "
      "training trips). The <b>feature importance</b> (how much each feature reduces the model's error across all its "
      "tree splits) is shown in Figure " + R("importance") + ". The relative-distance, neighbourhood and direction features carry about 85% of "
      "it. In the <b>ablation</b> experiment we removed one group at a time and retrained:")
    a = abl.set_index("experiment")
    base = a.loc["All 18 features", "cv_top1"]
    table([["Feature group removed", "CV Top-1 without it", "Change", "Folds worse (of 5)", "Top-1 with only this group"]] +
          [[g, f3(a.loc[f"without {g}", "cv_top1"]), f"{100 * a.loc[f'without {g}', 'delta_top1_vs_all']:+.2f} pts",
            str(int(a.loc[f"without {g}", "folds_worse_than_all"])), f3(a.loc[f"only {g}", "cv_top1"])] for g in GROUPS],
          [44, 32, 24, 30, 40], f"Feature-group ablation (LightGBM, all 18 features: {f3(base)}).", highlight=[2])
    p("Removing <b>any</b> group makes the model worse, in every fold. Direction is irreplaceable: without it accuracy "
      "falls by 29 points, because nothing else can separate a road from its reverse twin. Interestingly, relative "
      "distance has the largest importance but costs little when removed, because the neighbourhood features also "
      "carry distance information. So importance is not the same as necessity. We kept all 18 features in every model: "
      "this was feature <i>engineering</i>, not feature selection.")
    fig(GEN / "feature_importance.png", 120, "LightGBM gain importance per feature, coloured by group.", label="importance")
    fig(FIGURES_DIR / "feature_ablation.png", 170, "Feature-group ablation: drop in accuracy when a group is removed (left) and accuracy with one group alone (right).")

    # ============================================================================ 3. models
    h1("3. Model experimentation and comparison")
    p("<i>Partitions: 5-fold cross-validation (CV) on the 80 training trips, then the model is retrained on all 80 trips "
      "and scored once on the 10 student-test trips. Code: <font face='Courier'>scripts/train_models.py</font>, "
      "<font face='Courier'>scripts/train_mlp.py</font>, <font face='Courier'>src/pathsnap/evaluate.py</font>.</i>")
    p("Every model gets exactly the same 18 features, the same folds and the same metrics, so the comparison is fair. "
      "The folds are split <b>by trip</b> (GroupKFold): GPS points one second apart are almost identical, so mixing points "
      "of one trip between training and validation would make the scores look better than they really are. Models are "
      "trained on confident points only, but validation always counts every point.")
    p("<b>How we measure.</b> Our main metric is <b>Top-1 accuracy</b>: for each GPS point we take the candidate with the "
      "highest probability and check whether it is the correct road. A point whose correct road is not among its "
      "candidates counts as wrong. We chose it because the route is built from each point's best candidate, not from a "
      "0.5 threshold. We also report <b>macro F1</b> (the average of the F1 scores for \"correct road\" and \"wrong road\" "
      "candidates at a 0.5 threshold), <b>log loss</b> (how well-calibrated the probabilities are, which matters because "
      "the HMM uses them), and <b>route continuity</b> (the share of consecutive predicted roads that are actually "
      "connected, before any HMM). For reference, always picking the nearest road gives 55.9% Top-1, and picking the "
      "nearest road with the better heading gives 82.6%.")

    story.append(CondPageBreak(130 * mm))
    h2("3.1 Overall model comparison")
    order = ["LightGBM", "LightGBM (tuned)", "LightGBM Ranker", "Random Forest", "MLP", "KNN (distance-weighted)",
             "Logistic Regression", "Gaussian Naive Bayes"]
    m = mc.set_index("model").loc[order]
    rows = [["Model", "Family", "Key configuration", "CV macro F1", "CV Top-1", "Contin. *",
             "ST macro F1", "ST Top-1", "ST log loss", "Infer. ms / 1k"]]
    for name, r in m.iterrows():
        rows.append([name, r["algorithm_family"], str(r["key_configuration"]).replace("_", " "), f3(r["cv_macro_f1_candidate"]),
                     f"{f3(r['cv_top1'])} ± {r['cv_top1_std']:.3f}", f3(r["route_continuity_st"]), f3(r["st_macro_f1_candidate"]),
                     f3(r["st_top1"]), f3(r["st_log_loss"]), f"{r['inference_ms_per_1k_rows']:.1f}"])
    table(rows, [22, 22, 28, 13, 19, 14, 14, 14, 13, 14],
          "All models (same 18 features and folds). ST = student-test. Macro F1 is candidate level. *Route continuity of the "
          "classifier-only route on student-test (before the HMM). Highlighted row = deployed model.", highlight=[1])
    p("LightGBM is the best untuned classifier on every accuracy measure. Its tuned version and the ranker are only a few "
      "tenths of a point different (Sections 3.2 and 4). Inference time is measured per 1,000 candidate rows; a typical "
      "trip has about 17,000.")
    extra = [["Model", "Accuracy", "Precision", "Recall", "F1", "Candidate miss rate", "Training time (s)"]]
    for key, nm in [("lgbm", "LightGBM"), ("lgbm_tuned", "LightGBM (tuned)"), ("lgbm_ranker", "LightGBM ranker"),
                    ("rf", "Random Forest"), ("mlp", "MLP"), ("knn", "KNN"), ("logreg", "Logistic Regression"),
                    ("gnb", "Gaussian Naive Bayes")]:
        s_ = runs[key]["metrics"]["student_test"]
        extra.append([nm, f3(s_["cand_accuracy"]), f3(s_["cand_precision"]), f3(s_["cand_recall"]), f3(s_["cand_f1"]),
                      pct(s_["miss_rate_conf"], 2), f"{runs[key]['metrics']['train_time_s']:.1f}"])
    table(extra, [42, 20, 20, 20, 18, 28, 26],
          "More student-test metrics: candidate-level accuracy, precision, recall and F1 (threshold 0.5), the candidate "
          "miss rate (points whose correct road is not among the candidates, the same for every model because all models "
          "share the candidates) and the time to train on all 80 trips.")
    p("Accuracy looks high for every model because 88% of candidates are wrong roads, so even a weak model gets most of "
      "them right. That is why we use Top-1 accuracy, not plain accuracy, to compare models.")

    h2("3.2 Every model in detail")
    p("For each model we answer the same questions: why we tried it, what its strengths and limitations turned out to be, "
      "its important hyperparameters and their initial values, what it optimises, how it did on student-test, and whether "
      "it over- or underfitted. For over/underfitting we scored each final model on its own training data and compared "
      "that with the CV and student-test scores (Figure " + R("overfit") + ").")


    def model_section(title, key, why, strengths, limits, hyper, loss, fit, extra=None):
        story.append(CondPageBreak(75 * mm))   # keep each model's heading with its answers
        h3(title)
        r = runs[key]["metrics"]
        s_, c_ = r["student_test"], r["cv"]
        bullets([f"<b>Why we tried it:</b> {why}",
                 f"<b>Strengths we saw:</b> {strengths}",
                 f"<b>Limitations we saw:</b> {limits}",
                 f"<b>Hyperparameters (initial values):</b> {hyper}",
                 f"<b>What it optimises:</b> {loss}",
                 f"<b>Student-test results:</b> Top-1 {f3(s_['top1_conf'])} (all points {f3(s_['top1_all'])}), "
                 f"macro F1 {f3(s_['cand_macro_f1'])} (observation-level {f3(s_['obs_macro_f1_conf'])}), "
                 f"precision {f3(s_['cand_precision'])}, recall {f3(s_['cand_recall'])}, log loss {f3(s_['log_loss'])}. "
                 f"CV Top-1 {f3(c_['top1_conf_mean'])} ± {c_['top1_conf_std']:.3f}, CV macro F1 {f3(c_['cand_macro_f1_mean'])}.",
                 f"<b>Over- or underfitting?</b> {fit}"] + ([extra] if extra else []))


    def ov(name):
        o = over.loc[name]
        return f"training-data Top-1 {f3(o['train_top1'])} vs CV {f3(o['cv_top1'])} vs student-test {f3(o['st_top1'])}"


    model_section(
        "Logistic Regression (linear)", "logreg",
        "the simplest baseline. Several of our features already compare a candidate with its rivals, so a weighted sum "
        "might already rank candidates well. It also tells us how much a non-linear model is worth.",
        "the fastest model (0.3 ms per 1k rows, 2.7 s to train). Its coefficients agree with our intuition: the "
        "neighbourhood gap, heading difference and distance rank have the largest weights.",
        "lower Top-1 than every tree-based model (CV 0.861), and behind LightGBM in every fold. Heading only matters when two "
        "candidates are at the same distance, an interaction that a straight-line model cannot express. Balanced class "
        "weights give high recall (0.951) but low precision (0.689) and poorly calibrated probabilities.",
        "C = 1.0 (L2 regularisation), class_weight = balanced, lbfgs solver, max_iter = 2000.",
        "L2-regularised binary cross-entropy (log loss), with extra weight on the rarer \"correct road\" rows.",
        f"<b>underfitting.</b> {ov('Logistic Regression')}: almost no gap, but every score is low, so the model is "
        "too simple for this problem.")
    model_section(
        "Gaussian Naive Bayes (probabilistic)", "gnb",
        "a very fast probabilistic baseline that tests whether the features are useful on their own, one at a time.",
        "trains in under a second and still ranks candidates reasonably (Top-1 0.861).",
        "it assumes the features are independent of each other, but four of them all measure distance. It counts the same "
        "evidence several times and becomes over-confident: the worst log loss of all models and 5,854 false positives "
        "(precision 0.748).",
        "var_smoothing = 1e-9 (default), after our standard scaling.",
        "no iterative loss: it estimates a mean and variance per feature and class (maximum likelihood) and combines them with Bayes' rule.",
        f"<b>underfitting.</b> {ov('Gaussian Naive Bayes')}: no gap, lowest accuracy. The model is too simple for correlated features.")
    model_section(
        "K-Nearest Neighbours, distance-weighted (instance-based)", "knn",
        "it makes no assumption about the shape of the decision boundary: a candidate that looks like known correct "
        "candidates is probably correct.",
        "better than the linear and probabilistic models (CV 0.867, student-test 0.872) with no real training step.",
        "by far the slowest to predict (174.8 ms per 1k rows, about 70× LightGBM), because every prediction searches the stored "
        "training data, and the saved model contains that data (15 MB). To keep it practical we trained it on every 5th GPS point "
        "(neighbouring points one second apart are almost identical).",
        "k = 50 neighbours, distance weighting, Euclidean distance on the scaled features, every 5th point.",
        "none: it is a \"lazy\" learner. The prediction is the distance-weighted share of correct-road neighbours.",
        f"{ov('KNN')}. The 4-point gap is expected memorisation: with distance weighting a stored point is its own nearest "
        "neighbour. CV and student-test agree, so it generalises consistently.")
    model_section(
        "Random Forest (bagging ensemble)", "rf",
        "a robust non-linear model that captures feature interactions (such as distance × heading) without scaling and "
        "is not easily thrown off by noisy labels.",
        "second-best untuned classifier (CV 0.878, student-test 0.890), well-calibrated probabilities (log loss 0.063). "
        "It ranks the same features highest as LightGBM: heading difference, heading gap and distance gap.",
        "slowest to train (50 s) and behind LightGBM in every fold. A full-size forest on 1.5 million rows would be "
        "hundreds of MB, so we limited the trees.",
        "200 trees, max_features = sqrt, min_samples_leaf = 25, max_samples = 0.2 (each tree sees a 20% sample).",
        "each tree greedily minimises Gini impurity; the forest's probability is the average over trees.",
        f"<b>mild overfitting.</b> {ov('Random Forest')}: a 2-point gap kept small by the leaf size and sampling.")
    model_section(
        "LightGBM (gradient boosting): selected model", "lgbm",
        "boosted trees are usually the strongest choice for tabular data like ours: they model interactions, train "
        "quickly on 1.5 million rows and give probabilities the HMM can use.",
        "best CV Top-1 (0.881) and best student-test Top-1 (0.896) of the untuned models, best macro F1, lowest log "
        "loss, and ahead of Random Forest and Logistic Regression in every fold (Figure " + R("folds") + ").",
        "it has the most capacity, so it fits the training trips most tightly (see below); inference is 2.6 ms per 1k rows "
        "(about 45 ms per trip), slower than the linear model and MLP but fine for a live API.",
        "learning_rate 0.05, num_leaves 63, min_child_samples 50, subsample 0.8, colsample_bytree 0.8, reg_lambda 1, "
        "up to 3000 trees with early stopping (100 rounds) on 15% of the training <i>trips</i> held out inside each fit. "
        "The final model keeps 328 trees.",
        "binary log loss (cross-entropy).",
        f"<b>mild overfitting.</b> {ov('LightGBM')}. The training-loss curve (Figure {R('lgbm_curve')}) keeps falling while the "
        "validation loss flattens, and early stopping cuts training at the best validation point. Even so, its held-out "
        "accuracy is the best of all models, so the extra capacity pays off.")
    model_section(
        "LightGBM, Bayesian-tuned (boosting)", "lgbm_tuned",
        "after choosing LightGBM we wanted to know whether better hyperparameters, found by Bayesian optimisation "
        "(Section 4), would improve it.",
        "slightly higher numbers everywhere: student-test Top-1 0.900 against 0.896, and CV Top-1 0.882 against 0.881.",
        "the gains are much smaller than the 2.9-point difference between folds, and it was better in only 3 of 5 folds. "
        "It is about 10 times slower to predict (26.7 ms per 1k rows), 4 times slower to train, and 9 times bigger, "
        "because the small learning rate needs 976 trees.",
        "the best values from 51 Optuna trials: learning rate 0.0103, num_leaves 202, min_child_samples 24, "
        "colsample_bytree 0.789, subsample 0.659, reg_lambda 3.50, reg_alpha 6.38; early stopping kept 976 trees.",
        "binary log loss (cross-entropy), the same as the untuned model.",
        f"<b>mild overfitting, slightly more than the untuned model.</b> {ov('LightGBM (tuned)')}. Because the gain is so "
        "small and it is much slower, we kept the untuned model for the final system.")
    model_section(
        "LightGBM ranker (learning to rank)", "lgbm_ranker",
        "the real decision is \"which of this point's candidates wins\", which is a ranking problem. A LambdaRank model "
        "with one group per GPS point optimises exactly that.",
        "the highest student-test precision (0.929) and lowest student-test log loss.",
        "no measurable gain: CV Top-1 0.882 vs 0.881 for the normal LightGBM (better in 3 of 5 folds), student-test 0.896 vs 0.896. "
        "A ranker gives scores, not probabilities, so we turn them into probabilities with a softmax over each point's candidates.",
        "same tree settings as LightGBM; early stopping on NDCG@1; 202 trees.",
        "the LambdaRank ranking loss (NDCG@1, which equals per-point Top-1).",
        f"<b>clear overfitting.</b> {ov('LightGBM Ranker')}: the largest gap of all models, without better results on unseen trips.")
    model_section(
        "Multi-layer perceptron (neural network)", "mlp",
        "it learns its own combinations of features (for example, \"heading only matters when distances are tied\") and gives "
        "probabilities for the HMM.",
        "close second (student-test 0.893), the <b>best classifier-only route continuity (0.819)</b>, and fast inference "
        "(0.45 ms per 1k rows).",
        "needs scaling, slightly below LightGBM in CV (0.876), and the best stopping epoch varies a lot between folds "
        "(from 4 to 30), so the right training length depends on the data.",
        "two hidden layers (64, 32), ReLU, Adam, learning rate 0.001, L2 alpha 0.0001, batch 1024, up to 60 epochs, early "
        "stopping with patience 6 on the last 10% of training rows (whole trips held out).",
        "binary cross-entropy (log loss), no class weights so that the probabilities stay calibrated.",
        f"<b>a healthy fit.</b> {ov('MLP')}. The per-epoch curves (Figure {R('mlp_curve')}) show the validation loss levelling off by "
        "epochs 4–12 and never rising; early stopping kept the epoch-12 weights. The validation loss is <i>lower</i> than the "
        "training loss because the held-out trips happen to be cleaner; they were never trained on.")

    h2("3.3 Model visualisations")
    p("All plots below are made from the logged results (<font face='Courier'>scripts/plot_model_comparison.py</font>). "
      "Training/validation loss curves are only shown for the models that genuinely train in iterations (LightGBM, its "
      "tuned version, the ranker and the MLP). The other models have no training history, so we do not draw curves for them.")
    fig(FIGURES_DIR / "comparison" / "cv_macro_f1.png", 170, "Cross-validation macro F1 per model (mean ± standard deviation over the 5 folds).")
    fig(FIGURES_DIR / "comparison" / "student_test_top1_f1.png", 150, "Student-test Top-1 accuracy and macro F1.")
    fig(FIGURES_DIR / "comparison" / "log_loss.png", 135, "Log loss (lower is better). Naive Bayes is badly over-confident.")
    fig(FIGURES_DIR / "comparison" / "fold_stability.png", 140, "Top-1 accuracy in each CV fold. The same folds are hard for every model, and LightGBM is on top in every fold, which is stronger evidence than comparing averages.", label="folds")
    fig(FIGURES_DIR / "comparison" / "accuracy_vs_speed.png", 115, "Accuracy against inference cost.")
    fig(FIGURES_DIR / "comparison" / "overfitting_check.png", 170, "Each final model scored on its own training data, in CV and on student-test.", label="overfit")
    fig(FIGURES_DIR / "lgbm_loss_curve.png", 105, "LightGBM training and validation log loss per boosting iteration (real values).", label="lgbm_curve")
    fig(FIGURES_DIR / "mlp_loss_curve.png", 105, "MLP training and validation log loss per epoch (real values).", label="mlp_curve")
    fig(FIGURES_DIR / "lgbm_tuned_loss_curve.png", 100, "Tuned LightGBM training and validation log loss per boosting iteration.")
    fig(FIGURES_DIR / "lgbm_ranker_loss_curve.png", 100, "LightGBM ranker training and validation NDCG per boosting iteration "
        "(a ranking score, so higher is better).")
    fig(FIGURES_DIR / "comparison" / "confusion_matrices.png", 170, "Candidate-level confusion matrices on student-test (threshold 0.5, rows normalised).")

    h2("3.4 Choosing the final model")
    p("We chose <b>LightGBM</b> because: (1) it has the highest CV Top-1, our main metric, and is ahead of Random Forest and "
      "Logistic Regression in <i>every</i> fold; (2) it also generalises best to the unseen student-test trips; (3) it has the "
      "best-calibrated probabilities, which the HMM relies on; (4) it is fast enough for a live API; and (5) it outputs "
      "probabilities directly, unlike the ranker; and (6) it is a compact model (328 trees, 1 MB) compared with the "
      "Random Forest (17 MB) or KNN (15 MB, because it stores the training data). We did not choose it on training accuracy: the ranker scores higher on its "
      "own training data but no better on unseen trips. The MLP is a good fallback if prediction speed mattered more.")

    # ============================================================================ 4. HPO
    h1("4. Bayesian hyperparameter optimisation (LightGBM)")
    p("<i>Partition: 80 training trips only (same 5-fold cross-validation). Student-test was used once, after the search, to "
      "compare the tuned and untuned models. Code: <font face='Courier'>scripts/hpo.py</font>; records in "
      "<font face='Courier'>experiments/hpo/</font>.</i>")
    p("We used <b>Optuna's TPE sampler</b>, a Bayesian method: instead of trying settings blindly (grid or random search), it "
      "builds a model of which regions of the search space gave good results so far and proposes new trials there. A "
      "<i>median pruner</i> stopped clearly bad trials early, after a fold or two, to save time.")
    bullets([
        "<b>Hyperparameters and ranges:</b> learning rate 0.01–0.2, number of leaves 15–255, minimum samples per leaf 10–500, "
        "regularisation λ 0.001–30 and α 0.001–10 (all on a log scale, because their effect is multiplicative), plus the column "
        "and row sampling fractions 0.4–1.0 and 0.5–1.0. The number of trees is not searched; early stopping picks it.",
        "<b>Objective:</b> maximise the mean cross-validated Top-1 accuracy (confident points) on the training trips.",
        f"<b>Trials:</b> {best['n_trials']} trials ({best['trial_states'].get('COMPLETE', 0)} completed, "
        f"{best['trial_states'].get('PRUNED', 0)} stopped early by the pruner; one was interrupted when the run was paused and "
        f"resumed). Total search time about {best['search_time_s_all_trials'] / 3600:.1f} hours.",
        "<b>Best values</b> (trial " + str(best["best_trial"]) + "): " + ", ".join(
            f"{k.replace('_', ' ')} {v:.3g}" for k, v in best["best_params"].items()) + ".",
    ])
    fig(FIGURES_DIR / "hpo_history.png", 125, "Optimisation history: every completed trial lands between 0.880 and 0.882, so the objective is flat around the default settings.")
    fig(FIGURES_DIR / "hpo_param_importance.png", 125, "Which hyperparameters mattered most during the search (fANOVA importance).")
    TVU_LABEL = {"CV top-1 conf (mean)": "CV Top-1 accuracy (mean of 5 folds)", "CV candidate macro F1": "CV macro F1",
                 "CV log loss": "CV log loss", "ST top-1 conf": "Student-test Top-1 accuracy",
                 "ST candidate macro F1": "Student-test macro F1", "ST log loss": "Student-test log loss",
                 "Inference ms / 1k rows": "Inference time (ms per 1k rows)", "Train time s (80 trips)": "Training time (s, 80 trips)"}
    rows = [["Measure", "Untuned", "Tuned"]]
    for k in ["CV top-1 conf (mean)", "CV candidate macro F1", "CV log loss", "ST top-1 conf", "ST candidate macro F1",
              "ST log loss", "Inference ms / 1k rows", "Train time s (80 trips)"]:
        u, t = tvu.loc[k, "untuned"], tvu.loc[k, "tuned"]
        fmt = (lambda x: f"{x:.1f}") if ("ms" in k or "time" in k) else f3
        rows.append([TVU_LABEL[k], fmt(u), fmt(t)])
    table(rows, [70, 35, 35], "Tuned versus untuned LightGBM, same data and metrics.")
    p("<b>Did tuning help?</b> Only a little. CV Top-1 rose by 0.1 points, which is far smaller than the 2.9-point "
      "difference between folds, and the tuned model was better in only 3 of the 5 folds. Student-test Top-1 rose by 0.3 points, "
      "while the CV log loss got very slightly worse. The tuned model is also about 10 times slower to predict and 4 times "
      "slower to train, because its low learning rate needs 976 trees instead of 328. Our conclusion is that the default "
      "settings were already close to the best possible for these features, so we <b>deploy the untuned LightGBM</b> and keep "
      "the tuned model in the repository for comparison.")

    # ============================================================================ 5. HMM
    h1("5. HMM and Viterbi route reconstruction")
    p("<i>Partitions: decoder settings tuned on training trips only; every comparison below is on the 10 student-test trips. "
      "Code: <font face='Courier'>src/pathsnap/graph.py</font>, <font face='Courier'>hmm.py</font>, "
      "<font face='Courier'>scripts/hmm_ablation.py</font>.</i>")
    p("The classifier looks at each GPS point on its own, so its choices can jump between roads that are not connected. "
      "The HMM fixes this by looking at the whole trip at once.")
    bullets([
        "<b>Hidden states:</b> at each GPS point, the hidden state is which of its candidate roads the car is really on. Each "
        "state keeps its arc id and where on the arc the point falls, so every decoded state maps back to a real road segment.",
        "<b>Emission probabilities</b> (how well a road fits one GPS point): we take LightGBM's probabilities for the point's "
        "candidates, divide them by their sum so they add up to one, and use log(probability) × λ. The weight λ balances "
        "trusting the classifier against trusting the road network.",
        "<b>Transition probabilities</b> (how plausible a move from road A to road B is): we compute the <i>driving</i> distance "
        "from A to B along the road network (shortest path with Dijkstra, following one-way directions) and compare it with "
        "the straight-line distance between the two GPS points. If the car really went from A to B, the two distances should "
        "be similar, so the log-probability is −|driving distance − straight distance| / β.",
        "<b>Viterbi:</b> a dynamic-programming algorithm that finds the single best sequence of roads for the whole trip. "
        "We go through the trip point by point. The best score for \"the car is on road j at point t\" is the emission "
        "score of road j plus the best value, over every candidate road i at point t−1, of (best score of i + transition "
        "score from i to j). We remember which i gave that best value. At the end we start from the best road at the "
        "last point and follow the remembered choices backwards, which gives one road for every GPS point. We add "
        "log-probabilities instead of multiplying probabilities, so even a 19,000-point trip doesn't underflow.",
        "<b>Missing candidates and large gaps:</b> if a point has no road within 50 m, we give it its nearest road at any "
        "distance (and flag it), so every point still gets an answer. If two readings are more than G seconds apart, we cut the "
        "trip there and decode the pieces separately, because a long gap says nothing about which road follows which.",
        "<b>Disconnected road segments:</b> a move to a road that cannot be reached (or would need an impossible speed) gets a "
        "large but <i>finite</i> penalty instead of probability zero. With a zero, one bad reading or a missing road in the map "
        "would make every path impossible and the decoding would fail.",
        f"<b>Tuning:</b> λ, β and G were tuned with Optuna on 24 training trips, using out-of-fold classifier probabilities "
        f"(the final model is over-confident on its own training trips). Final values: λ = "
        f"{hmm_params['classifier_hmm']['lam']:.3f}, β = {hmm_params['classifier_hmm']['beta']:.1f} m, "
        f"G = {hmm_params['classifier_hmm']['gap_s']:.0f} s.",
    ])
    names = {"nearest": "Nearest road", "dist_hmm": "HMM with distance only (no classifier)",
             "clf": "Classifier only (LightGBM)", "clf_hmm": "Classifier + HMM (final system)"}
    rows = [["Method", "Top-1 accuracy", "Route continuity", "Route precision", "Route recall", "Route F1"]]
    for k, n in names.items():
        r = hmm.loc[k]
        rows.append([n, pct(r["top1_conf"]), pct(r["continuity"]), f3(r["route_precision"]), f3(r["route_recall"]), f3(r["route_f1"])])
    table(rows, [62, 24, 24, 22, 20, 18], "Classifier-only predictions against classifier + HMM decoding (student-test, 10 trips).", highlight=[4])
    p("<b>Did the HMM improve the route? Yes.</b> Top-1 rose from 89.6% to 92.8%, route continuity from 80.9% to 90.4%, and "
      "route F1 from 0.839 to 0.974. It improved Top-1 on <b>all 10</b> student-test trips (Figure " + R("hmm_trips") + "). Almost all of the gain "
      "is in route <i>precision</i> (0.736 to 0.958): without the HMM the predicted route zig-zags onto neighbouring and "
      "reverse-direction roads that were never driven. Recall is about 0.99 for every method. Interestingly, the HMM "
      "with distance alone and no classifier already beats the classifier on its own, which shows how much information the "
      "road network carries; the classifier then adds another 2.2 points on top.")
    p("Continuity is 90%, not 100%, because between two readings a car can pass a very short road without any point "
      "landing on it. For the displayed route we fill such gaps with the shortest connecting path.")
    fig(GEN / "hmm_per_trip.png", 120, "Top-1 accuracy per student-test trip, classifier only against classifier + HMM.", label="hmm_trips")
    p("<b>Where the HMM did not help.</b> We looked for honest failure cases:")
    bullets([
        "On trip 77 the decoder switches to the next road about 20 points too early, although the classifier is confident "
        "and correct there (Figure " + R("hmm_nohelp") + "). With our tuned λ = 0.28, the smoothness of the route outweighs the classifier. With "
        "λ = 1 that stretch is fully correct. Over all student-test trips, the untuned default settings would score "
        f"{pct(hmm_def.loc['clf_hmm', 'top1_conf'])} instead of {pct(hmm.loc['clf_hmm', 'top1_conf'])}, but almost all of that "
        "difference comes from this one trip (the defaults are better on 3 trips and worse on 4). We kept the settings tuned on "
        "the training trips, so that student-test stays a fair test rather than something we tuned on.",
        "When a car is parked, the GPS wanders around and the direction of travel becomes meaningless, so the HMM can pick the "
        "opposite-direction twin of the correct road. This is the same physical road (zero distance error), but it counts as "
        "wrong. It is really a limitation of the labels.",
    ])
    fig(FIGURES_DIR / "maps" / "hmm_fix_1_00000076.png", 150, "Trip 76 at a motorway interchange: the classifier alone (orange) jumps to the parallel road; "
        "the HMM route (red dashed) stays on the correct road (green). 30/60 points correct before, 60/60 after.", label="hmm_fix1")
    fig(FIGURES_DIR / "maps" / "hmm_fix_3_00000000.png", 150, "Trip 0: another stretch the HMM corrects (48/60 to 60/60 points).")
    fig(FIGURES_DIR / "maps" / "hmm_nohelp_00000077.png", 150, "Trip 77: a case where the HMM makes things worse (53/60 to 40/60), discussed above.", label="hmm_nohelp")

    # ============================================================================ 6. maps
    h1("6. Geographical visualisation using Folium")
    p("<i>Partition: student-test (reference routes are allowed in these report maps; the UI never shows them). Code: "
      "<font face='Courier'>src/pathsnap/viz.py</font>, <font face='Courier'>scripts/report_maps.py</font>, "
      "<font face='Courier'>scripts/hmm_maps.py</font>. Interactive HTML maps are in <font face='Courier'>reports/maps/</font>.</i>")
    p("Every map is produced by our own code from the actual pipeline output. Each one shows the <b>original GPS points</b> "
      "(blue), the <b>nearby road network</b> (grey), the <b>reference route</b> (green) and the <b>predicted route</b> (red), "
      "with a legend, start and end markers, and a layer control to switch each layer on or off. We draw the routes on top of "
      "the GPS points so that a dense trace does not hide them. The map zooms to the trip's extent automatically. The "
      "predicted routes come from exactly the same <font face='Courier'>predict_trip</font> function the API uses.")
    fig(FIGURES_DIR / "maps" / "st_route_00000076.png", 160, "Student-test trip 76: the predicted route (red) lies on top of the reference route (green) for the whole trip.")
    fig(FIGURES_DIR / "maps" / "st_route_00000056.png", 160, "Student-test trip 56: a longer trip through a dense city network.")
    p("The windows in Figures " + R("hmm_fix1") + "–" + R("hmm_nohelp") + " zoom in on places where the classifier alone and the HMM disagree, and add the "
      "classifier-only route in orange so the effect of the HMM can be seen directly.")

    # ============================================================================ 7. API / UI
    h1("7. API, user interface and deployment")
    p("The system runs on two laptops, as required. <b>Laptop 1</b> runs the FastAPI backend with the trained model, the HMM, "
      "the road networks and the server-side scoring. <b>Laptop 2</b> runs a Streamlit user interface, which has no data and no "
      "model of its own and only talks to the API over the network. The step-by-step setup is in "
      "<font face='Courier'>docs/DEPLOY.md</font>.")
    bullets([
        "<b>Endpoints:</b> <font face='Courier'>GET /trips</font> lists the trips being served (the 10 evaluator trips on demo "
        "day, read from the data folder, never hard-coded); <font face='Courier'>POST /predict</font> takes a trip id and returns "
        "the GPS points, the predicted route and its accuracy; <font face='Courier'>GET /health</font> is a quick status check.",
        "<b>What happens per request:</b> load the trip <i>without</i> its route → convert to metres → candidates and features "
        "→ LightGBM → Viterbi → connected route → coordinates for the map. Only after that does a separate scoring module read "
        "the reference route, on the server, to compute the accuracy. The reference route is never sent to the UI.",
        "<b>Logs:</b> every request is printed in the API console with the trip id, status and time taken, plus how long each "
        "stage took, so the evaluators can see the requests arriving on Laptop 1.",
        "<b>Speed:</b> 0.3–1.8 seconds per trip end to end on one laptop.",
        "<b>Errors:</b> an unknown trip returns 404 and a crash returns 500 with a message; the UI shows readable errors, "
        "including a hint about the hotspot, IP address or firewall if the API cannot be reached.",
    ])
    fig(FIGURES_DIR / "ui" / "ui_02_trip_00000076.png", 165, "The UI after predicting a trip: summary metrics and the full metrics table returned by the API.")
    fig(FIGURES_DIR / "ui" / "ui_02_trip_00000076_map.png", 165, "The same screen scrolled to the map: GPS points and the predicted route.")
    shots = sorted([p_ for p_ in DEPLOY_SHOTS.glob("*") if p_.suffix.lower() in (".png", ".jpg", ".jpeg")]) \
        if DEPLOY_SHOTS.exists() else []

    def shot_caption(path):
        import re
        stem = path.stem.lower()
        trip = re.search(r"(\d{8})", stem)
        trip = trip.group(1) if trip else ""
        if ("console" in stem or "cmd" in stem) and "startup" in stem:
            return ("Two-laptop run, Laptop 1 console: the API starts (model lgbm, 10 trips) and receives its first requests "
                    "from Laptop 2 (client 10.171.60.55), including crash-only checks where scoring is skipped.")
        if ("console" in stem or "cmd" in stem) and "prediction" in stem:
            return ("Two-laptop run, Laptop 1 console: successful POST /predict requests from Laptop 2 (status 200, about "
                    "0.4–0.7 s each), with the time spent in each stage and the accuracy computed on the server.")
        if "console" in stem or "cmd" in stem:
            return "Two-laptop run, Laptop 1 console: the API's log lines for the incoming requests (trip, status, time per stage)."
        if "health" in stem:
            return "Two-laptop run: Laptop 2's browser opens /health on Laptop 1's API, so the connection works."
        if "map" in stem:
            return f"Two-laptop run, Laptop 2 UI, trip {trip}: GPS points and the predicted route on the map."
        if "metrics" in stem or "ui" in stem:
            return f"Two-laptop run, Laptop 2 UI after predicting trip {trip}: the metrics returned by Laptop 1's API."
        return "Two-laptop run: " + path.stem.replace("_", " ") + "."

    if shots:
        p("The screenshots below are from the real two-laptop setup on the same hotspot: the UI on Laptop 2 sends each "
          "request to the API on Laptop 1 (address 10.171.60.10:8000).")
    for s in shots:
        fig(s, 150 if "health" not in s.stem else 120, shot_caption(s))
    if not any("console" in s.stem.lower() or "cmd" in s.stem.lower() for s in shots):
        box = Table([[Paragraph("<b>Screenshot still to add: the Laptop 1 console</b> showing the "
                                "<font face='Courier'>POST /predict</font> log lines (the brief asks for API responses "
                                "\"on the CMD of the receiving laptop\"). Save it in "
                                "<font face='Courier'>reports/figures/deployment/</font> with \"console\" in the file name "
                                "and run <font face='Courier'>python -m scripts.build_report</font> again.", B)]],
                    colWidths=[170 * mm])
        box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#d7301f")),
                                 ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff3f0")),
                                 ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        story += [box, Spacer(1, 6)]

    # ============================================================================ 8. conclusion
    h1("8. Conclusions and limitations")
    bullets([
        "Building the right <b>features</b> mattered more than the choice of model: all good models land within about two "
        "points of each other, while removing the direction features alone costs 29 points.",
        "<b>LightGBM</b> was the best classifier, and Bayesian optimisation confirmed that its default settings were already "
        "near-optimal.",
        "The <b>HMM</b> adds a lot on top of the classifier, mostly by keeping the route on connected roads: route F1 0.84 → 0.97.",
        "<b>Limitations.</b> Our labels come from aligning the GPS to the reference route, which is itself imperfect (11 routes "
        "have breaks). At a U-turn on a two-way street, or when a car is parked, the labelled direction can be wrong; we "
        "measured that a heading-aware alignment would change 0.95% of labels and kept the original labels for consistency. "
        "Points in off-network areas cannot be matched by any method. Finally, the student-test set has only 10 trips, so "
        "differences of a few tenths of a point between models should not be over-interpreted.",
    ])

    h2("Reproducibility")
    p("All code, saved models and result files are in the repository. Seeds are fixed at 42. The README lists the commands "
      "that regenerate every number and figure in this report, in order: features, models, MLP, HPO, comparison plots, "
      "over/underfitting check, feature ablation, HMM tuning and ablation, maps, and finally this report.")


# ============================================================================ build
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(18 * mm, 10 * mm, "PathSnap: technical report")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"page {doc.page}")
    canvas.restoreState()


build_story()                 # pass 1: collect figure numbers
REFS.update(FIG_LABELS)
build_story()                 # pass 2: text uses the real numbers
doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                        bottomMargin=18 * mm, title="PathSnap: Technical Report", author="PathSnap team")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"wrote {OUT_PDF} ({fig_no[0]} figures, {tab_no[0]} tables)")
