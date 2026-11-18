"""Report maps (section 6): full-trip Folium maps of the predicted route, plus PNG screenshots.

Run (after models/lgbm.joblib and models/hmm_params.json exist):
    python -m scripts.report_maps                 # all 10 student-test trips + screenshots
    python -m scripts.report_maps --no-screenshots

For every STUDENT-TEST trip, the route is predicted with the same code the API uses
(pathsnap.pipeline.predict_trip: classifier + HMM/Viterbi, never reading the route).
Then a map is drawn with viz.trip_map containing the four layers the brief requires:
original GPS observations, nearby road network, reference route (allowed here: these are
student-test report maps, never shown in the UI) and the predicted route. Each layer can
be toggled, and there is a legend and start/end markers.

Screenshots: the interactive HTML maps (these and the EDA / HMM maps already in
reports/maps) are rendered to PNG with a headless Chromium browser (Edge or Chrome),
because the PDF report needs static images. Map tiles come from OpenStreetMap, so this
step needs an internet connection. Set BROWSER to a chrome/msedge executable if
auto-detection fails.

Outputs:
  reports/maps/st_route_<trip>.html          interactive maps
  reports/figures/maps/<map name>.png        screenshots for the report
  experiments/results/report_maps.csv        per-trip map details (points, route arcs, flags)
"""
import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

from pathsnap.config import FIGURES_DIR, MAPS_DIR, RESULTS_DIR, STUDENT_TEST
from pathsnap.io import list_trips, load_trip
from pathsnap.pipeline import predict_trip
from pathsnap.viz import trip_map

SHOT_DIR = FIGURES_DIR / "maps"
BROWSER_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser", "microsoft-edge",
]


def find_browser() -> str | None:
    if os.environ.get("BROWSER"):
        return os.environ["BROWSER"]
    for c in BROWSER_CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    return None


def screenshot(browser: str, html: Path, png: Path, size=(1400, 900), wait_ms: int = 10000) -> bool:
    """Render an HTML map to PNG in a headless browser; virtual time lets the map tiles load."""
    png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={size[0]},{size[1]}",
           f"--virtual-time-budget={wait_ms}", f"--screenshot={png.resolve()}", html.resolve().as_uri()]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    return png.exists() and png.stat().st_size > 10_000


def build_maps() -> list[Path]:
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    rows, paths = [], []
    for tid in list_trips(STUDENT_TEST):
        t0 = time.perf_counter()
        pred = predict_trip(tid, partition=STUDENT_TEST)                 # route-free prediction
        trip = load_trip(tid, STUDENT_TEST, with_route=True)              # reference only for drawing
        m = trip_map(trip, reference_route=trip.route, predicted_route=pred["route_arcs"],
                     title=f"Trip {tid} (student-test): GPS, reference route and predicted route")
        path = MAPS_DIR / f"st_route_{tid}.html"
        m.save(path)
        paths.append(path)
        rows.append({"trip_id": tid, "n_points": pred["n_points"], "n_route_arcs": len(pred["route_arcs"]),
                     "n_reference_arcs": len(trip.route), **pred["flags"], "seconds": round(time.perf_counter() - t0, 2)})
        print(f"{tid}: {pred['n_points']} points, {len(pred['route_arcs'])} predicted arcs -> {path.name}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "report_maps.csv", index=False)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-screenshots", action="store_true")
    args = ap.parse_args()
    paths = build_maps()
    if args.no_screenshots:
        return
    browser = find_browser()
    if browser is None:
        print("No Chrome/Edge found: set BROWSER to take screenshots. HTML maps were still written.")
        return
    # Also screenshot the existing EDA and HMM maps (not the world overview, which is mostly ocean).
    extra = [p for p in sorted(MAPS_DIR.glob("*.html")) if p.name.startswith(("eda_trip_0", "hmm_"))]
    ok = 0
    for html in paths + extra:
        png = SHOT_DIR / f"{html.stem}.png"
        good = screenshot(browser, html, png)
        ok += good
        print(f"  screenshot {png.name}: {'ok' if good else 'FAILED'}", flush=True)
    print(f"{ok}/{len(paths) + len(extra)} screenshots written to {SHOT_DIR}")


if __name__ == "__main__":
    main()
