"""PathSnap evaluator UI (Laptop 2).

Run from the repo root:
    set API_URL=http://<laptop-1-ip>:8000        (PowerShell: $env:API_URL="http://<ip>:8000")
    streamlit run ui/app.py

The UI only talks to the API: it lists the trips from GET /trips, sends the chosen trip
to POST /predict and draws what comes back (GPS points + predicted route + metrics). It
has no access to the data or the model, and the API never sends reference routes, so the
UI cannot show one.
"""
from __future__ import annotations

import math
import os

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_S = 180
MAX_GPS_MARKERS = 2500          # more markers than this makes the browser sluggish

GPS_COLOR = "#2563eb"
ROUTE_COLOR = "#dc2626"

st.set_page_config(page_title="PathSnap", layout="wide")


# --------------------------------------------------------------------------- API calls

class ApiError(Exception):
    pass


def _call(method: str, path: str, **kwargs):
    url = f"{API_URL}{path}"
    try:
        r = requests.request(method, url, timeout=kwargs.pop("timeout", 10), **kwargs)
    except requests.ConnectionError:
        raise ApiError(f"Cannot reach the API at {API_URL}. Is it running on Laptop 1, are both laptops on the "
                       f"same network, and is port 8000 allowed through Laptop 1's firewall?")
    except requests.Timeout:
        raise ApiError(f"The API at {API_URL} did not answer in time.")
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise ApiError(f"API error {r.status_code} on {path}: {detail}")
    return r.json()


@st.cache_data(ttl=30, show_spinner=False)
def get_trips():
    return _call("GET", "/trips")


def predict(trip_id: str):
    return _call("POST", "/predict", json={"trip_id": trip_id}, timeout=TIMEOUT_S)


# --------------------------------------------------------------------------- map

def build_map(res: dict) -> folium.Map:
    gps = res["gps"]
    lats, lons = [p[0] for p in gps], [p[1] for p in gps]
    # streamlit-folium does not reliably apply fit_bounds, so set centre and zoom explicitly:
    # the largest web-mercator zoom at which the track's extent fits a ~700x600 px map.
    lat0, lat1, lon0, lon1 = min(lats), max(lats), min(lons), max(lons)
    span_lon = max(lon1 - lon0, 1e-4)
    span_lat = max((lat1 - lat0) / max(math.cos(math.radians((lat0 + lat1) / 2)), 0.2), 1e-4)
    zoom = int(max(3, min(18, math.floor(math.log2(min(360 * 700 / (256 * span_lon),
                                                          360 * 560 / (256 * span_lat)))))))
    m = folium.Map(location=[(lat0 + lat1) / 2, (lon0 + lon1) / 2], zoom_start=zoom,
                   tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat0, lon0], [lat1, lon1]], padding=(20, 20))

    g_track = folium.FeatureGroup(name="GPS observations", show=True)
    step = max(1, len(gps) // MAX_GPS_MARKERS)
    for lat, lon, t in gps[::step]:
        folium.CircleMarker([lat, lon], radius=2, color=GPS_COLOR, fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"t = {t:.0f} s").add_to(g_track)
    g_track.add_to(m)

    g_route = folium.FeatureGroup(name="Predicted route", show=True)
    for piece in res["route_coords"]:
        if len(piece) >= 2:
            folium.PolyLine(piece, color=ROUTE_COLOR, weight=5, opacity=0.8).add_to(g_route)
    g_route.add_to(m)

    folium.Marker([lats[0], lons[0]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lats[-1], lons[-1]], tooltip="End", icon=folium.Icon(color="black", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    legend = (f'<div style="position: fixed; bottom: 36px; left: 12px; z-index: 9999; background: white; '
              f'padding: 8px 12px; border: 1px solid #999; border-radius: 6px; font-size: 13px; color: #111;">'
              f'<span style="color:{GPS_COLOR}">&#9679;</span> GPS observations'
              f'{" (every %dth shown)" % step if step > 1 else ""}<br>'
              f'<span style="color:{ROUTE_COLOR}">&#9644;</span> Predicted route</div>')
    m.get_root().html.add_child(folium.Element(legend))
    return m


# --------------------------------------------------------------------------- metrics

METRIC_ROWS = [
    ("top1_conf", "Point top-1 accuracy (confident points)"),
    ("top1_all", "Point top-1 accuracy (all points)"),
    ("continuity", "Route continuity (connected consecutive arcs)"),
    ("route_precision", "Route precision"),
    ("route_recall", "Route recall"),
    ("route_f1", "Route F1"),
    ("route_precision_len", "Route precision (length-weighted)"),
    ("route_recall_len", "Route recall (length-weighted)"),
    ("route_f1_len", "Route F1 (length-weighted)"),
    ("raw_route_f1", "Route F1 before gap filling"),
]


def pct(v):
    return f"{100 * v:.2f}%" if isinstance(v, (int, float)) else "n/a"


def show_metrics(res: dict):
    m = res.get("metrics") or {}
    if not m.get("available"):
        st.warning(f"Metrics not available: {m.get('reason', 'unknown reason')}")
        return
    c = st.columns(4)
    c[0].metric("Top-1 (all points)", pct(m.get("top1_all")))
    c[1].metric("Route F1", pct(m.get("route_f1")))
    c[2].metric("Route continuity", pct(m.get("continuity")))
    c[3].metric("Server time", f"{res['timing_s'].get('total', 0) + res['timing_s'].get('score', 0):.2f} s")
    table = pd.DataFrame([(label, pct(m[k])) for k, label in METRIC_ROWS if k in m],
                         columns=["Metric", "Value"])
    st.dataframe(table, hide_index=True, width="stretch")


# --------------------------------------------------------------------------- page

st.title("PathSnap: map-matching evaluator")

with st.sidebar:
    st.subheader("Connection")
    st.code(API_URL, language=None)
    if st.button("Refresh trip list"):
        get_trips.clear()

try:
    trips = get_trips()
except ApiError as e:
    st.error(str(e))
    st.stop()

ids = trips.get("trip_ids", [])
if not ids:
    st.warning(f"The API has no trips for partition '{trips.get('partition')}'.")
    st.stop()

with st.sidebar:
    st.caption(f"Partition: {trips.get('partition')} ({len(ids)} trips)")
    trip_id = st.selectbox("Trip", ids)
    run = st.button("Predict route", type="primary", width="stretch")

if run:
    with st.spinner(f"Running the pipeline for trip {trip_id} on the API..."):
        try:
            st.session_state["result"] = predict(trip_id)
            st.session_state.pop("error", None)
        except ApiError as e:
            st.session_state["error"] = str(e)
            st.session_state.pop("result", None)

if "error" in st.session_state:
    st.error(st.session_state["error"])

res = st.session_state.get("result")
if res:
    st.subheader(f"Trip {res['trip_id']}  ·  {res['n_points']:,} GPS points  ·  model: {res['model']}")
    show_metrics(res)
    st_folium(build_map(res), height=620, use_container_width=True, returned_objects=[])
    with st.expander("Run details"):
        f = res["flags"]
        st.write({
            "candidates scored": f["n_candidates"],
            "points with no candidate (nearest-arc fallback)": f["n_fallback_points"],
            "soft breaks in the decoded path": f["n_soft_breaks"],
            "GPS segments (split at long time gaps)": f["n_segments"],
            "route gaps that could not be filled": f["n_unfilled_gaps"],
            "server timing (s)": res["timing_s"],
            "HMM parameters": res["hmm_params"],
        })
else:
    st.info("Pick a trip in the sidebar and press **Predict route**.")
