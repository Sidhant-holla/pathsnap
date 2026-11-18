"""Folium maps of trips: GPS points, road network, reference route, predicted route.

Folium/Leaflet expect [lat, lon], while our files store lon first, so every
coordinate is flipped here and nowhere else.

Each layer is a separate FeatureGroup, so it can be switched on and off from the layer
control. Only arcs near the GPS track are drawn: a whole trip network (up to 300k
arcs) would make the HTML huge and unreadable.

Never pass the reference route when building maps for the UI or the evaluator trips.
Ground truth must not leave the server. Reference routes are only for report/EDA maps
of train and student-test trips.
"""
import folium
import numpy as np
import shapely

from .geo import arc_table, project_trip
from .io import Trip

COLORS = {
    "gps": "#1f5fd1",        # blue dots
    "network": "#9a9a9a",    # grey lines
    "reference": "#1a9850",  # green
    "predicted": "#d7301f",  # red
}


def _arc_lines(trip: Trip, arc_ids) -> list:
    """[[lat, lon], [lat, lon]] line per arc, for a multi-part PolyLine."""
    arcs = trip.arcs.loc[list(arc_ids)]
    lon, lat = trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy()
    u, v = arcs["u"].to_numpy(), arcs["v"].to_numpy()
    return [[[lat[a], lon[a]], [lat[b], lon[b]]] for a, b in zip(u, v)]


def _dedupe_consecutive(arc_ids) -> list:
    """Collapse per-point arc ids (5,5,5,9,9,14) into a route (5,9,14)."""
    out = []
    for a in arc_ids:
        if not out or out[-1] != a:
            out.append(int(a))
    return out


def nearby_arcs(trip: Trip, buffer_m: float = 150.0, max_points: int = 3000) -> np.ndarray:
    """Arc ids within buffer_m of the GPS track (track subsampled for speed)."""
    if "x" not in trip.track:
        project_trip(trip)
    arcs = arc_table(trip)
    coords = np.stack([arcs[["ax", "ay"]].to_numpy(), arcs[["bx", "by"]].to_numpy()], axis=1)
    tree = shapely.STRtree(shapely.linestrings(coords))
    step = max(1, len(trip.track) // max_points)
    pts = shapely.points(trip.track["x"].to_numpy()[::step], trip.track["y"].to_numpy()[::step])
    _, hit = tree.query(pts, predicate="dwithin", distance=buffer_m)
    return arcs.index.to_numpy()[np.unique(hit)]


def _legend(entries: list[tuple[str, str]]) -> folium.Element:
    rows = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};'
        f'margin-right:6px;vertical-align:middle"></span>{name}</div>' for name, c in entries)
    return folium.Element(
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
        'padding:8px 10px;border:1px solid #888;border-radius:4px;font:13px sans-serif">'
        f'<b>Legend</b>{rows}</div>')


def trip_map(trip: Trip, reference_route=None, predicted_route=None, show_network: bool = True,
             network_buffer_m: float = 150.0, max_gps_points: int = 3000, title: str | None = None) -> folium.Map:
    """Interactive map of one trip.

    reference_route / predicted_route: lists of arc ids (consecutive duplicates are fine,
    so per-point predictions can be passed directly). Leave reference_route=None for
    anything shown in the UI.
    """
    track = trip.track
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    legend = []

    if show_network:
        fg = folium.FeatureGroup(name="Road network (nearby arcs)")
        folium.PolyLine(_arc_lines(trip, nearby_arcs(trip, network_buffer_m)),
                        color=COLORS["network"], weight=2, opacity=0.6).add_to(fg)
        fg.add_to(m)
        legend.append(("Road network", COLORS["network"]))

    if reference_route is not None:
        fg = folium.FeatureGroup(name="Reference route")
        folium.PolyLine(_arc_lines(trip, _dedupe_consecutive(reference_route)),
                        color=COLORS["reference"], weight=6, opacity=0.8).add_to(fg)
        fg.add_to(m)
        legend.append(("Reference route", COLORS["reference"]))

    if predicted_route is not None:
        fg = folium.FeatureGroup(name="Predicted route")
        folium.PolyLine(_arc_lines(trip, _dedupe_consecutive(predicted_route)),
                        color=COLORS["predicted"], weight=3, opacity=0.9, dash_array="6 4").add_to(fg)
        fg.add_to(m)
        legend.append(("Predicted route", COLORS["predicted"]))

    fg = folium.FeatureGroup(name="GPS observations")
    step = max(1, len(track) // max_gps_points)
    for row in track.iloc[::step].itertuples():
        folium.CircleMarker([row.lat, row.lon], radius=2, color=COLORS["gps"], fill=True,
                            fill_opacity=0.8, weight=0, tooltip=f"t = {row.t:.0f} s").add_to(fg)
    fg.add_to(m)
    legend.append((f"GPS points{' (every %dth)' % step if step > 1 else ''}", COLORS["gps"]))

    first, last = track.iloc[0], track.iloc[-1]
    folium.Marker([first.lat, first.lon], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([last.lat, last.lon], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)

    m.fit_bounds([[track["lat"].min(), track["lon"].min()], [track["lat"].max(), track["lon"].max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(_legend(legend))
    if title:
        m.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:12px;left:60px;z-index:9999;background:white;padding:4px 10px;'
            f'border:1px solid #888;border-radius:4px;font:bold 14px sans-serif">{title}</div>'))
    return m
