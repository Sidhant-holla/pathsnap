"""Coordinate projection and basic geometry, all in metres.

Why projection: longitude/latitude are angles. One degree of longitude is about 111 km
at the equator but about 53 km at latitude 61, and our trips span latitudes -26 to 61.
We project every trip to its own UTM zone, a conformal projection whose
distance error is under 0.1% within a zone, so plain Euclidean geometry on (x, y)
is accurate in metres for city-scale trips.

Bearing convention: compass degrees, 0 = north, 90 = east, measured clockwise.
"""
import numpy as np
import pandas as pd
from pyproj import Geod, Transformer

from .io import Trip


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone containing (lon, lat)."""
    zone = int((lon + 180) // 6) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def project_trip(trip: Trip) -> Trip:
    """Add x, y (metres) columns to trip.track and trip.nodes, in place, and return the trip.

    The UTM zone is chosen from the GPS track's centroid, and the same transform is
    applied to the road nodes so both live in one coordinate system.
    """
    epsg = utm_epsg(trip.track["lon"].mean(), trip.track["lat"].mean())
    # always_xy=True means input order is (lon, lat), matching our files.
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    for df in (trip.track, trip.nodes):
        df["x"], df["y"] = tf.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    trip.crs = f"EPSG:{epsg}"
    return trip


def arc_table(trip: Trip) -> pd.DataFrame:
    """Per-arc geometry: start/end coordinates, length (m) and bearing. Needs project_trip first."""
    u, v = trip.arcs["u"].to_numpy(), trip.arcs["v"].to_numpy()
    nx_, ny_ = trip.nodes["x"].to_numpy(), trip.nodes["y"].to_numpy()
    ax, ay, bx, by = nx_[u], ny_[u], nx_[v], ny_[v]
    return pd.DataFrame({
        "u": u, "v": v,
        "ax": ax, "ay": ay, "bx": bx, "by": by,
        "length_m": np.hypot(bx - ax, by - ay),
        "bearing": bearing_deg(bx - ax, by - ay),
    }, index=trip.arcs.index)


def point_to_segment(px, py, ax, ay, bx, by):
    """Distance from points P to segments AB (vectorised, broadcasting).

    Returns (dist_m, frac, qx, qy). frac in [0, 1] is how far along A->B the closest
    point Q lies (0 = at A, 1 = at B). Zero-length segments are treated as a point.
    """
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = ((px - ax) * dx + (py - ay) * dy) / len2
    frac = np.clip(np.nan_to_num(frac, nan=0.0), 0.0, 1.0)
    qx, qy = ax + frac * dx, ay + frac * dy
    return np.hypot(px - qx, py - qy), frac, qx, qy


def bearing_deg(dx, dy):
    """Compass bearing of a direction vector (0 = north, clockwise), in [0, 360)."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)


_GEOD = Geod(ellps="WGS84")


def geodesic_m(lon1, lat1, lon2, lat2):
    """Exact distance on the WGS84 ellipsoid, in metres. Used to sanity-check the projection.

    We use this rather than haversine because haversine assumes a spherical Earth and is
    off by up to ~0.5%, more than UTM's own error (<0.04%).
    """
    return _GEOD.inv(lon1, lat1, lon2, lat2)[2]
