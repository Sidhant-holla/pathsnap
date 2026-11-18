"""Reading raw trip files.

Raw format (verified on all 90 dev trips). There are no headers, values are
tab/whitespace separated, and .track files have Windows line endings:

    <id>.track  lon  lat  t        one GPS fix per row, t in seconds (~1 Hz, with gaps)
    <id>.nodes  lon  lat           row index = node_id
    <id>.arcs   u    v             directed arc u -> v, row index = arc_id
    <id>.route  arc_id             arcs actually driven, in order (reference labels)

Each trip has its OWN road network, so an arc_id only means something within its trip.

Leakage rule: load_trip() does NOT read the .route file unless asked
(with_route=True). Prediction code must always use the default.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DATA_ROOT, PARTITIONS


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame        # columns: lon, lat, t   (+ x, y after geo.project_trip)
    nodes: pd.DataFrame        # index: node_id; columns: lon, lat   (+ x, y)
    arcs: pd.DataFrame         # index: arc_id;  columns: u, v  (directed u -> v)
    route: list[int] | None = None   # reference arc_ids, only when with_route=True
    crs: str | None = None           # projected CRS, set by geo.project_trip


def _read(path: Path, names: list[str], dtype) -> pd.DataFrame:
    # sep=r"\s+" handles tabs, spaces and the \r\n line endings in .track files.
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype)


def list_trips(partition: str, data_root: Path = DATA_ROOT) -> list[str]:
    """Sorted trip IDs in a partition (folder names, e.g. '00000001')."""
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition {partition!r}; expected one of {PARTITIONS}")
    folder = Path(data_root) / partition
    if not folder.exists():
        return []
    return sorted(p.name for p in folder.iterdir() if p.is_dir())


def find_partition(trip_id: str, data_root: Path = DATA_ROOT) -> str:
    """Which partition folder a trip lives in (trip IDs are unique across partitions)."""
    for partition in PARTITIONS:
        if (Path(data_root) / partition / trip_id).is_dir():
            return partition
    raise FileNotFoundError(f"trip {trip_id} not found under {data_root}")


def trip_dir(trip_id: str, partition: str | None = None, data_root: Path = DATA_ROOT) -> Path:
    partition = partition or find_partition(trip_id, data_root)
    return Path(data_root) / partition / trip_id


def load_route(trip_id: str, partition: str | None = None, data_root: Path = DATA_ROOT) -> list[int]:
    """Reference route (ground truth). Only evaluation/scoring code should call this."""
    path = trip_dir(trip_id, partition, data_root) / f"{trip_id}.route"
    return _read(path, ["arc_id"], "int64")["arc_id"].tolist()


def load_trip(trip_id: str, partition: str | None = None, with_route: bool = False,
              data_root: Path = DATA_ROOT) -> Trip:
    """Load one trip's GPS track and road network (and the route only if with_route=True)."""
    partition = partition or find_partition(trip_id, data_root)
    d = trip_dir(trip_id, partition, data_root)

    track = _read(d / f"{trip_id}.track", ["lon", "lat", "t"], "float64")
    nodes = _read(d / f"{trip_id}.nodes", ["lon", "lat"], "float64")
    arcs = _read(d / f"{trip_id}.arcs", ["u", "v"], "int64")
    nodes.index.name = "node_id"
    arcs.index.name = "arc_id"

    route = load_route(trip_id, partition, data_root) if with_route else None
    return Trip(trip_id, partition, track, nodes, arcs, route)
