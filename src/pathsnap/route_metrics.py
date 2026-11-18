"""Route-level helpers and metrics for decoded paths (PLAN section 6, "after decoding").

Pure functions: nothing here reads files, and in particular nothing here loads the
reference route. Callers that are ALLOWED to see the route (the ablation script and the
API's server-side scorer) pass it in. The prediction pipeline only uses collapse() and
fill_route().

Definitions
  point top-1      share of GPS points whose decoded arc equals the labelled arc
                   (labels from labels.align_route). "conf" = points within 50 m of the
                   reference route (main), "all" = every point (secondary).
  continuity       share of consecutive DISTINCT decoded arcs that connect (a.v == b.u),
                   measured on the raw per-point decode, BEFORE gap filling (filling would
                   make it trivially 100%). Same definition as evaluate.route_continuity.
  route P/R/F1     the predicted route as a set of arcs vs the set of arcs in .route:
                   precision = |pred & ref| / |pred|, recall = |pred & ref| / |ref|.
                   Main numbers use the gap-filled route (what the map shows); the
                   unfilled ("raw") route is reported alongside. Length-weighted
                   versions weight every arc by its length in metres, so a missed 300 m
                   road counts more than a missed 5 m junction stub.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .graph import DEFAULT_CUTOFF_M, shortest_arc_path


def collapse(arcs) -> list[int]:
    """Drop consecutive repeats: per-point arcs -> route arc sequence."""
    a = np.asarray(arcs, dtype=np.int64)
    if len(a) == 0:
        return []
    return a[np.r_[True, a[1:] != a[:-1]]].tolist()


def fill_route(G: nx.DiGraph, route: list[int], cutoff: float = DEFAULT_CUTOFF_M) -> tuple[list[list[int]], int]:
    """Insert shortest-path arcs between consecutive route arcs that don't connect.

    Returns (pieces, n_unfilled): the route split into drivable pieces. A new piece starts
    only where no path exists within `cutoff` m (off-network episodes, long GPS gaps).
    """
    if not route:
        return [], 0
    pieces, cur, unfilled = [], [route[0]], 0
    for a, b in zip(route[:-1], route[1:]):
        mid = shortest_arc_path(G, a, b, cutoff=cutoff)
        if mid is None:
            pieces.append(cur)
            cur = [b]
            unfilled += 1
        else:
            cur.extend(mid)
            cur.append(b)
    pieces.append(cur)
    return pieces, unfilled


def continuity(point_arcs, u: np.ndarray, v: np.ndarray) -> tuple[int, int]:
    """(connected pairs, total pairs) over consecutive distinct arcs of a per-point decode."""
    r = np.asarray(collapse(point_arcs), dtype=np.int64)
    if len(r) < 2:
        return 0, 0
    return int((v[r[:-1]] == u[r[1:]]).sum()), len(r) - 1


def route_prf(pred_arcs, ref_arcs, length: np.ndarray | None = None) -> dict:
    """Set-based route precision/recall/F1 (and length-weighted versions if `length` given)."""
    pred, ref = set(int(a) for a in pred_arcs), set(int(a) for a in ref_arcs)
    inter = pred & ref

    def prf(w_inter, w_pred, w_ref):
        p = w_inter / w_pred if w_pred else 0.0
        r = w_inter / w_ref if w_ref else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    p, r, f = prf(len(inter), len(pred), len(ref))
    out = {"route_precision": p, "route_recall": r, "route_f1": f}
    if length is not None:
        w = lambda s: float(length[list(s)].sum()) if s else 0.0
        p, r, f = prf(w(inter), w(pred), w(ref))
        out.update(route_precision_len=p, route_recall_len=r, route_f1_len=f)
    return out


def point_accuracy(pred_arcs, label_arcs, low_conf) -> dict:
    """Point top-1 on confident points (main) and all points."""
    pred = np.asarray(pred_arcs, dtype=np.int64)
    lab = np.asarray(label_arcs, dtype=np.int64)
    conf = ~np.asarray(low_conf, dtype=bool)
    ok = pred == lab
    return {
        "top1_conf": float(ok[conf].mean()) if conf.any() else float("nan"),
        "top1_all": float(ok.mean()) if len(ok) else float("nan"),
        "n_points": int(len(ok)),
        "n_points_conf": int(conf.sum()),
    }


def score_decoded(G: nx.DiGraph, point_arcs, ref_route, label_arcs, low_conf, filled_route=None) -> dict:
    """Every route metric for one trip's decode (used by the ablation and the API scorer)."""
    u, v, length = G.graph["u"], G.graph["v"], G.graph["length"]
    raw = collapse(point_arcs)
    if filled_route is None:
        pieces, _ = fill_route(G, raw)
        filled_route = [a for piece in pieces for a in piece]
    ok, total = continuity(point_arcs, u, v)
    m = point_accuracy(point_arcs, label_arcs, low_conf)
    m.update(continuity=ok / total if total else float("nan"), continuity_pairs=total)
    m.update(route_prf(filled_route, ref_route, length))
    m.update({f"raw_{k}": val for k, val in route_prf(raw, ref_route, length).items()})
    return m
