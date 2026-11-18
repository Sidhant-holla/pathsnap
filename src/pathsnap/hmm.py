"""HMM map matching decoded with Viterbi.

Hidden state at GPS point t: one of that point's candidate arcs (we keep each
candidate's arc_id and its projected position proj_frac on the arc).

Emission, log scale:
  classifier  log e_i = lam * log(p_i / sum_j p_j)      p = classifier P(correct)
              Normalising per point turns independent probabilities into a distribution
              over that point's candidates; lam weighs emissions against transitions.
  distance    log e_i = -d_i^2 / (2 sigma^2)            the distance-only baseline:
              a Gaussian GPS error model, no classifier needed.

Transition from candidate a (point t-1) to candidate b (point t), log scale:
  log a_ab = -| d_network(a -> b) - d_straight(t-1, t) | / beta
  d_network is the driving distance between the two projected positions following
  arc directions (graph.distance_matrix), d_straight the distance between the two GPS
  fixes. A car really driving a -> b covers about as much road as the GPS moved.

Soft breaks: a move that is unreachable within the Dijkstra cutoff, or implies a speed
above v_max (with `speed_slack_m` of GPS noise allowed), gets log a = -break_penalty, a
large but FINITE value. Ordinary transitions are capped at the same value, so an
impossible move is never better than a possible one, but a single bad GPS fix can't
make every path -inf and kill the decode.

Gaps: where the time step exceeds gap_s seconds, the track is split into segments and
each one is decoded independently (no transition across the gap).

Missing candidates: a point with no candidate within the radius gets its nearest arc
(any distance) as its only state and is flagged `fallback`. So the decoder always
returns exactly one arc per GPS point.

Everything is in log space, so long trips (19k points) never underflow.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np
import pandas as pd

from .graph import BACKWARD_TOL_M, distance_matrix, nearest_arc

PROB_FLOOR = 1e-9


@dataclass
class HMMParams:
    emission: str = "classifier"   # "classifier" or "distance"
    lam: float = 1.0               # emission weight (classifier mode)
    sigma: float = 10.0            # GPS noise std in metres (distance mode)
    beta: float = 5.0              # transition scale in metres
    gap_s: float = 30.0            # split the track where dt > gap_s
    v_max: float = 50.0            # m/s (180 km/h); faster implied moves are soft breaks
    speed_slack_m: float = 30.0    # GPS noise allowed on top of v_max * dt
    break_penalty: float = 50.0    # log-penalty of a soft break (finite)
    backward_tol_m: float = BACKWARD_TOL_M

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "HMMParams":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def emission_logp(cands: pd.DataFrame, params: HMMParams) -> np.ndarray:
    """Per-candidate log emission (rows in the order of `cands`)."""
    if params.emission == "distance":
        d = cands["dist_m"].to_numpy(float)
        return -(d * d) / (2.0 * params.sigma ** 2)
    if params.emission == "classifier":
        p = np.clip(cands["proba"].to_numpy(float), PROB_FLOOR, None)
        total = pd.Series(p).groupby(cands["obs_idx"].to_numpy()).transform("sum").to_numpy()
        return params.lam * np.log(p / total)
    raise ValueError(f"unknown emission {params.emission!r}")


def transition_logp(dnet: np.ndarray, d_straight: float, dt: float, params: HMMParams) -> np.ndarray:
    """log a for a matrix of network distances; soft breaks at -break_penalty."""
    with np.errstate(invalid="ignore"):
        logp = -np.abs(dnet - d_straight) / params.beta
    impossible = ~np.isfinite(dnet) | (dnet > params.v_max * dt + params.speed_slack_m)
    logp = np.maximum(logp, -params.break_penalty)
    logp[impossible] = -params.break_penalty
    return logp


def _states(cands: pd.DataFrame, points: pd.DataFrame, G: nx.DiGraph, params: HMMParams):
    """Per GPS point: (arc_ids, proj_frac, log_emission, fallback flag)."""
    n = len(points)
    c = cands.sort_values(["obs_idx", "dist_m", "arc_id"], kind="stable").reset_index(drop=True)
    loge = emission_logp(c, params)
    obs = c["obs_idx"].to_numpy()
    starts = np.searchsorted(obs, np.arange(n), side="left")
    ends = np.searchsorted(obs, np.arange(n), side="right")
    arc = c["arc_id"].to_numpy(np.int64)
    frac = c["proj_frac"].to_numpy(float)

    missing = np.flatnonzero(ends == starts)
    fb = {}
    if len(missing):
        fa, fd, ff = nearest_arc(G, points["x"].to_numpy()[missing], points["y"].to_numpy()[missing])
        fb = {int(i): (a, f) for i, a, f in zip(missing, fa, ff)}

    states = []
    for i in range(n):
        s, e = starts[i], ends[i]
        if e > s:
            states.append((arc[s:e], frac[s:e], loge[s:e], False))
        else:
            a, f = fb[i]
            # Single forced state: its emission value is irrelevant to the argmax.
            states.append((np.array([a], np.int64), np.array([f]), np.zeros(1), True))
    return states


def viterbi(cands: pd.DataFrame, graph: nx.DiGraph, params: HMMParams | dict | None, points: pd.DataFrame) -> pd.DataFrame:
    """Most likely arc for every GPS point of ONE trip.

    cands   candidate rows: obs_idx, arc_id, dist_m, proj_frac (+ proba for classifier emissions)
    graph   graph.build_graph(trip)
    points  the trip's GPS points in order, one row per obs_idx 0..n-1: x, y, t (projected track)

    Returns one row per GPS point: obs_idx, arc_id, segment, fallback (no candidates, nearest
    arc used), soft_break (the transition INTO this point was a soft break), n_states.
    """
    params = params if isinstance(params, HMMParams) else HMMParams.from_dict(params)
    points = points.reset_index(drop=True)
    n = len(points)
    if n == 0:
        return pd.DataFrame(columns=["obs_idx", "arc_id", "segment", "fallback", "soft_break", "n_states"])
    x, y, t = (points[c].to_numpy(float) for c in ("x", "y", "t"))
    states = _states(cands, points, graph, params)

    dt_all = np.diff(t)
    segment = np.concatenate([[0], np.cumsum(dt_all > params.gap_s)])
    out_arc = np.empty(n, np.int64)
    soft = np.zeros(n, bool)

    for seg in np.unique(segment):
        idx = np.flatnonzero(segment == seg)
        arcs0, _, loge0, _ = states[idx[0]]
        delta = loge0.copy()
        back = []           # back[k][j] = best previous state for state j at idx[k+1]
        brk = []            # brk[k][j] = that best transition was a soft break
        for k in range(1, len(idx)):
            i_prev, i = idx[k - 1], idx[k]
            a_prev, f_prev, _, _ = states[i_prev]
            a_cur, f_cur, loge, _ = states[i]
            dt = max(t[i] - t[i_prev], 1e-6)
            d_str = float(np.hypot(x[i] - x[i_prev], y[i] - y[i_prev]))
            cutoff = min(params.v_max * dt + params.speed_slack_m, d_str + params.beta * params.break_penalty)
            cutoff = max(cutoff, d_str + params.speed_slack_m)
            dnet = distance_matrix(graph, a_prev, f_prev, a_cur, f_cur, cutoff=cutoff,
                                   backward_tol_m=params.backward_tol_m)
            loga = transition_logp(dnet, d_str, dt, params)
            score = delta[:, None] + loga             # (prev, cur)
            best = np.argmax(score, axis=0)
            cols = np.arange(len(a_cur))
            delta = score[best, cols] + loge
            back.append(best)
            brk.append(loga[best, cols] <= -params.break_penalty)
        # Backtrack.
        j = int(np.argmax(delta))
        for k in range(len(idx) - 1, -1, -1):
            out_arc[idx[k]] = states[idx[k]][0][j]
            if k > 0:
                soft[idx[k]] = bool(brk[k - 1][j])
                j = int(back[k - 1][j])

    return pd.DataFrame({
        "obs_idx": np.arange(n),
        "arc_id": out_arc,
        "segment": segment,
        "fallback": np.array([s[3] for s in states]),
        "soft_break": soft,
        "n_states": np.array([len(s[0]) for s in states]),
    })


def argmax_path(cands: pd.DataFrame, graph: nx.DiGraph, points: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Per-point best candidate WITHOUT the HMM (classifier-only / nearest-road baselines).

    score_col is maximised; ties go to the lower arc_id. Points with no candidates get the
    nearest arc, exactly as in viterbi(), so every method is compared on the same points.
    """
    n = len(points)
    best = (cands.sort_values(["obs_idx", score_col, "arc_id"], ascending=[True, False, True])
                 .drop_duplicates("obs_idx").set_index("obs_idx")["arc_id"].reindex(range(n)))
    fallback = best.isna().to_numpy()
    if fallback.any():
        miss = np.flatnonzero(fallback)
        fa, _, _ = nearest_arc(graph, points["x"].to_numpy()[miss], points["y"].to_numpy()[miss])
        best.iloc[miss] = fa
    return pd.DataFrame({"obs_idx": np.arange(n), "arc_id": best.to_numpy(np.int64), "fallback": fallback})
