"""Crash-only smoke test of a running API: does every served trip come back with HTTP 200?

Run (API already running, from either laptop):
    python -m scripts.smoke_api --url http://<laptop-1-ip>:8000

It sends {"score": false}, so the API skips scoring entirely: no metrics are computed,
returned or written to the API log, and the reference route is not even read. On the
evaluator trips we may check that nothing crashes, but we must not see or tune on their
scores. It prints only the status code, timing and response size per trip.
"""
import argparse
import sys
import time

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--timeout", type=float, default=300)
    a = ap.parse_args()
    url = a.url.rstrip("/")

    trips = requests.get(f"{url}/trips", timeout=30).json()
    ids = trips["trip_ids"]
    print(f"{url}: partition={trips['partition']} trips={len(ids)}")
    failures = 0
    for tid in ids:
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{url}/predict", json={"trip_id": tid, "score": False}, timeout=a.timeout)
            ok = r.status_code == 200
            body = r.json() if ok else {}
            extra = (f"points={body['n_points']} route_pieces={len(body['route_coords'])} "
                     f"scored={body['metrics'].get('available')}") if ok else r.text[:200]
            status = r.status_code
        except Exception as e:                     # connection error, timeout, bad JSON
            ok, status, extra = False, type(e).__name__, str(e)[:200]
        failures += not ok
        print(f"  {tid}: {status} in {time.perf_counter() - t0:.1f}s  {extra}")
    print("ALL OK" if failures == 0 else f"{failures} FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
