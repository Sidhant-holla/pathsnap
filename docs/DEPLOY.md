# PathSnap: two-laptop deployment

Laptop 1 runs the API (model, HMM/Viterbi, road graphs, server-side scoring).
Laptop 2 runs the UI. Both must be on the **same network** (one phone hotspot).

Both laptops: clone, `python -m venv .venv`, `.venv\Scripts\activate`,
`pip install -r requirements.txt`, `pip install -e .`. Laptop 2 needs no data and no model.

## Laptop 1: API

1. Put the data where the API can see it. The evaluator trips go in a partition folder next to the
   others, e.g. `data/raw/evaluator_test/<trip_id>/...` (also accepted: `evaluator`, `eval`, `test`, ...),
   or anywhere else with `PATHSNAP_DATA` pointing at the folder that contains the partitions.
2. Allow port 8000 through the Windows firewall. Run **once** in an **Administrator** PowerShell
   (hotspots usually count as a *Public* network, so the rule covers both profiles):

   ```powershell
   New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private,Public
   ```

   Remove it after the event with `Remove-NetFirewallRule -DisplayName "PathSnap API 8000"`.
3. Find Laptop 1's IP: `ipconfig`, then the **IPv4 Address** of the Wi-Fi adapter (e.g. `192.168.43.17`).
4. Start the API from the repo root and keep this window visible (the evaluators read these logs):

   ```powershell
   $env:PATHSNAP_PARTITION = "evaluator_test"     # default; "student_test" for rehearsals
   # $env:PATHSNAP_DATA = "D:\path\to\folder\with\partitions"   # only if the data lives elsewhere
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

   Startup prints `PathSnap API ready | model=lgbm | partition=... | 10 trips`. Every request then
   prints one line with the trip id, status and time, plus a per-stage timing line for `/predict`.

## Laptop 2: UI

```powershell
$env:API_URL = "http://192.168.43.17:8000"     # Laptop 1's IP from step 3
streamlit run ui/app.py
```

Open the URL Streamlit prints (http://localhost:8501), pick a trip, press **Predict route**.

## Checks

- From Laptop 2, open `http://<laptop-1-ip>:8000/health` in a browser. You should see JSON with `"status": "ok"`.
  If it times out: wrong IP, different networks, or the firewall rule is missing.
- Crash-only check of every served trip (prints status/timing only, **no metrics**, so it is safe to run
  on the evaluator trips):

  ```powershell
  python -m scripts.smoke_api --url http://<laptop-1-ip>:8000
  ```

- Rehearsal without evaluator data: start the API with `$env:PATHSNAP_PARTITION = "student_test"`.

## What the API returns

`POST /predict {"trip_id": "..."}` returns the GPS points, the **predicted** route geometry
(`route_coords`), run flags, timings and metrics. The prediction (`pathsnap.pipeline.predict_trip`)
never reads the `.route` file. Afterwards `api/scorer.py`, the only API module that reads it,
computes the metrics. The reference arcs are never put in the response.
