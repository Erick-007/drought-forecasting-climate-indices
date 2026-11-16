import os
import socket

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

import drought_core as core

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "static")
FIRST_YEAR, LAST_YEAR = 2000, 2025
PORT = 5000

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")

for path in (core.METEO_CSV, core.CLIM_CSV):
    if not os.path.exists(path):
        raise SystemExit(f"Dataset file not found: {path}")

print("Loading datasets (this takes a few seconds)...")
DATA = core.load_data()
print("Loading trained models...")
MODELS = core.load_models()
TEST_METRICS = core.load_test_metrics()
print("Ready.")


def to_grid(arr):
    return [[None if not np.isfinite(v) else round(float(v), 4) for v in row] for row in arr]


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/info")
def info():
    rows = TEST_METRICS.to_dict(orient="records")
    return jsonify(first_year=FIRST_YEAR, last_year=LAST_YEAR, targets=core.TARGETS,
                   test_metrics=rows, land_cells=int(np.isfinite(DATA["Y"][-1, 0]).sum()))


@app.route("/api/forecast")
def forecast():
    try:
        year = int(request.args.get("year", LAST_YEAR))
        month = int(request.args.get("month", 1))
        lead = int(request.args.get("lead", 1))
        scale = request.args.get("scale", "sp")
    except ValueError:
        return jsonify(error="Invalid input."), 400
    if not (FIRST_YEAR <= year <= LAST_YEAR and 1 <= month <= 12 and lead in (1, 3)
            and scale in core.TARGETS):
        return jsonify(error="Invalid selection."), 400

    target_time = pd.Timestamp(year=year, month=month, day=1)
    if target_time not in DATA["times"]:
        return jsonify(error=f"{target_time.strftime('%B %Y')} is not in the dataset."), 404
    tau = int(DATA["times"].get_loc(target_time))
    end_time = DATA["times"][tau - lead]

    p1_all, p2_all = core.forecast(DATA, MODELS, tau, lead)
    k = core.TARGETS.index(scale)
    obs = DATA["Y"][tau, k]
    land = np.isfinite(obs)
    p1 = np.where(land, p1_all[k], np.nan)
    p2 = np.where(land, p2_all[k], np.nan)

    window = DATA["clim_df"].loc[:end_time].tail(core.T_CLIM)
    climate = {"months": [t.strftime("%Y-%m") for t in window.index],
               "series": {c: [round(float(v), 3) for v in window[c].values] for c in core.CLIM_INPUTS}}

    return jsonify(
        target=target_time.strftime("%B %Y"),
        inputs_until=end_time.strftime("%B %Y"),
        period="test period, unseen during training" if year > core.VAL_END_YEAR else "validation period",
        lead=lead,
        scale=scale,
        lats=[float(v) for v in DATA["lats"]],
        lons=[float(v) for v in DATA["lons"]],
        observed=to_grid(obs),
        model1=to_grid(p1),
        model2=to_grid(p2),
        scores={"model1": core.scores(p1_all[k], obs), "model2": core.scores(p2_all[k], obs)},
        climate=climate,
    )


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


if __name__ == "__main__":
    print(f"\nOpen on this laptop:          http://localhost:{PORT}")
    print(f"Open from another device (same Wi-Fi): http://{local_ip()}:{PORT}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)
