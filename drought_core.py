import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "models")
METEO_CSV = "dataset1"
CLIM_CSV = "dataset2"

GRID_INPUTS = ["gridded data"]
PAST_SP = ["sp"]
TARGETS = ["sp"]
CLIM_INPUTS = ["climatic data"]
T_IN, T_CLIM = 12, 24
TRAIN_END_YEAR, VAL_END_YEAR = 1999, 2012
PAD_MULTIPLE = 4
EMBED, HIDDEN, KERNEL, CLIM_HIDDEN = 16, 64, 3, 8

class ConvLSTMCell(nn.Module):
    def __init__(self, cin, hid, k):
        super().__init__()
        # ConvLSTM model is here.....

    def forward(self, x, state):
        # ConvLSTM model is here.....
        pass


class ConvLSTMNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        # ConvLSTM model is here.....

    def forward(self, x):
        # ConvLSTM model is here.....
        pass


class ClimateBranch(nn.Module):
    def __init__(self, n_in, n_aux, combine_ch):
        super().__init__()
        # LSTM model is here.....
        # Combining module is here.....

    def forward(self, c):
        # LSTM model is here.....
        pass

    def combine_params(self, h):
        # Combining module is here.....
        pass


class ConvLSTMClimate(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        # ConvLSTM model is here.....
        # LSTM model is here.....
        # Combining module is here.....

    def forward(self, x, c):
        # ConvLSTM model, LSTM model and combining module are here.....
        pass


def normalise(arr, train_part, axes):
    m = np.nanmean(train_part, axis=axes, keepdims=True)
    s = np.nanstd(train_part, axis=axes, keepdims=True) + 1e-6
    return np.nan_to_num((arr - m) / s, nan=0.0).astype(np.float32)


def find_column(df, name):
    for c in df.columns:
        if c.strip().lower() == name.lower():
            return c
    raise KeyError(f"Column '{name}' not found. Available columns: {list(df.columns)}")


def load_data(meteo_csv=METEO_CSV, clim_csv=CLIM_CSV):
    df = pd.read_csv(meteo_csv, parse_dates=["time"])
    times = pd.DatetimeIndex(np.sort(df["time"].unique()))
    lats = np.sort(df["latitude"].unique())[::-1]
    lons = np.sort(df["longitude"].unique())
    T, H, W = len(times), len(lats), len(lons)
    ti = np.searchsorted(times.values, df["time"].values)
    hi = np.searchsorted(-lats, -df["latitude"].values)
    wi = np.searchsorted(lons, df["longitude"].values)

    def to_cube(cols):
        cube = np.full((T, len(cols), H, W), np.nan, dtype=np.float32)
        for k, c in enumerate(cols):
            cube[ti, k, hi, wi] = df[find_column(df, c)].values.astype(np.float32)
        return cube

    X = to_cube(GRID_INPUTS)
    X = np.concatenate([X, to_cube(PAST_SP)], axis=1)
    Y = to_cube(TARGETS)

    train_months = np.where(times.year.values <= TRAIN_END_YEAR)[0]
    Xn = normalise(X, X[train_months], axes=(0, 2, 3))

    cdf = pd.read_csv(clim_csv, parse_dates=["time"]).sort_values("time").set_index("time")
    cols = [find_column(cdf, c) for c in CLIM_INPUTS]
    clim = cdf[cols].apply(pd.to_numeric, errors="coerce")
    clim = clim.mask(clim < -50).ffill().reindex(times).ffill()
    C = clim.values.astype(np.float32)
    Cn = normalise(C, C[train_months], axes=0)

    Hp = int(np.ceil(H / PAD_MULTIPLE) * PAD_MULTIPLE)
    Wp = int(np.ceil(W / PAD_MULTIPLE) * PAD_MULTIPLE)
    Xn = np.pad(Xn, [(0, 0), (0, 0), (0, Hp - H), (0, Wp - W)])
    return dict(times=times, lats=lats, lons=lons, Xn=Xn, Y=Y, C=C, Cn=Cn, H=H, W=W,
                clim_df=clim.set_axis(CLIM_INPUTS, axis=1))


def load_models(model_dir=MODEL_DIR):
    models = {}
    for lead in (1, 3):
        m1 = ConvLSTMNet(len(GRID_INPUTS) + len(PAST_SP), len(TARGETS))
        m1.load_state_dict(torch.load(os.path.join(model_dir, f"model1_lead{lead}.pt"),
                                      map_location="cpu", weights_only=True))
        m2 = ConvLSTMClimate(len(GRID_INPUTS) + len(PAST_SP), len(TARGETS))
        m2.load_state_dict(torch.load(os.path.join(model_dir, f"model2_lead{lead}.pt"),
                                      map_location="cpu", weights_only=True))
        models[(1, lead)], models[(2, lead)] = m1.eval(), m2.eval()
    return models


def load_test_metrics():
    return pd.read_csv(os.path.join(MODEL_DIR, "test_metrics_2013_2025.csv"))


@torch.no_grad()
def forecast(data, models, tau, lead):
    end = tau - lead
    x = torch.from_numpy(data["Xn"][end - T_IN + 1:end + 1])[None]
    c = torch.from_numpy(data["Cn"][end - T_CLIM + 1:end + 1])[None]
    H, W = data["H"], data["W"]
    p1 = models[(1, lead)](x)[0, :, :H, :W].numpy()
    p2 = models[(2, lead)](x, c)[0, :, :H, :W].numpy()
    return p1, p2


def scores(pred, obs):
    m = np.isfinite(obs)
    p, o = pred[m], obs[m]
    err = p - o
    sst = ((o - o.mean()) ** 2).sum()
    return dict(cells=int(m.sum()), MSE=float((err ** 2).mean()), MAE=float(np.abs(err).mean()),
                R2=float(1 - (err ** 2).sum() / sst) if sst > 0 else float("nan"))
