RUN_NOCLIMATE = True
RUN_CLIMATE = True
COMPARE_DIR = "results_convlstm_comparison_lead1"
COMPARE_DIRS = {1: "results_convlstm_comparison_lead1",
                3: "results_convlstm_comparison_lead3"}


import os
import json
import math
import time
import random
import warnings
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class Config:
    grid_csv: str = "dataset1"
    out_dir: str = "results_convlstm_noclimate_lead1"

    grid_inputs: tuple = ("gridded data",)
    grid_targets: tuple = ("sp",)
    use_past_sp: bool = True
    past_sp: tuple = ("sp",)
    t_in: int = 12
    t_align: int = 24
    leads: tuple = (1,)
    train_end_year: int = 1999
    val_end_year: int = 2012
    pad_multiple: int = 4

    convlstm_embed: int = 16
    convlstm_hidden: int = 64
    convlstm_kernel: int = 3
    convlstm_layers: int = 1
    grid_dropout: float = 0.1

    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 16
    seeds: tuple = (123,)
    drought_threshold: float = -1.0


CFG = Config()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_grid(cfg):
    df = pd.read_csv(cfg.grid_csv, parse_dates=["time"])
    sp_in = cfg.past_sp if cfg.use_past_sp else ()
    missing = [c for c in ("time", "latitude", "longitude", *cfg.grid_inputs, *cfg.grid_targets,
                           *sp_in) if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset 1 is missing columns: {missing}")
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
            cube[ti, k, hi, wi] = df[c].values.astype(np.float32)
        return cube

    X, Y = to_cube(cfg.grid_inputs), to_cube(cfg.grid_targets)
    if sp_in:
        X = np.concatenate([X, to_cube(sp_in)], axis=1)
    log(f"Grid inputs ({X.shape[1]} channels): {list(cfg.grid_inputs) + list(sp_in)}")
    log(f"Dataset 1: {T} months ({times[0].date()} -> {times[-1].date()}), grid {H} x {W}, "
        f"{int(np.isfinite(Y[:, 0]).any(axis=0).sum())} land cells")
    return times, lats, lons, X, Y


def build_samples(cfg, times, Y, lead):
    years = times.year.values
    samples = {"train": [], "val": [], "test": []}
    for tau in range(max(cfg.t_in, cfg.t_align) - 1 + lead, len(times)):
        if not np.isfinite(Y[tau]).any():
            continue
        y = years[tau]
        split = "train" if y <= cfg.train_end_year else ("val" if y <= cfg.val_end_year else "test")
        samples[split].append(tau)
    for k, v in samples.items():
        log(f"{k:5s}: {len(v)} samples, targets {times[v[0]].date()} -> {times[v[-1]].date()}")
    return samples


def normalise(arr, train_part, axes):
    m = np.nanmean(train_part, axis=axes, keepdims=True)
    s = np.nanstd(train_part, axis=axes, keepdims=True) + 1e-6
    return np.nan_to_num((arr - m) / s, nan=0.0).astype(np.float32)


class DroughtDataset(Dataset):
    def __init__(self, idx, Xn, Y, cfg, pad_hw, lead):
        self.idx, self.Xn, self.Y, self.cfg = idx, Xn, Y, cfg
        self.ph, self.pw = pad_hw
        self.lead = lead

    def __len__(self):
        return len(self.idx)

    def _pad(self, arr, value):
        return np.pad(arr, [(0, 0)] * (arr.ndim - 2) + [(0, self.ph), (0, self.pw)],
                      constant_values=value)

    def __getitem__(self, i):
        tau = self.idx[i]
        end = tau - self.lead
        x = self._pad(self.Xn[end - self.cfg.t_in + 1:end + 1], 0.0)
        y = self._pad(self.Y[tau], np.nan)
        m = np.isfinite(y)
        return (torch.from_numpy(x), torch.from_numpy(np.nan_to_num(y)),
                torch.from_numpy(m), torch.tensor(tau))


class ConvLSTMCell(nn.Module):
    def __init__(self, cin, hid, k):
        super().__init__()
        # ConvLSTM model is here.....

    def forward(self, x, state):
        # ConvLSTM model is here.....
        pass


class ConvLSTMNet(nn.Module):

    def __init__(self, cfg, n_in, n_out):
        super().__init__()
        # ConvLSTM model is here.....

    def forward(self, x):
        # ConvLSTM model is here.....
        pass


def masked_l1l2(pred, y, m):
    m = m.float()
    d = (pred - y) * m
    return (d.pow(2).sum() + d.abs().sum()) / m.sum().clamp_min(1.0)


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tot, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y, m, _ in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            loss = masked_l1l2(model(x), y, m)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
    return tot / n


def train_model(model, loaders, device, cfg):
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, best_ep, best_state, wait, history = math.inf, 0, None, 0, []
    for ep in range(1, cfg.max_epochs + 1):
        tr = run_epoch(model, loaders["train"], device, opt)
        va = run_epoch(model, loaders["val"], device)
        history.append({"epoch": ep, "train_main": tr, "val_main": va})
        improved = va < best - 1e-5
        if improved:
            best, best_ep, wait = va, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if ep == 1 or ep % 5 == 0 or improved:
            log(f"  ep {ep:3d} | train main {tr:.4f} | val main {va:.4f}{' *' if improved else ''}")
        if wait >= cfg.patience:
            log(f"  early stop at epoch {ep}")
            break
    model.load_state_dict(best_state)
    log(f"  restored best weights from epoch {best_ep} (val main {best:.4f})")
    return pd.DataFrame(history)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P, O, M, TAU = [], [], [], []
    for x, y, m, tau in loader:
        P.append(model(x.to(device)).cpu().numpy()); O.append(y.numpy()); M.append(m.numpy())
        TAU.append(tau.numpy())
    return np.concatenate(P), np.concatenate(O), np.concatenate(M), np.concatenate(TAU)


def metrics_for(pred, obs, mask, names, thr):
    rows = []
    for k, name in enumerate(names):
        mk = mask[:, k]
        p, o = pred[:, k][mk], obs[:, k][mk]
        err = p - o
        sst = ((o - o.mean()) ** 2).sum()
        ev_o, ev_p = o <= thr, p <= thr
        hits = (ev_o & ev_p).sum(); miss = (ev_o & ~ev_p).sum(); fa = (~ev_o & ev_p).sum()
        rows.append({
            "target": name,
            "MSE": float((err ** 2).mean()),
            "RMSE": float(np.sqrt((err ** 2).mean())),
            "MAE": float(np.abs(err).mean()),
            "R2": float(1 - (err ** 2).sum() / sst) if sst > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def climatology(Y, times, train_idx):
    months, tr = times.month.values, np.array(train_idx)
    clim = np.zeros((12,) + Y.shape[1:], dtype=np.float32)
    for mo in range(1, 13):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            clim[mo - 1] = np.nan_to_num(np.nanmean(Y[tr[months[tr] == mo]], axis=0))
    return clim


MODEL_LABEL = "ConvLSTM (no climate)"


def run_lead(cfg, lead, times, lats, lons, Xn, Y, device):
    out = os.path.join(cfg.out_dir, f"lead{lead}")
    os.makedirs(out, exist_ok=True)
    log(f"########## lead {lead} ##########")
    samples = build_samples(cfg, times, Y, lead)
    H, W = Y.shape[-2:]
    Hp, Wp = (math.ceil(H / cfg.pad_multiple) * cfg.pad_multiple,
              math.ceil(W / cfg.pad_multiple) * cfg.pad_multiple)
    ds = {k: DroughtDataset(v, Xn, Y, cfg, (Hp - H, Wp - W), lead) for k, v in samples.items()}
    names = list(cfg.grid_targets)
    clim_map = climatology(Y, times, samples["train"])
    rows = []
    for seed in cfg.seeds:
        log(f"===== lead {lead} | seed {seed} =====")
        set_seed(seed)
        g = torch.Generator().manual_seed(seed)
        loaders = {"train": DataLoader(ds["train"], cfg.batch_size, shuffle=True, generator=g),
                   "val": DataLoader(ds["val"], cfg.batch_size),
                   "test": DataLoader(ds["test"], cfg.batch_size)}
        model = ConvLSTMNet(cfg, Xn.shape[1], len(names)).to(device)
        if seed == cfg.seeds[0] and lead == cfg.leads[0]:
            log(f"Parameters: {sum(p.numel() for p in model.parameters()):,} | padded grid {Hp} x {Wp}")
        hist = train_model(model, loaders, device, cfg)
        hist.to_csv(os.path.join(out, f"history_seed{seed}.csv"), index=False)
        torch.save(model.state_dict(), os.path.join(out, f"model_seed{seed}.pt"))
        pred, obs, mask, tau = predict(model, loaders["test"], device)
        sl = (slice(None), slice(None), slice(0, H), slice(0, W))
        pred, obs, mask = pred[sl], obs[sl], mask[sl]
        np.savez_compressed(os.path.join(out, f"test_predictions_seed{seed}.npz"),
                            pred=pred, obs=obs, mask=mask, target_time=times[tau].astype(str),
                            lat=lats, lon=lons, targets=np.array(names))
        df = metrics_for(pred, obs, mask, names, cfg.drought_threshold)
        df.insert(0, "model", MODEL_LABEL); df.insert(0, "lead", lead); df.insert(2, "seed", seed)
        rows.append(df)
        log("\n" + df.round(4).to_string(index=False))
        if seed == cfg.seeds[0]:
            b = metrics_for(clim_map[times[tau].month.values - 1], obs, mask, names, cfg.drought_threshold)
            b.insert(0, "model", "Climatology"); b.insert(0, "lead", lead); b.insert(2, "seed", -1)
            rows.append(b)
    return pd.concat(rows, ignore_index=True)


def main(cfg=CFG):
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert all(l >= 1 for l in cfg.leads) and cfg.t_align >= cfg.t_in
    log(f"{MODEL_LABEL} | leads {cfg.leads} | device {device}")
    log(f"Grid regularisation: dropout {cfg.grid_dropout}, weight decay {cfg.weight_decay} | "
        f"no climate indices, no auxiliary loss")
    times, lats, lons, X, Y = load_grid(cfg)
    train_months = np.where(times.year.values <= cfg.train_end_year)[0]
    Xn = normalise(X, X[train_months], axes=(0, 2, 3))
    res = pd.concat([run_lead(cfg, lead, times, lats, lons, Xn, Y, device) for lead in cfg.leads],
                    ignore_index=True)
    res.to_csv(os.path.join(cfg.out_dir, "metrics_per_seed.csv"), index=False)
    cols = ["MSE", "RMSE", "MAE", "R2"]
    summary = res.groupby(["lead", "model", "target"], sort=False)[cols].agg(["mean", "std"])
    summary.to_csv(os.path.join(cfg.out_dir, "metrics_summary.csv"))
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    log("Test set, mean and std over seeds (climatology has no std):")
    print(summary.round(4).to_string())


from dataclasses import replace
NOCLIM_CFG = CFG
NOCLIM_CFGS = {1: NOCLIM_CFG,
               3: replace(NOCLIM_CFG, leads=(3,), out_dir="results_convlstm_noclimate_lead3")}
if __name__ == "__main__" and RUN_NOCLIMATE:
    for _lead in (1, 3):
        main(NOCLIM_CFGS[_lead])


import os
import json
import math
import time
import random
import warnings
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class Config:
    grid_csv: str = "dataset1"
    clim_csv: str = "dataset2"
    out_dir: str = "results_convlstm_climate_lead1"

    grid_inputs: tuple = ("gridded data",)
    grid_targets: tuple = ("sp",)
    use_past_sp: bool = True
    past_sp: tuple = ("sp",)
    clim_inputs: tuple = ("climatic data",)
    clim_targets: tuple = ("sp_mean",)
    t_in: int = 12
    t_clim: int = 24
    leads: tuple = (1,)
    train_end_year: int = 1999
    val_end_year: int = 2012
    pad_multiple: int = 4

    convlstm_embed: int = 16
    convlstm_hidden: int = 64
    convlstm_kernel: int = 3
    convlstm_layers: int = 1
    grid_dropout: float = 0.1

    clim_hidden: int = 8
    clim_dropout: float = 0.3
    clim_weight_decay: float = 1e-4
    combine_at: str = "input"
    aux_weight: float = 0.0
    clim_placebo: str = "none"
    placebo_seed: int = 0

    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 16
    seeds: tuple = (123,)
    drought_threshold: float = -1.0


CFG = Config()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_grid(cfg):
    df = pd.read_csv(cfg.grid_csv, parse_dates=["time"])
    sp_in = cfg.past_sp if cfg.use_past_sp else ()
    missing = [c for c in ("time", "latitude", "longitude", *cfg.grid_inputs, *cfg.grid_targets,
                           *sp_in) if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset 1 is missing columns: {missing}")
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
            cube[ti, k, hi, wi] = df[c].values.astype(np.float32)
        return cube

    X, Y = to_cube(cfg.grid_inputs), to_cube(cfg.grid_targets)
    if sp_in:
        X = np.concatenate([X, to_cube(sp_in)], axis=1)
    log(f"Grid inputs ({X.shape[1]} channels): {list(cfg.grid_inputs) + list(sp_in)}")
    log(f"Dataset 1: {T} months ({times[0].date()} -> {times[-1].date()}), grid {H} x {W}, "
        f"{int(np.isfinite(Y[:, 0]).any(axis=0).sum())} land cells")
    return times, lats, lons, X, Y


def load_climate(cfg, times):
    df = pd.read_csv(cfg.clim_csv, parse_dates=["time"]).sort_values("time").set_index("time")
    missing = [c for c in (*cfg.clim_inputs, *cfg.clim_targets) if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset 2 is missing columns: {missing}")
    for c in cfg.clim_inputs:
        bad = df[c] < -50
        if bad.any():
            log(f"{c}: {int(bad.sum())} missing months forward-filled")
            df.loc[bad, c] = np.nan
    df[list(cfg.clim_inputs)] = df[list(cfg.clim_inputs)].ffill()
    df = df.reindex(times)
    if df[list(cfg.clim_inputs)].isna().any().any():
        raise ValueError("Dataset 2 does not cover every month of dataset 1")
    C = df[list(cfg.clim_inputs)].values.astype(np.float32)
    A = df[list(cfg.clim_targets)].values.astype(np.float32)
    if cfg.clim_placebo == "shuffle_years":
        years = times.year.values
        uniq = np.unique(years)
        rng = np.random.default_rng(cfg.placebo_seed)
        perm = rng.permutation(uniq)
        while np.any(perm == uniq):
            perm = rng.permutation(uniq)
        src = dict(zip(uniq, perm))
        C2 = np.empty_like(C)
        for yr in uniq:
            dst_idx = np.where(years == yr)[0]
            src_idx = np.where(years == src[yr])[0]
            n = min(len(dst_idx), len(src_idx))
            C2[dst_idx[:n]] = C[src_idx[:n]]
            if n < len(dst_idx):
                C2[dst_idx[n:]] = C[src_idx[-1]]
        C = C2
        log("PLACEBO: climate indices shuffled between years (link to sp broken)")
    elif cfg.clim_placebo != "none":
        raise ValueError(f"clim_placebo must be none or shuffle_years, not {cfg.clim_placebo!r}")
    log(f"Dataset 2: inputs {list(cfg.clim_inputs)}, auxiliary targets {list(cfg.clim_targets)}")
    return C, A


def build_samples(cfg, times, Y, lead):
    years = times.year.values
    samples = {"train": [], "val": [], "test": []}
    for tau in range(max(cfg.t_in, cfg.t_clim) - 1 + lead, len(times)):
        if not np.isfinite(Y[tau]).any():
            continue
        y = years[tau]
        split = "train" if y <= cfg.train_end_year else ("val" if y <= cfg.val_end_year else "test")
        samples[split].append(tau)
    for k, v in samples.items():
        log(f"{k:5s}: {len(v)} samples, targets {times[v[0]].date()} -> {times[v[-1]].date()}")
    return samples


def normalise(arr, train_part, axes):
    m = np.nanmean(train_part, axis=axes, keepdims=True)
    s = np.nanstd(train_part, axis=axes, keepdims=True) + 1e-6
    return np.nan_to_num((arr - m) / s, nan=0.0).astype(np.float32)


class DroughtDataset(Dataset):
    def __init__(self, idx, Xn, Y, Cn, A, cfg, pad_hw, lead):
        self.idx, self.Xn, self.Y, self.Cn, self.A, self.cfg = idx, Xn, Y, Cn, A, cfg
        self.ph, self.pw = pad_hw
        self.lead = lead

    def __len__(self):
        return len(self.idx)

    def _pad(self, arr, value):
        return np.pad(arr, [(0, 0)] * (arr.ndim - 2) + [(0, self.ph), (0, self.pw)],
                      constant_values=value)

    def __getitem__(self, i):
        tau = self.idx[i]
        end = tau - self.lead
        x = self._pad(self.Xn[end - self.cfg.t_in + 1:end + 1], 0.0)
        y = self._pad(self.Y[tau], np.nan)
        m = np.isfinite(y)
        c = self.Cn[end - self.cfg.t_clim + 1:end + 1]
        a = self.A[tau]
        return (torch.from_numpy(x), torch.from_numpy(c), torch.from_numpy(np.nan_to_num(y)),
                torch.from_numpy(m), torch.from_numpy(np.nan_to_num(a)),
                torch.from_numpy(np.isfinite(a)), torch.tensor(tau))


class ConvLSTMCell(nn.Module):
    def __init__(self, cin, hid, k):
        super().__init__()
        # ConvLSTM model is here.....

    def forward(self, x, state):
        # ConvLSTM model is here.....
        pass


class ClimateBranch(nn.Module):

    def __init__(self, cfg, n_in, n_aux, combine_ch):
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

    def __init__(self, cfg, n_in, n_out):
        super().__init__()
        # ConvLSTM model is here.....
        # Combining module is here.....

    def forward(self, x, c, return_combine=False):
        # ConvLSTM model is here.....
        # Combining module is here.....
        pass


def masked_mse(pred, y, m):
    m = m.float()
    return ((pred - y) * m).pow(2).sum() / m.sum().clamp_min(1.0)


def masked_l1l2(pred, y, m):
    m = m.float()
    d = (pred - y) * m
    return (d.pow(2).sum() + d.abs().sum()) / m.sum().clamp_min(1.0)


def run_epoch(model, loader, device, cfg, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tot_main = tot_aux = n = 0.0
    with torch.set_grad_enabled(train):
        for x, c, y, m, a, am, _ in loader:
            x, c, y, m, a, am = (t.to(device) for t in (x, c, y, m, a, am))
            pred, aux = model(x, c)
            loss_main = masked_l1l2(pred, y, m)
            loss_aux = masked_mse(aux, a, am)
            loss = loss_main + cfg.aux_weight * loss_aux
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            b = x.size(0)
            tot_main += loss_main.item() * b
            tot_aux += loss_aux.item() * b
            n += b
    return tot_main / n, tot_aux / n


def train_model(model, loaders, device, cfg):
    clim = [p for n, p in model.named_parameters() if n.startswith("climate.")]
    grid = [p for n, p in model.named_parameters() if not n.startswith("climate.")]
    opt = torch.optim.Adam([{"params": grid, "weight_decay": cfg.weight_decay},
                            {"params": clim, "weight_decay": cfg.clim_weight_decay}], lr=cfg.lr)
    best, best_ep, best_state, wait, history = math.inf, 0, None, 0, []
    for ep in range(1, cfg.max_epochs + 1):
        tr = run_epoch(model, loaders["train"], device, cfg, opt)
        va = run_epoch(model, loaders["val"], device, cfg)
        history.append({"epoch": ep, "train_main": tr[0], "train_aux": tr[1],
                        "val_main": va[0], "val_aux": va[1]})
        improved = va[0] < best - 1e-5
        if improved:
            best, best_ep, wait = va[0], ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if ep == 1 or ep % 5 == 0 or improved:
            log(f"  ep {ep:3d} | train main {tr[0]:.4f} aux {tr[1]:.4f} | "
                f"val main {va[0]:.4f} aux {va[1]:.4f}{' *' if improved else ''}")
        if wait >= cfg.patience:
            log(f"  early stop at epoch {ep}")
            break
    model.load_state_dict(best_state)
    log(f"  restored best weights from epoch {best_ep} (val main {best:.4f})")
    return pd.DataFrame(history)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P, O, M, TAU, S = [], [], [], [], []
    for x, c, y, m, _, _, tau in loader:
        pred, _, strength = model(x.to(device), c.to(device), return_combine=True)
        P.append(pred.cpu().numpy()); O.append(y.numpy()); M.append(m.numpy())
        TAU.append(tau.numpy()); S.append(strength.cpu().numpy())
    return (np.concatenate(P), np.concatenate(O), np.concatenate(M), np.concatenate(TAU),
            np.concatenate(S))


def metrics_for(pred, obs, mask, names, thr):
    rows = []
    for k, name in enumerate(names):
        mk = mask[:, k]
        p, o = pred[:, k][mk], obs[:, k][mk]
        err = p - o
        sst = ((o - o.mean()) ** 2).sum()
        ev_o, ev_p = o <= thr, p <= thr
        hits = (ev_o & ev_p).sum(); miss = (ev_o & ~ev_p).sum(); fa = (~ev_o & ev_p).sum()
        rows.append({
            "target": name,
            "MSE": float((err ** 2).mean()),
            "RMSE": float(np.sqrt((err ** 2).mean())),
            "MAE": float(np.abs(err).mean()),
            "R2": float(1 - (err ** 2).sum() / sst) if sst > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def climatology(Y, times, train_idx):
    months, tr = times.month.values, np.array(train_idx)
    clim = np.zeros((12,) + Y.shape[1:], dtype=np.float32)
    for mo in range(1, 13):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            clim[mo - 1] = np.nan_to_num(np.nanmean(Y[tr[months[tr] == mo]], axis=0))
    return clim


def model_label(cfg):
    lab = f"ConvLSTM + climate (combining-{cfg.combine_at})"
    return lab + (" PLACEBO" if cfg.clim_placebo != "none" else "")


def run_lead(cfg, lead, times, lats, lons, Xn, Y, Cn, A, device):
    out = os.path.join(cfg.out_dir, f"lead{lead}")
    os.makedirs(out, exist_ok=True)
    log(f"########## lead {lead} ##########")
    samples = build_samples(cfg, times, Y, lead)
    H, W = Y.shape[-2:]
    Hp, Wp = (math.ceil(H / cfg.pad_multiple) * cfg.pad_multiple,
              math.ceil(W / cfg.pad_multiple) * cfg.pad_multiple)
    ds = {k: DroughtDataset(v, Xn, Y, Cn, A, cfg, (Hp - H, Wp - W), lead) for k, v in samples.items()}
    names = list(cfg.grid_targets)
    clim_map = climatology(Y, times, samples["train"])
    rows, combine_rows = [], []
    for seed in cfg.seeds:
        log(f"===== lead {lead} | seed {seed} =====")
        set_seed(seed)
        g = torch.Generator().manual_seed(seed)
        loaders = {"train": DataLoader(ds["train"], cfg.batch_size, shuffle=True, generator=g),
                   "val": DataLoader(ds["val"], cfg.batch_size),
                   "test": DataLoader(ds["test"], cfg.batch_size)}
        model = ConvLSTMClimate(cfg, Xn.shape[1], len(names)).to(device)
        if seed == cfg.seeds[0] and lead == cfg.leads[0]:
            n_par = sum(p.numel() for p in model.parameters())
            n_clim = sum(p.numel() for n, p in model.named_parameters() if n.startswith("climate."))
            log(f"Parameters: {n_par:,} (climate branch {n_clim:,}) | padded grid {Hp} x {Wp}")
        hist = train_model(model, loaders, device, cfg)
        hist.to_csv(os.path.join(out, f"history_seed{seed}.csv"), index=False)
        torch.save(model.state_dict(), os.path.join(out, f"model_seed{seed}.pt"))
        pred, obs, mask, tau, strength = predict(model, loaders["test"], device)
        sl = (slice(None), slice(None), slice(0, H), slice(0, W))
        pred, obs, mask = pred[sl], obs[sl], mask[sl]
        np.savez_compressed(os.path.join(out, f"test_predictions_seed{seed}.npz"),
                            pred=pred, obs=obs, mask=mask, target_time=times[tau].astype(str),
                            lat=lats, lon=lons, targets=np.array(names))
        combine_rows.append(pd.DataFrame({"lead": lead, "seed": seed,
                                       "target_time": times[tau].strftime("%Y-%m-%d"),
                                       "target_month": times[tau].month, "combine_strength": strength}))
        df = metrics_for(pred, obs, mask, names, cfg.drought_threshold)
        df.insert(0, "model", model_label(cfg)); df.insert(0, "lead", lead); df.insert(2, "seed", seed)
        rows.append(df)
        log("\n" + df.round(4).to_string(index=False))
        if seed == cfg.seeds[0]:
            b = metrics_for(clim_map[times[tau].month.values - 1], obs, mask, names, cfg.drought_threshold)
            b.insert(0, "model", "Climatology"); b.insert(0, "lead", lead); b.insert(2, "seed", -1)
            rows.append(b)
    combine = pd.concat(combine_rows, ignore_index=True)
    combine.to_csv(os.path.join(out, "combine_strength_test.csv"), index=False)
    by_month = combine.groupby("target_month").combine_strength.mean().round(4)
    log("Combining strength by target month (test, mean over seeds):\n" + by_month.to_string())
    return pd.concat(rows, ignore_index=True)


def main(cfg=CFG):
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert all(l >= 1 for l in cfg.leads) and cfg.t_clim >= cfg.t_in
    log(f"{model_label(cfg)} | leads {cfg.leads} | device {device}")
    log(f"Grid regularisation: dropout {cfg.grid_dropout}, weight decay {cfg.weight_decay} | "
        f"climate: {list(cfg.clim_inputs)}, {cfg.t_clim} months, "
        f"dropout {cfg.clim_dropout}, aux weight {cfg.aux_weight}, placebo {cfg.clim_placebo}")
    times, lats, lons, X, Y = load_grid(cfg)
    train_months = np.where(times.year.values <= cfg.train_end_year)[0]
    Xn = normalise(X, X[train_months], axes=(0, 2, 3))
    C, A = load_climate(cfg, times)
    Cn = normalise(C, C[train_months], axes=0)
    res = pd.concat([run_lead(cfg, lead, times, lats, lons, Xn, Y, Cn, A, device)
                     for lead in cfg.leads], ignore_index=True)
    res.to_csv(os.path.join(cfg.out_dir, "metrics_per_seed.csv"), index=False)
    cols = ["MSE", "RMSE", "MAE", "R2"]
    summary = res.groupby(["lead", "model", "target"], sort=False)[cols].agg(["mean", "std"])
    summary.to_csv(os.path.join(cfg.out_dir, "metrics_summary.csv"))
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    log("Test set, mean and std over seeds (climatology has no std):")
    print(summary.round(4).to_string())


CLIM_CFG = CFG
CLIM_CFGS = {1: CLIM_CFG,
             3: replace(CLIM_CFG, leads=(3,), out_dir="results_convlstm_climate_lead3")}
if __name__ == "__main__" and RUN_CLIMATE:
    for _lead in (1, 3):
        main(CLIM_CFGS[_lead])


import glob
import shutil

import matplotlib
import matplotlib.pyplot as plt

PLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
PLOT_INK, PLOT_INK2, PLOT_GRID = "#1f1f1d", "#5f5e5a", "#e6e5e0"


def _style(ax):
    ax.grid(axis="y", color=PLOT_GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(PLOT_INK2)
    ax.spines["bottom"].set_color(PLOT_INK2)
    ax.tick_params(colors=PLOT_INK2, length=0)


def plot_training_curves(lead_dir, fig_dir, label):
    files = sorted(glob.glob(os.path.join(lead_dir, "history_seed*.csv")))
    if not files:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, f in enumerate(files):
        seed = os.path.basename(f)[len("history_seed"):-len(".csv")]
        h = pd.read_csv(f)
        col = PLOT_COLORS[k % len(PLOT_COLORS)]
        ax.plot(h["epoch"], h["train_main"], color=col, linewidth=2, linestyle="--",
                label=f"seed {seed} train")
        ax.plot(h["epoch"], h["val_main"], color=col, linewidth=2, label=f"seed {seed} validation")
        best = h["val_main"].idxmin()
        ax.scatter(h.loc[best, "epoch"], h.loc[best, "val_main"], s=60, color=col,
                   edgecolor="white", linewidth=1.5, zorder=5)
    ax.set_xlabel("Epoch", color=PLOT_INK2)
    ax.set_ylabel("Loss (masked L1 + L2)", color=PLOT_INK2)
    ax.set_title(f"{label}: training curves (dot = best validation epoch)", color=PLOT_INK,
                 loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    path = os.path.join(fig_dir, "line_training_curves.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)
    return path


def plot_test_timeseries(lead_dir, fig_dir, label):
    files = sorted(glob.glob(os.path.join(lead_dir, "test_predictions_seed*.npz")))
    if not files:
        return None
    preds = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        preds.append(z["pred"])
        obs, mask, targets = z["obs"], z["mask"], [str(t) for t in z["targets"]]
        times = pd.to_datetime(z["target_time"])
    pred = np.mean(preds, axis=0)
    m = mask.astype(float)
    fig, axes = plt.subplots(len(targets), 1, figsize=(12, 3.2 * len(targets)), sharex=True)
    axes = np.atleast_1d(axes)
    for k, (ax, t) in enumerate(zip(axes, targets)):
        o = (obs[:, k] * m[:, k]).sum((1, 2)) / m[:, k].sum((1, 2))
        p = (pred[:, k] * m[:, k]).sum((1, 2)) / m[:, k].sum((1, 2))
        ax.axhspan(-10, -1.0, color="#f7e3dc", zorder=0)
        ax.plot(times, o, color=PLOT_INK, linewidth=2, label="Observed")
        ax.plot(times, p, color=PLOT_COLORS[1], linewidth=2, label="Predicted (mean of seeds)")
        lo, hi = min(o.min(), p.min()), max(o.max(), p.max())
        ax.set_ylim(lo - 0.2, hi + 0.2)
        ax.axhline(0, color=PLOT_INK2, linewidth=0.8)
        ax.set_ylabel(t, color=PLOT_INK)
        _style(ax)
    axes[0].legend(frameon=False, loc="upper left", ncol=2)
    axes[0].set_title(f"{label}: state-mean sp, test period (shaded = drought, sp <= -1)",
                      color=PLOT_INK, loc="left")
    fig.tight_layout()
    path = os.path.join(fig_dir, "line_test_timeseries.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)
    return path


def plot_metric_bars(out_dir, fig_dir):
    f = os.path.join(out_dir, "metrics_per_seed.csv")
    if not os.path.exists(f):
        return None
    res = pd.read_csv(f)
    res = res[res["model"] != "Climatology"]
    label = res["model"].iloc[0]
    agg = res.groupby("target", sort=False)[["MSE", "RMSE", "MAE", "R2"]].agg(["mean", "std"])
    targets = list(agg.index)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    for ax, met in zip(axes, ["R2", "MSE", "RMSE", "MAE"]):
        mean, std = agg[(met, "mean")].values, agg[(met, "std")].fillna(0).values
        bars = ax.bar(range(len(targets)), mean, 0.6, yerr=std, color=PLOT_COLORS[0],
                      edgecolor="white", linewidth=1.5, zorder=3,
                      error_kw=dict(ecolor=PLOT_INK2, elinewidth=1, capsize=4))
        top = 1.0 if met == "R2" else max(mean + std) * 1.25
        bottom = min(0.0, float(np.min(mean - std)) * 1.25 - 0.05)
        pad = 0.01 * (top - bottom)
        for b, v, s in zip(bars, mean, std):
            if v >= 0:
                ax.text(b.get_x() + b.get_width() / 2, v + s + pad, f"{v:.4f}", ha="center",
                        va="bottom", fontsize=9, color=PLOT_INK)
            else:
                ax.text(b.get_x() + b.get_width() / 2, v - s - pad, f"{v:.4f}", ha="center",
                        va="top", fontsize=9, color=PLOT_INK)
        if bottom < 0:
            ax.axhline(0, color=PLOT_INK2, linewidth=0.8, zorder=4)
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(targets, color=PLOT_INK)
        ax.set_ylim(bottom, top)
        ax.set_title(f"{'R²' if met == 'R2' else met} ({'higher' if met == 'R2' else 'lower'} is better)",
                     color=PLOT_INK)
        _style(ax)
    fig.suptitle(f"{label}: test metrics, mean ± std over seeds", x=0.01, ha="left",
                 color=PLOT_INK, fontsize=13)
    fig.tight_layout()
    path = os.path.join(fig_dir, "bar_test_metrics.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)
    return path


def package_and_plot(cfg=CFG):
    out_dir = cfg.out_dir
    res = pd.read_csv(os.path.join(out_dir, "metrics_per_seed.csv"))
    label = res.loc[res["model"] != "Climatology", "model"].iloc[0]
    for lead in cfg.leads:
        lead_dir = os.path.join(out_dir, f"lead{lead}")
        fig_dir = os.path.join(lead_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        for fn in (plot_training_curves, plot_test_timeseries):
            p = fn(lead_dir, fig_dir, f"{label}, lead {lead}")
            if p:
                log(f"Saved {p}")
    p = plot_metric_bars(out_dir, os.path.join(out_dir, f"lead{cfg.leads[0]}", "figures"))
    if p:
        log(f"Saved {p}")

    models = sorted(glob.glob(os.path.join(out_dir, "lead*", "model_seed*.pt")))
    log(f"Trained models saved ({len(models)}): " + ", ".join(os.path.relpath(m, out_dir) for m in models))
    zip_path = shutil.make_archive(out_dir, "zip", root_dir=out_dir)
    log(f"Download package: {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB)")
    try:
        from IPython.display import FileLink, display
        display(FileLink(os.path.relpath(zip_path, os.getcwd())))
    except Exception:
        pass
    return zip_path


from matplotlib.patches import Patch

CMP_METRICS = [("R2", "R²", True), ("MSE", "MSE", False), ("RMSE", "RMSE", False),
               ("MAE", "MAE", False)]
CMP_GOOD, CMP_BAD = "#0b7a53", "#b3261e"


def _model_means(out_dir):
    res = pd.read_csv(os.path.join(out_dir, "metrics_per_seed.csv"))
    res = res[res["model"] != "Climatology"]
    return res.groupby("target", sort=False)[[m for m, _, _ in CMP_METRICS]].mean()


def comparison_table(noclim_dir, clim_dir):
    a, b = _model_means(noclim_dir), _model_means(clim_dir)
    rows = []
    for t in a.index:
        for met, _, higher in CMP_METRICS:
            v0, v1 = float(a.loc[t, met]), float(b.loc[t, met])
            d = v1 - v0
            rows.append({"target": t, "metric": met, "non_climatic": v0, "climatic": v1,
                         "change": d, "change_pct": 100 * d / abs(v0) if v0 else np.nan,
                         "improved": (d > 0) if higher else (d < 0)})
    return pd.DataFrame(rows)


def _axis_break(ax):
    kw = dict(transform=ax.transAxes, color=PLOT_INK2, clip_on=False, linewidth=1)
    for dy in (0.0, 0.025):
        ax.plot((-0.018, 0.018), (0.01 + dy - 0.012, 0.01 + dy + 0.012), **kw)


def plot_comparison_bars(table, fig_dir, lead):
    targets = list(dict.fromkeys(table["target"]))
    fig, axes = plt.subplots(1, len(targets), figsize=(5.6 * len(targets), 6.2))
    axes = np.atleast_1d(axes)
    w = 0.36
    for ax, t in zip(axes, targets):
        sub = table[table["target"] == t].set_index("metric")
        vals = sub[["non_climatic", "climatic"]].values
        lo, hi = float(vals.min()), float(vals.max())
        span = hi - lo if hi > lo else abs(hi) or 1.0
        bottom = max(0.0, lo - 0.25 * span)
        top = hi + 0.32 * span
        x = np.arange(len(CMP_METRICS))
        b0 = ax.bar(x - w / 2 - 0.01, sub["non_climatic"].values, w, bottom=0, color=PLOT_COLORS[0],
                    edgecolor="white", linewidth=1.5, zorder=3)
        b1 = ax.bar(x + w / 2 + 0.01, sub["climatic"].values, w, color=PLOT_COLORS[1],
                    edgecolor="white", hatch="///", linewidth=1.5, zorder=3)
        for bars in (b0, b1):
            for bar in bars:
                v = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012 * (top - bottom), f"{v:.4f}",
                        ha="center", va="bottom", rotation=90, fontsize=8, color=PLOT_INK)
        for k, (met, _, higher) in enumerate(CMP_METRICS):
            r = sub.loc[met]
            arrow = "▲" if r["change"] > 0 else "▼"
            y = max(r["non_climatic"], r["climatic"]) + 0.14 * (top - bottom)
            ax.text(x[k], y, f"{arrow} {abs(r['change_pct']):.2f}%", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=CMP_GOOD if r["improved"] else CMP_BAD)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab, _ in CMP_METRICS], color=PLOT_INK, fontsize=11)
        ax.set_ylim(bottom, top)
        ax.set_title(t, color=PLOT_INK, fontsize=13, pad=10)
        _style(ax)
        if bottom > 0:
            ticks = [v for v in ax.get_yticks() if bottom + 0.05 * (top - bottom) <= v <= top]
            ax.set_yticks(ticks)
            _axis_break(ax)
    fig.legend(handles=[Patch(facecolor=PLOT_COLORS[0], edgecolor="white", label="ConvLSTM (non-climatic)"),
                        Patch(facecolor=PLOT_COLORS[1], edgecolor="white", hatch="///",
                              label="ConvLSTM + climate (climatic)")],
               loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.01, "Y-axes are zoomed and do NOT start at zero (marked //). Green = improvement "
             "with climate branch, red = worse (▲ increase, ▼ decrease). "
             f"Lead {lead}, test 2013–2025, mean of {len(CLIM_CFG.seeds)} seeds.",
             ha="center", fontsize=9, color=PLOT_INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, "bar_comparison_by_sp.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path[:-4] + ".pdf", bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)
    return path


def compare_and_package(noclim_cfg, clim_cfg, compare_dir=COMPARE_DIR):
    for cfg in (noclim_cfg, clim_cfg):
        package_and_plot(cfg)
    table = comparison_table(noclim_cfg.out_dir, clim_cfg.out_dir)
    os.makedirs(compare_dir, exist_ok=True)
    table.to_csv(os.path.join(compare_dir, "comparison_noclimate_vs_climate.csv"), index=False)
    log("Non-climatic -> climatic (mean over seeds):\n" + table.round(4).to_string(index=False))
    p = plot_comparison_bars(table, compare_dir, clim_cfg.leads[0])
    log(f"Saved {p} (+ .pdf)")
    zip_path = shutil.make_archive(compare_dir, "zip", root_dir=compare_dir)
    log(f"Download package: {zip_path}")
    try:
        from IPython.display import FileLink, display
        display(FileLink(os.path.relpath(zip_path, os.getcwd())))
    except Exception:
        pass
    return zip_path


if __name__ == "__main__":
    for _lead in (1, 3):
        compare_and_package(NOCLIM_CFGS[_lead], CLIM_CFGS[_lead], COMPARE_DIRS[_lead])