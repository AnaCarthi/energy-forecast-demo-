"""Forecasting core for the public demo.

This is the model cell from "Full Summer Forecasting.ipynb" (cell 3) turned into
a function. The model configuration is unchanged: same NeuralProphet
hyperparameters, same lagged temperature regressor, same 183-day horizon.

Every line that differs from the notebook is tagged:

    # ORIGINAL  - copied from the notebook as written
    # CHANGED   - needed so the code runs on current library versions
    # NEW       - added for the public demo

Run `diff`-style against the notebook if you want to audit it.
"""
import os

# CHANGED (compatibility): PyTorch >= 2.6 refuses to reload the temporary training
# checkpoint that NeuralProphet 0.8 writes during learning-rate search. That file
# is created locally by this same process, so loading it normally is safe.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import warnings

import numpy as np
import pandas as pd
# CHANGED: the notebook also imported `trapz` from scipy.integrate. It was never
# used, and recent SciPy versions have removed it, so the import was dropped.
from neuralprophet import NeuralProphet, set_log_level, set_random_seed

warnings.filterwarnings("ignore")

N_FORECAST_DAYS = 183  # ORIGINAL: n_forecasts=183 / periods=183


def default_future_temperatures(df: pd.DataFrame, future_dates) -> list[float]:
    """NEW. The notebook hard-coded one list of future temperatures. A public tool
    can't do that, so each future day gets the average temperature of the same
    calendar month in the uploaded history (falling back to the last 30 days'
    average for months the history doesn't cover)."""
    by_month = df.assign(m=pd.to_datetime(df["ds"]).dt.month).groupby("m")["temperature"].mean()
    fallback = float(df["temperature"].tail(30).mean())
    return [float(by_month.get(d.month, fallback)) for d in pd.to_datetime(future_dates)]


def run_forecast(df: pd.DataFrame, use_temperature: bool = True) -> dict:
    """df has columns: ds (date), y (daily usage) and, optionally, temperature."""
    df = df.copy()
    if not use_temperature and "temperature" in df:  # NEW
        df = df.drop(columns="temperature")

    # ORIGINAL: group by date and sum
    df = df.groupby("ds").sum().reset_index()

    # ORIGINAL
    set_log_level("ERROR")

    # NEW: fixed seed so running the same data twice gives the same numbers
    # (NeuralProphet training is otherwise random). Does not change the model.
    set_random_seed(0)

    # ORIGINAL: model configuration, unchanged
    m = NeuralProphet(
        n_changepoints=10,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        n_forecasts=N_FORECAST_DAYS,
    )
    m.set_plotting_backend("plotly")

    # ORIGINAL: temperature as a lagged regressor
    if use_temperature:  # NEW: only when the user provided temperature data
        m.add_lagged_regressor("temperature")

    # ORIGINAL: train. CHANGED: progress=None hides the console progress bar.
    metrics = m.fit(df, progress=None)

    # ORIGINAL: build the future dataframe
    df_future = m.make_future_dataframe(df, n_historic_predictions=True, periods=N_FORECAST_DAYS)

    if use_temperature:
        # CHANGED: the notebook used a hard-coded `future_temps` list (see
        # default_future_temperatures). It also used Series.append, which was
        # removed in pandas 2.0; pd.concat does the same thing.
        future_temps = default_future_temperatures(df, df_future["ds"].iloc[-N_FORECAST_DAYS:])
        df_future["temperature"] = pd.concat(
            [df["temperature"], pd.Series(future_temps)]
        ).reset_index(drop=True)

    # ORIGINAL: predict
    forecast = m.predict(df_future)

    # ORIGINAL: the notebook's "net energy consumption" number.
    original_total = float(forecast["yhat1"].sum())

    # NEW: read the actual forecast. With this configuration (n_forecasts=183,
    # n_lags=0) `yhat1` is filled in for the historical rows but is empty for
    # nearly all future rows, so the notebook's sum above is essentially the
    # model's fit to the past. The forecast for the k-th future day is stored in
    # column `yhat{k}` on that day's row.
    T = len(df)
    if len(forecast) != T + N_FORECAST_DAYS:
        raise RuntimeError(
            f"Unexpected forecast length {len(forecast)}; expected {T + N_FORECAST_DAYS}."
        )
    cols = forecast[[f"yhat{k}" for k in range(1, N_FORECAST_DAYS + 1)]].to_numpy()
    future = pd.DataFrame(
        {
            "ds": forecast["ds"].iloc[T:].to_numpy(),
            "yhat": [cols[T + k - 1, k - 1] for k in range(1, N_FORECAST_DAYS + 1)],
        }
    )
    if future["yhat"].isna().any():
        raise RuntimeError("The model returned an incomplete forecast. Try again with more data.")

    fitted = pd.DataFrame({"ds": forecast["ds"].iloc[:T].to_numpy(), "yhat": forecast["yhat1"].iloc[:T].to_numpy()})
    in_sample_mae = float(np.nanmean(np.abs(df["y"].to_numpy() - fitted["yhat"].to_numpy())))

    return {
        "history": df,
        "future": future,
        "fitted": fitted,
        "in_sample_mae": in_sample_mae,
        "original_total": original_total,
        "used_temperature": use_temperature,
    }
