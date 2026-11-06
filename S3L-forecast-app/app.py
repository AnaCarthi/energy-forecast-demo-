"""Energy Usage Forecast: public demo adapted from an S3L research sample.

Run locally:   streamlit run app.py
"""
import hashlib
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pge_loader import DataError, attach_temperature, load_temperature_csv, load_usage_csv
from recommendations import build_recommendations, yoy_change

HERE = Path(__file__).parent
SAMPLE_USAGE = HERE / "sample_data" / "sample_pge_greenbutton_style_SYNTHETIC.csv"
SAMPLE_TEMP = HERE / "sample_data" / "sample_temperature_SYNTHETIC.csv"
REPO_URL = "https://github.com/AnaCarthi/Copy-of-S3L-Forecasting-Research-Sample"
N_DAYS = 183

HISTORY_COLOR = "#6b7280"
FORECAST_COLOR = "#2563eb"
TEMP_COLOR = "#d97706"

st.set_page_config(page_title="Energy Usage Forecast", layout="wide")


@st.cache_data(show_spinner=False, max_entries=3, ttl=3600)
def cached_forecast(df: pd.DataFrame, use_temperature: bool) -> dict:
    # Imported here so the page loads instantly; PyTorch is only needed once a forecast runs.
    from forecast_core import run_forecast

    return run_forecast(df, use_temperature)


def explore_chart(data: pd.DataFrame, unit: str, has_temp: bool) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=data["ds"], y=data["y"], name=f"Usage ({unit}/day)", line=dict(color=HISTORY_COLOR, width=1.6))
    if has_temp:
        fig.add_scatter(x=data["ds"], y=data["temperature"], name="Temperature", yaxis="y2",
                        line=dict(color=TEMP_COLOR, width=1.2), opacity=0.8)
        fig.update_layout(yaxis2=dict(title="Temperature", overlaying="y", side="right", showgrid=False))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title=f"{unit}/day",
                      legend=dict(orientation="h", y=1.12), hovermode="x unified")
    return fig


def forecast_chart(res: dict, unit: str) -> go.Figure:
    hist, fut, fit = res["history"], res["future"], res["fitted"]
    fig = go.Figure()
    fig.add_scatter(x=hist["ds"], y=hist["y"], name="Your usage", line=dict(color=HISTORY_COLOR, width=1.6))
    fig.add_scatter(x=fit["ds"], y=fit["yhat"], name="Model fit to history", visible="legendonly",
                    line=dict(color=HISTORY_COLOR, width=1, dash="dot"))
    fig.add_scatter(x=fut["ds"], y=fut["yhat"], name="Forecast", line=dict(color=FORECAST_COLOR, width=2.2))
    start = str(pd.to_datetime(fut["ds"].iloc[0]).date())
    fig.add_shape(type="line", x0=start, x1=start, y0=0, y1=1, yref="paper", line=dict(color="#9ca3af", width=1, dash="dot"))
    fig.add_annotation(x=start, y=1, yref="paper", text="forecast starts", showarrow=False, xanchor="left", font=dict(size=11))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title=f"{unit}/day",
                      legend=dict(orientation="h", y=1.1), hovermode="x unified")
    return fig


def monthly_table(fut: pd.DataFrame, unit: str) -> pd.DataFrame:
    f = fut.assign(ds=pd.to_datetime(fut["ds"]))
    g = f.groupby(f["ds"].dt.to_period("M"))["yhat"].agg(["sum", "mean", "size"]).reset_index()
    g["Month"] = [
        p.strftime("%b %Y") + ("" if n == p.days_in_month else " (partial)") for p, n in zip(g["ds"], g["size"])
    ]
    return pd.DataFrame({"Month": g["Month"], "Days": g["size"], f"Forecast ({unit})": g["sum"].round(0),
                         f"Avg per day ({unit})": g["mean"].round(1)})


# ---------------------------------------------------------------- header
st.title("Energy Usage Forecast")
st.write(f"Upload your PG&E electricity usage and get a {N_DAYS}-day (about 6-month) forecast, "
         "a month-by-month breakdown and practical suggestions for reducing usage.")
st.info("This public demo is adapted from a research sample created during an S3L internship. "
        "The forecasting model is the original NeuralProphet setup; the upload interface, data cleaning and "
        "suggestions were added for this demo. See **About this demo** at the bottom for exactly what changed.")

# ---------------------------------------------------------------- step 1
with st.container(border=True):
    st.subheader("1. Add your data")
    c1, c2 = st.columns(2)
    usage_file = c1.file_uploader(
        "PG&E usage CSV", type=["csv"],
        help="PG&E's Green Button 'Download My Data' CSV (hourly, 15-minute or daily), or any CSV with a DATE "
             "column and a USAGE column. A TEMP column is used if present. A year or more of data works best.")
    temp_file = c2.file_uploader(
        "Temperature CSV (optional)", type=["csv"],
        help="Two columns: DATE and TEMP (daily). PG&E exports don't include temperature. Skip this if your usage "
             "file already has a TEMP column, or to run without temperature.")
    b1, b2 = st.columns([1, 3])
    if b1.button("Try the sample data"):
        st.session_state["use_sample"] = True
    b2.caption("Your files are processed in memory for this session and are not saved by this app. "
               "Name, address and account lines at the top of PG&E exports are ignored. "
               "Not affiliated with or endorsed by PG&E.")

use_sample = st.session_state.get("use_sample", False) and usage_file is None
if usage_file is None and not use_sample:
    st.stop()

try:
    if usage_file is not None:
        data, meta = load_usage_csv(usage_file.getvalue())
        temp_bytes = temp_file.getvalue() if temp_file is not None else None
    else:
        data, meta = load_usage_csv(SAMPLE_USAGE.read_bytes())
        temp_bytes = temp_file.getvalue() if temp_file is not None else SAMPLE_TEMP.read_bytes()
        st.success("Showing synthetic sample data (generated for this demo, not real usage).")
    if temp_bytes is not None:
        if meta["has_temperature"]:
            st.caption("Your usage file already includes temperature, so the separate temperature file was not used.")
        else:
            data, _ = attach_temperature(data, load_temperature_csv(temp_bytes))
            meta["has_temperature"] = True
except DataError as err:
    st.error(str(err))
    st.stop()

unit, has_temp = meta["unit"], meta["has_temperature"]
data_key = hashlib.md5(pd.util.hash_pandas_object(data).to_numpy().tobytes()).hexdigest() + str(has_temp)

# ---------------------------------------------------------------- your data
st.subheader("Your data")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Days of data", f"{len(data):,}")
m2.metric("Date range", f"{data['ds'].iloc[0]:%b %d, %Y} to {data['ds'].iloc[-1]:%b %d, %Y}")
m3.metric("Total usage", f"{data['y'].sum():,.0f} {unit}")
m4.metric("Average per day", f"{data['y'].mean():,.1f} {unit}")
notes = []
if meta["intervals_per_day"] > 1:
    notes.append(f"Interval data was added up into daily totals ({meta['intervals_per_day']} readings per day).")
if meta["dropped_partial_last_day"]:
    notes.append("The last day was incomplete, so it was left out.")
if meta["filled_days"]:
    notes.append(f"{meta['filled_days']} missing day(s) were filled in by interpolation.")
if meta["trimmed_to_recent_days"]:
    notes.append("Only the most recent 3 years were used.")
if notes:
    st.caption(" ".join(notes))
if len(data) < 365:
    st.warning("Less than a year of data: the model can't see a full seasonal cycle, so forecasts for times of year "
               "missing from your history will be much less reliable.")
if not has_temp:
    st.info("No temperature data found, so the forecast will run without the temperature input and the "
            "temperature insight will be skipped. Add a temperature CSV above to include it.")
st.plotly_chart(explore_chart(data, unit, has_temp), width="stretch")

# ---------------------------------------------------------------- step 2
with st.container(border=True):
    st.subheader("2. Run the forecast")
    r1, r2 = st.columns([2, 1])
    rate = r1.number_input(f"Optional: your average price per {unit} ($)", min_value=0.0, max_value=5.0, value=0.0,
                           step=0.01, format="%.2f", help="Leave at 0 to skip cost estimates.")
    r2.write("")
    if r2.button(f"Run {N_DAYS}-day forecast", type="primary", width="stretch"):
        st.session_state["run_key"] = data_key

if st.session_state.get("run_key") != data_key:
    st.stop()

try:
    with st.spinner("Training the model on your data. This usually takes a minute or two..."):
        res = cached_forecast(data, has_temp)
except Exception as err:  # noqa: BLE001 - show a friendly message, keep details available
    st.error("The forecast couldn't be completed with this data. Try a longer or more complete export.")
    with st.expander("Technical details"):
        st.code(f"{type(err).__name__}: {err}")
    st.stop()

fut, hist = res["future"], res["history"]
total, avg = float(fut["yhat"].sum()), float(fut["yhat"].mean())
yoy = yoy_change(hist, fut)

# ---------------------------------------------------------------- results
st.subheader("Forecast")
k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Projected usage, next {N_DAYS} days", f"{total:,.0f} {unit}")
k2.metric("Projected average per day", f"{avg:,.1f} {unit}")
k3.metric("vs same dates last year", f"{yoy['pct']:+.1f}%" if yoy else "n/a",
          help=None if yoy else "Needs history covering the same dates one year earlier.")
k4.metric("Typical daily error (in-sample)", f"{res['in_sample_mae']:,.1f} {unit}",
          help="Average gap between the model's fit and your actual past usage. It measures fit to history, "
               "not how accurate future predictions will be.")
st.plotly_chart(forecast_chart(res, unit), width="stretch")

left, right = st.columns([3, 2])
with left:
    st.markdown("**Month by month**")
    st.dataframe(monthly_table(fut, unit), hide_index=True, width="stretch")
with right:
    st.write("")
    out = pd.DataFrame({"date": pd.to_datetime(fut["ds"]).dt.date, f"forecast_{unit}": fut["yhat"].round(2)})
    st.download_button("Download daily forecast (CSV)", out.to_csv(index=False), "forecast.csv", "text/csv",
                       width="stretch")

st.subheader("Suggestions")
st.caption("Generated from simple calculations on your history and the forecast. General guidance, not an energy audit.")
for rec in build_recommendations(hist, fut, unit, rate if rate > 0 else None):
    with st.container(border=True):
        st.markdown(f"**{rec['title']}**")
        st.write(rec["body"])

with st.expander("Technical details: the original notebook's total"):
    st.write(
        f"The original notebook printed the sum of the model's `yhat1` column as its \"net energy consumption\": "
        f"**{res['original_total']:,.1f} {unit}** for your data. With this model configuration `yhat1` is filled in "
        "for your historical days (plus the first forecast day) but not for the rest of the forecast, so that figure "
        f"mostly reflects the model's fit to your past usage. This app reports the sum of the {N_DAYS} forecast days "
        "instead, shown above.")

# ---------------------------------------------------------------- about
with st.expander("About this demo"):
    st.markdown(f"""
**What it is.** A web version of a forecasting notebook from an S3L internship. [Source code and the original notebook]({REPO_URL}).

**What is original.** The model: NeuralProphet with 10 trend changepoints, yearly, weekly and daily seasonality, temperature as a lagged regressor, and a {N_DAYS}-day horizon. Those settings are unchanged.

**What changed from the notebook**
- Code fixes so it runs on current library versions (removed an unused import, replaced a pandas method that no longer exists, a PyTorch loading flag).
- The notebook's hard-coded list of future temperatures is replaced by each calendar month's average from your own data.
- The total shown is the sum of the {N_DAYS} forecast days. The notebook's number is shown under *Technical details*.
- Added: CSV upload and cleaning for PG&E exports, interval-to-daily totals, charts, monthly table, and the suggestions.

**Limitations**
- It is a statistical projection from your past usage. It can't anticipate changes such as new appliances, solar, an EV, or people moving in or out.
- The app does not test accuracy on data the model hasn't seen; the error shown is in-sample. In one test on synthetic data with a known answer (about 13 months of history), the 183-day total came out roughly 10% low.
- Temperature is included as in the original, but in testing it changed the forecast only slightly, so don't treat the forecast as a weather-driven prediction.
- Less than a year of data means the model hasn't seen all seasons.
""")
