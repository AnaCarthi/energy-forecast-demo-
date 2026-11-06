"""Rule-based recommendations. NEW in the public demo.

Not part of the original notebook, and not a machine-learning step: each item
is a simple calculation on the uploaded history and the forecast, followed by
general guidance. Wording is deliberately cautious. None of this is an energy
audit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _fmt(x: float, d: int = 0) -> str:
    return f"{x:,.{d}f}"


def _temperature_sensitivity(hist: pd.DataFrame):
    """Fit usage = a + b*max(0, base - T) + c*max(0, T - base), choosing the balance
    point `base` that fits best. Unit-agnostic (works for degrees F or C)."""
    d = hist.dropna(subset=["temperature", "y"])
    if len(d) < 60 or d["temperature"].std() < 2:
        return None
    T, y = d["temperature"].to_numpy(float), d["y"].to_numpy(float)
    sst = float(((y - y.mean()) ** 2).sum())
    if sst == 0:
        return None
    best = None
    for base in np.linspace(np.percentile(T, 20), np.percentile(T, 80), 25):
        X = np.column_stack([np.ones_like(T), np.maximum(0, base - T), np.maximum(0, T - base)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(((y - X @ coef) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, base, coef)
    sse, base, coef = best
    return {"r2": 1 - sse / sst, "base": float(base), "heat": float(coef[1]), "cool": float(coef[2])}


def yoy_change(hist: pd.DataFrame, future: pd.DataFrame):
    """Percent change of the forecast vs the same dates a year earlier (weekday-aligned,
    364 days back), or None if the history doesn't cover at least 80% of those dates."""
    prior = hist.assign(ds=pd.to_datetime(hist["ds"])).set_index("ds")["y"]
    dates = pd.to_datetime(future["ds"]) - pd.Timedelta(days=364)
    matched = prior.reindex(dates)
    if matched.notna().mean() < 0.8:
        return None
    last_year = float(matched.mean())
    return {"pct": (float(future["yhat"].mean()) / last_year - 1) * 100, "last_year_mean": last_year}


def build_recommendations(hist: pd.DataFrame, future: pd.DataFrame, unit: str = "kWh",
                          rate: float | None = None) -> list[dict]:
    """hist: ds, y[, temperature]. future: ds, yhat. Returns [{'title','body'}, ...]."""
    recs: list[dict] = []
    f = future.assign(ds=pd.to_datetime(future["ds"]))
    h = hist.assign(ds=pd.to_datetime(hist["ds"]))
    f_mean = float(f["yhat"].mean())
    f_total = float(f["yhat"].sum())

    # 1. Cost, only if the user supplied a price.
    if rate and rate > 0:
        monthly = f.groupby(f["ds"].dt.to_period("M"))["yhat"].sum()
        top = monthly.idxmax()
        recs.append({
            "title": f"Estimated cost: about ${_fmt(f_total * rate)} over the forecast period",
            "body": (f"At ${rate:.2f} per {unit}, the forecast of {_fmt(f_total)} {unit} comes to roughly "
                     f"${_fmt(f_total * rate)}. The most expensive month is projected to be "
                     f"{top.strftime('%B %Y')} (about ${_fmt(float(monthly[top]) * rate)}, if it is a full month). "
                     f"Every 5% you cut would save about ${_fmt(f_total * rate * 0.05)}. "
                     "Your actual bill depends on your rate plan, tiers and fixed charges."),
        })

    # 2. Highest-usage month ahead (average per day, so partial months compare fairly).
    by_month = f.groupby(f["ds"].dt.to_period("M"))["yhat"].mean()
    peak = by_month.idxmax()
    if by_month[peak] > 1.10 * f_mean:
        pct = (by_month[peak] / f_mean - 1) * 100
        recs.append({
            "title": f"Plan for {peak.strftime('%B %Y')}, your highest-usage month ahead",
            "body": (f"Projected usage averages {_fmt(by_month[peak], 1)} {unit}/day that month, "
                     f"{_fmt(pct)}% above the forecast average of {_fmt(f_mean, 1)}. Before then, it is worth "
                     "servicing heating or cooling equipment, checking filters and thermostat schedules, and, "
                     "if you are on a time-of-use plan, moving flexible loads such as laundry, dishwasher and "
                     "EV charging out of the plan's peak-price hours."),
        })
    else:
        recs.append({
            "title": "Your forecast is fairly steady month to month",
            "body": (f"No month is projected to run more than 10% above the {_fmt(f_mean, 1)} {unit}/day average, "
                     "so savings are more likely to come from year-round habits and equipment than from "
                     "preparing for one peak season."),
        })

    # 3. Same period last year.
    yoy = yoy_change(hist, future)
    if yoy is not None:
        change = yoy["pct"]
        if abs(change) >= 5:
            word = "higher" if change > 0 else "lower"
            recs.append({
                "title": f"Forecast is {_fmt(abs(change))}% {word} than the same period last year",
                "body": (f"The forecast averages {_fmt(f_mean, 1)} {unit}/day against {_fmt(yoy['last_year_mean'], 1)} "
                         "for the matching dates a year earlier. "
                         + ("If nothing changed at home, that gap is worth a look: new appliances, more time at home, "
                            "or equipment running less efficiently can all cause it."
                            if change > 0 else
                            "That is a good sign if it reflects changes you made. If it doesn't, double-check "
                            "that the history looks right, since the model extrapolates from it.")),
            })

    # 4. Weekday vs weekend.
    wk = h.assign(weekend=h["ds"].dt.dayofweek >= 5).groupby("weekend")["y"].mean()
    if len(wk) == 2 and wk[False] > 0:
        diff = (wk[True] / wk[False] - 1) * 100
        if abs(diff) >= 8:
            if diff > 0:
                recs.append({
                    "title": f"Weekends use {_fmt(diff)}% more than weekdays",
                    "body": (f"Your history averages {_fmt(wk[True], 1)} {unit} on weekends vs {_fmt(wk[False], 1)} on "
                             "weekdays. Spreading laundry and other big chores across the week, or running them "
                             "off-peak, can flatten this."),
                })
            else:
                recs.append({
                    "title": f"Weekdays use {_fmt(abs(diff))}% more than weekends",
                    "body": (f"Your history averages {_fmt(wk[False], 1)} {unit} on weekdays vs {_fmt(wk[True], 1)} on "
                             "weekends. If the home is often empty on weekdays, check whether heating or cooling "
                             "schedules and always-on devices are running when nobody needs them."),
                })

    # 5. Always-on share, from the quietest days.
    quiet = float(h["y"].quantile(0.10))
    mean_all = float(h["y"].mean())
    if mean_all > 0:
        share = quiet / mean_all
        if share >= 0.6:
            recs.append({
                "title": "Most of your usage looks like always-on load",
                "body": (f"Your quietest 10% of days still use about {_fmt(quiet, 1)} {unit}, {_fmt(share * 100)}% of "
                         "your average day, so weather and routine explain relatively little. Standby electronics, an "
                         "older refrigerator or freezer, a water heater, or a pool or well pump are common culprits."),
            })
        elif share < 0.35:
            recs.append({
                "title": "Your usage swings a lot from day to day",
                "body": (f"Your quietest days use only {_fmt(share * 100)}% of an average day, so the biggest "
                         "savings are probably on your highest-usage days. Looking at what is different on those "
                         "days (heating or cooling, guests, EV charging) is a good place to start."),
            })

    # 6. Temperature sensitivity, only if there is temperature data.
    if "temperature" in h and h["temperature"].notna().any():
        s = _temperature_sensitivity(h)
        if s and s["r2"] >= 0.15:
            parts = []
            if s["cool"] > 0.02:
                parts.append(f"about {_fmt(s['cool'], 2)} {unit}/day more for each degree above ~{_fmt(s['base'])}")
            if s["heat"] > 0.02:
                parts.append(f"about {_fmt(s['heat'], 2)} {unit}/day more for each degree below it")
            if parts:
                recs.append({
                    "title": "Temperature explains part of your usage",
                    "body": ("In your history, usage rises by " + " and ".join(parts)
                             + f" (temperature explains roughly {_fmt(s['r2'] * 100)}% of day-to-day variation). "
                             "This is a pattern in your data, not proof of cause. If it is strong, thermostat "
                             "settings, air sealing, insulation and shading are usually the most effective levers."),
                })
    return recs
