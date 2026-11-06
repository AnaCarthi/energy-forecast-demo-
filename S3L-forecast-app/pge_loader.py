"""Load a PG&E "Green Button" CSV export, or a simple DATE / USAGE / TEMP CSV,
into one row per day.

NEW in the public demo (not part of the original notebook, which started from
an already-cleaned "DATE, USAGE, TEMP" sheet).

Privacy: PG&E exports start with name / address / account-number lines. Those
lines are skipped before parsing and are never displayed, stored or sent
anywhere. Only the date, usage and (optional) temperature columns are kept.
"""
from __future__ import annotations

import io
import re

import pandas as pd

MAX_DAYS = 1095          # keep the most recent 3 years so training stays quick
MIN_DAYS = 60            # below this the model has too little to learn from
MAX_MISSING_FRACTION = 0.10

_DATE_NAMES = {"DATE", "DS", "DAY", "START DATE", "DATETIME", "TIMESTAMP"}
_USAGE_EXACT = ["USAGE", "Y", "CONSUMPTION", "KWH"]


class DataError(ValueError):
    """A problem with the uploaded file, worded so it can be shown to the user."""


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _split_fields(line: str) -> list[str]:
    return [f.strip().strip('"').strip().upper() for f in line.split(",")]


def _find_header(lines: list[str]) -> int:
    """Index of the first line that looks like a table header (has a date column)."""
    for i, line in enumerate(lines[:80]):
        fields = _split_fields(line)
        if len(fields) >= 2 and any(f in _DATE_NAMES for f in fields):
            return i
    raise DataError(
        "Couldn't find a header row with a DATE column. Expected a PG&E Green Button "
        "CSV, or a file with columns like DATE, USAGE and (optionally) TEMP."
    )


def _read_table(raw: bytes) -> pd.DataFrame:
    lines = _decode(raw).splitlines()
    start = _find_header(lines)
    try:
        df = pd.read_csv(
            io.StringIO("\n".join(lines[start:])),
            skipinitialspace=True,
            on_bad_lines="skip",
            index_col=False,
        )
    except Exception as exc:  # pandas raises several different error types
        raise DataError(f"Couldn't parse this CSV ({type(exc).__name__}).") from exc
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def _pick(df: pd.DataFrame, exact: list[str], contains: list[str], exclude: list[str]) -> str | None:
    for name in exact:
        if name in df.columns:
            return name
    for col in df.columns:
        if any(k in col for k in contains) and not any(x in col for x in exclude):
            return col
    return None


def _to_number(s: pd.Series) -> pd.Series:
    cleaned = s.astype(str).str.replace(r"[^0-9eE+\-.]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _to_dates(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, errors="coerce")
    if d.dt.tz is not None:
        d = d.dt.tz_localize(None)
    return d.dt.normalize()


def _date_col(df: pd.DataFrame) -> str:
    for name in ("DATE", "DS", "START DATE", "DAY", "DATETIME", "TIMESTAMP"):
        if name in df.columns:
            return name
    raise DataError("No DATE column found.")


def load_usage_csv(raw: bytes) -> tuple[pd.DataFrame, dict]:
    """Return (daily dataframe with columns ds, y[, temperature], metadata dict)."""
    df = _read_table(raw)

    usage_col = _pick(
        df,
        exact=_USAGE_EXACT,
        contains=["USAGE", "IMPORT", "KWH", "THERM"],
        exclude=["COST", "EXPORT", "$"],
    )
    if usage_col is None:
        raise DataError(
            "Couldn't find a usage column. Looking for a column named like USAGE, "
            f"USAGE (kWh) or IMPORT (kWh). Columns found: {', '.join(df.columns)}."
        )
    temp_col = _pick(df, exact=["TEMP", "TEMPERATURE"], contains=["TEMP"], exclude=[])

    m = re.search(r"\(([^)]+)\)", usage_col)
    unit = "kWh" if not m else {"KWH": "kWh"}.get(m.group(1).strip().upper(), m.group(1).strip().lower())

    work = pd.DataFrame(
        {
            "ds": _to_dates(df[_date_col(df)]),
            "y": _to_number(df[usage_col]),
        }
    )
    if temp_col:
        work["temperature"] = _to_number(df[temp_col])
    n_rows_in = len(work)
    work = work.dropna(subset=["ds"])
    if len(work) < 0.5 * max(n_rows_in, 1):
        raise DataError("Most dates in the DATE column couldn't be read.")
    if work["y"].notna().sum() == 0:
        raise DataError(f"The usage column '{usage_col}' has no numeric values.")

    # Interval data (hourly / 15-minute) -> daily totals. Daily data passes through.
    per_day = work.groupby("ds").size()
    agg = {"y": lambda s: s.sum(min_count=1)}
    if "temperature" in work:
        agg["temperature"] = "mean"
    daily = work.groupby("ds").agg(agg)

    # A partially downloaded last day would look like a sudden drop in usage.
    dropped_partial = False
    if len(per_day) > 3 and per_day.max() > 1 and per_day.iloc[-1] < 0.8 * per_day.median():
        daily = daily.iloc[:-1]
        dropped_partial = True

    daily, filled, trimmed = _regularise(daily)
    daily = daily.reset_index()  # the index is named "ds"

    meta = {
        "unit": unit,
        "intervals_per_day": int(per_day.median()) if len(per_day) else 1,
        "dropped_partial_last_day": dropped_partial,
        "filled_days": filled,
        "trimmed_to_recent_days": trimmed,
        "has_temperature": "temperature" in daily.columns and daily["temperature"].notna().any(),
        "usage_column": usage_col,
    }
    if "temperature" in daily.columns and not meta["has_temperature"]:
        daily = daily.drop(columns="temperature")
    return daily, meta


def _regularise(daily: pd.DataFrame) -> tuple[pd.DataFrame, int, bool]:
    """Full daily index, small gaps interpolated, most recent MAX_DAYS kept."""
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full)
    daily.index.name = "ds"
    missing = int(daily["y"].isna().sum())
    if len(daily) < MIN_DAYS:
        raise DataError(
            f"Only {len(daily)} days of data found. At least {MIN_DAYS} days are needed; "
            "a year or more gives much better forecasts."
        )
    if missing / len(daily) > MAX_MISSING_FRACTION:
        raise DataError(
            f"{missing} of {len(daily)} days have no usage value (more than "
            f"{int(MAX_MISSING_FRACTION * 100)}%). Please upload a more complete export."
        )
    daily = daily.interpolate(method="linear", limit_direction="both")
    trimmed = False
    if len(daily) > MAX_DAYS:
        daily = daily.iloc[-MAX_DAYS:]
        trimmed = True
    return daily, missing, trimmed


def load_temperature_csv(raw: bytes) -> pd.Series:
    """Daily mean temperature from a two-column DATE / TEMP file."""
    df = _read_table(raw)
    temp_col = _pick(df, exact=["TEMP", "TEMPERATURE"], contains=["TEMP"], exclude=[])
    date_col = _date_col(df)
    if temp_col is None:
        others = [c for c in df.columns if c != date_col]
        numeric = [c for c in others if _to_number(df[c]).notna().mean() > 0.8]
        if len(numeric) != 1:
            raise DataError("Couldn't tell which column holds temperature. Name it TEMP.")
        temp_col = numeric[0]
    out = pd.DataFrame({"ds": _to_dates(df[date_col]), "temperature": _to_number(df[temp_col])})
    out = out.dropna()
    if out.empty:
        raise DataError("The temperature file has no usable rows.")
    return out.groupby("ds")["temperature"].mean()


def attach_temperature(daily: pd.DataFrame, temps: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Add a temperature column from a separate file. Requires decent date overlap."""
    t = temps.reindex(pd.DatetimeIndex(daily["ds"]))
    coverage = float(t.notna().mean())
    if coverage < 0.8:
        raise DataError(
            f"The temperature file only covers {coverage:.0%} of your usage dates. "
            "It needs to cover at least 80%."
        )
    out = daily.copy()
    out["temperature"] = t.interpolate(limit_direction="both").to_numpy()
    return out, {"temperature_coverage": coverage, "has_temperature": True}
