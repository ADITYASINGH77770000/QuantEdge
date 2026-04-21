"""Shared graph analytics used by both Streamlit and API consumers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


DEFAULT_GRAPH_BENCHMARK = "SPY"
GRAPH_FEATURES = [
    {"key": "relative_strength", "label": "Relative Strength Studio"},
    {"key": "volume_profile", "label": "Volume Profile"},
    {"key": "gap_session", "label": "Gap & Session Decomposition"},
    {"key": "seasonality", "label": "Seasonality Atlas"},
    {"key": "volume_shock", "label": "Volume Shock Analyzer"},
    {"key": "breakout_context", "label": "Breakout Context Map"},
    {"key": "candle_structure", "label": "Candle Structure Lab"},
]


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close", "Volume"]
    frame = df.copy()
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    for column in required:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[required].apply(pd.to_numeric, errors="coerce")
    return frame.dropna(subset=["Open", "High", "Low", "Close"])


def _to_number(value, digits: int = 6):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, digits)


def _to_bool(value) -> bool | None:
    if value is None or pd.isna(value):
        return None
    return bool(value)


def _records(df: pd.DataFrame) -> list[dict]:
    frame = df.copy()
    if isinstance(frame.index, pd.DatetimeIndex):
        frame.index = frame.index.strftime("%Y-%m-%d")
        frame.index.name = "Date"
    frame = frame.reset_index()

    records: list[dict] = []
    for row in frame.to_dict(orient="records"):
        cleaned = {}
        for key, value in row.items():
            if isinstance(value, (np.bool_, bool)) or key.startswith("Is") or key.endswith("Flag") or key in {"Filled", "SameDirection"}:
                cleaned[key] = _to_bool(value)
            elif isinstance(value, (np.integer,)):
                cleaned[key] = int(value)
            elif isinstance(value, (str, int)) and not isinstance(value, bool):
                cleaned[key] = value
            elif value is None or pd.isna(value):
                cleaned[key] = None
            else:
                cleaned[key] = _to_number(value)
        records.append(cleaned)
    return records


def _latest_value(series: pd.Series, digits: int = 6):
    clean = series.dropna()
    if clean.empty:
        return None
    return _to_number(clean.iloc[-1], digits)


def _feature_map() -> dict[str, str]:
    return {item["key"]: item["label"] for item in GRAPH_FEATURES}


def _relative_strength_feature(
    asset: pd.DataFrame,
    benchmark: pd.DataFrame,
    ticker: str,
    benchmark_ticker: str,
) -> dict:
    aligned = pd.DataFrame(
        {
            "AssetClose": asset["Close"],
            "BenchmarkClose": benchmark["Close"],
        }
    ).dropna()

    if aligned.empty:
        return {
            "label": _feature_map()["relative_strength"],
            "benchmark": benchmark_ticker,
            "cards": {},
            "relative_series": [],
            "rolling_stats": [],
        }

    normalised = aligned.div(aligned.iloc[0]).mul(100.0)
    ratio = aligned["AssetClose"].div(aligned["BenchmarkClose"]).mul(100.0)
    returns = aligned.pct_change()
    rolling_beta = returns["AssetClose"].rolling(63).cov(returns["BenchmarkClose"]) / returns["BenchmarkClose"].rolling(63).var()
    rolling_outperformance = aligned["AssetClose"].pct_change(21) - aligned["BenchmarkClose"].pct_change(21)

    relative_series = pd.DataFrame(
        {
            ticker: normalised["AssetClose"],
            benchmark_ticker: normalised["BenchmarkClose"],
            "RelativeRatio": ratio,
        },
        index=aligned.index,
    )
    rolling_stats = pd.DataFrame(
        {
            "Outperformance21d": rolling_outperformance,
            "Beta63d": rolling_beta,
        },
        index=aligned.index,
    ).dropna(how="all")

    cards = {
        "Asset Return": _to_number(aligned["AssetClose"].iloc[-1] / aligned["AssetClose"].iloc[0] - 1.0, 4),
        "Benchmark Return": _to_number(aligned["BenchmarkClose"].iloc[-1] / aligned["BenchmarkClose"].iloc[0] - 1.0, 4),
        "Outperformance": _to_number(
            aligned["AssetClose"].iloc[-1] / aligned["AssetClose"].iloc[0]
            - aligned["BenchmarkClose"].iloc[-1] / aligned["BenchmarkClose"].iloc[0],
            4,
        ),
        "Rolling Beta": _latest_value(rolling_beta, 3),
    }

    return {
        "label": _feature_map()["relative_strength"],
        "benchmark": benchmark_ticker,
        "cards": cards,
        "relative_series": _records(relative_series),
        "rolling_stats": _records(rolling_stats),
    }


def _volume_profile_feature(frame: pd.DataFrame) -> dict:
    working = frame.copy()
    typical_price = (working["High"] + working["Low"] + working["Close"]) / 3.0
    valid = typical_price.notna() & working["Volume"].notna()
    working = working.loc[valid]
    typical_price = typical_price.loc[valid]

    if working.empty:
        return {
            "label": _feature_map()["volume_profile"],
            "cards": {},
            "profile": [],
            "price_context": [],
            "note": "Built from daily OHLCV only.",
        }

    price_min = float(typical_price.min())
    price_max = float(typical_price.max())
    bins = max(12, min(28, int(math.sqrt(len(working))) + 8))
    if math.isclose(price_min, price_max):
        price_max = price_min + 1e-6

    edges = np.linspace(price_min, price_max, bins + 1)
    bucket = pd.Series(pd.cut(typical_price, bins=edges, include_lowest=True, duplicates="drop"), index=working.index, name="PriceBucket")
    grouped = working.groupby(bucket, observed=False)["Volume"].sum()
    profile = grouped.reset_index(name="Volume")
    profile["PriceMid"] = profile["PriceBucket"].apply(lambda interval: float(interval.mid) if pd.notna(interval) else np.nan)
    profile = profile.dropna(subset=["PriceMid"])
    profile = profile[["PriceMid", "Volume"]].sort_values("PriceMid").reset_index(drop=True)

    total_volume = float(profile["Volume"].sum()) or 1.0
    profile["VolumeShare"] = profile["Volume"] / total_volume

    poc_idx = int(profile["Volume"].idxmax())
    selected = {poc_idx}
    value_area_volume = float(profile.loc[poc_idx, "Volume"])
    target_volume = 0.70 * total_volume
    left = poc_idx - 1
    right = poc_idx + 1
    while value_area_volume < target_volume and (left >= 0 or right < len(profile)):
        left_volume = float(profile.loc[left, "Volume"]) if left >= 0 else -1.0
        right_volume = float(profile.loc[right, "Volume"]) if right < len(profile) else -1.0
        if right_volume >= left_volume and right < len(profile):
            selected.add(right)
            value_area_volume += float(profile.loc[right, "Volume"])
            right += 1
        elif left >= 0:
            selected.add(left)
            value_area_volume += float(profile.loc[left, "Volume"])
            left -= 1
        else:
            break

    selected_rows = profile.loc[sorted(selected)]
    value_area_low = float(selected_rows["PriceMid"].min())
    value_area_high = float(selected_rows["PriceMid"].max())
    inside_value_area = working["Close"].between(value_area_low, value_area_high).mean()

    cards = {
        "POC Price": _to_number(profile.loc[poc_idx, "PriceMid"], 2),
        "Value Area Low": _to_number(value_area_low, 2),
        "Value Area High": _to_number(value_area_high, 2),
        "Inside VA": _to_number(inside_value_area, 4),
    }

    price_context = pd.DataFrame({"Close": working["Close"]}, index=working.index)
    return {
        "label": _feature_map()["volume_profile"],
        "cards": cards,
        "profile": _records(profile),
        "price_context": _records(price_context),
        "note": "Daily approximation using typical price ((H + L + C) / 3) because daily bars do not expose intraday volume-by-price ladders.",
    }


def _gap_session_feature(frame: pd.DataFrame) -> dict:
    gap_df = frame.copy()
    gap_df["PrevClose"] = gap_df["Close"].shift(1)
    gap_df["GapReturn"] = gap_df["Open"] / gap_df["PrevClose"] - 1.0
    gap_df["SessionReturn"] = gap_df["Close"] / gap_df["Open"] - 1.0
    gap_df["Filled"] = np.where(
        gap_df["GapReturn"] > 0,
        gap_df["Low"] <= gap_df["PrevClose"],
        np.where(gap_df["GapReturn"] < 0, gap_df["High"] >= gap_df["PrevClose"], False),
    )
    gap_df["SameDirection"] = (
        np.sign(gap_df["GapReturn"]).replace(0, np.nan) == np.sign(gap_df["SessionReturn"]).replace(0, np.nan)
    )
    gap_df = gap_df.dropna(subset=["PrevClose", "GapReturn", "SessionReturn"])

    non_zero_gap = gap_df["GapReturn"].abs() > 1e-9
    cards = {
        "Avg Abs Gap": _to_number(gap_df["GapReturn"].abs().mean(), 4),
        "Avg Session Move": _to_number(gap_df["SessionReturn"].abs().mean(), 4),
        "Fill Rate": _to_number(gap_df.loc[non_zero_gap, "Filled"].mean(), 4),
        "Continuation Rate": _to_number(gap_df.loc[non_zero_gap, "SameDirection"].mean(), 4),
    }

    series = gap_df[["GapReturn", "SessionReturn", "Filled", "SameDirection"]]
    return {
        "label": _feature_map()["gap_session"],
        "cards": cards,
        "series": _records(series),
    }


def _seasonality_feature(frame: pd.DataFrame) -> dict:
    ret = frame["Close"].pct_change().dropna()
    if ret.empty:
        return {
            "label": _feature_map()["seasonality"],
            "cards": {},
            "weekday": [],
            "month_of_year": [],
            "monthly_heatmap": [],
        }

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    weekday = (
        ret.to_frame("Return")
        .assign(Weekday=lambda data: data.index.day_name())
        .groupby("Weekday")
        .agg(AvgReturn=("Return", "mean"), HitRate=("Return", lambda values: (values > 0).mean()), Count=("Return", "count"))
        .reindex(weekday_order)
        .dropna(how="all")
    )

    monthly_ret = frame["Close"].resample("ME").last().pct_change().dropna()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_of_year = (
        monthly_ret.to_frame("Return")
        .assign(MonthNum=lambda data: data.index.month, Month=lambda data: data.index.month.map(lambda value: month_names[value - 1]))
        .groupby(["MonthNum", "Month"])
        .agg(AvgReturn=("Return", "mean"), HitRate=("Return", lambda values: (values > 0).mean()), Count=("Return", "count"))
        .reset_index()
        .sort_values("MonthNum")
        .drop(columns=["MonthNum"])
    )

    monthly_heatmap = monthly_ret.to_frame("Return")
    monthly_heatmap["Year"] = monthly_heatmap.index.year
    monthly_heatmap["Month"] = monthly_heatmap.index.month.map(lambda value: month_names[value - 1])
    monthly_heatmap["MonthNum"] = monthly_heatmap.index.month
    monthly_heatmap = monthly_heatmap.sort_values(["Year", "MonthNum"])[["Year", "Month", "Return"]]

    best_weekday = weekday["AvgReturn"].dropna().idxmax() if not weekday["AvgReturn"].dropna().empty else "N/A"
    best_month_row = month_of_year.loc[month_of_year["AvgReturn"].idxmax()] if not month_of_year.empty else None
    cards = {
        "Best Weekday": best_weekday,
        "Best Weekday Avg": _to_number(weekday["AvgReturn"].max(), 4),
        "Best Month": str(best_month_row["Month"]) if best_month_row is not None else "N/A",
        "Best Month Avg": _to_number(best_month_row["AvgReturn"], 4) if best_month_row is not None else None,
    }

    return {
        "label": _feature_map()["seasonality"],
        "cards": cards,
        "weekday": _records(weekday),
        "month_of_year": _records(month_of_year),
        "monthly_heatmap": _records(monthly_heatmap),
    }


def _volume_shock_feature(frame: pd.DataFrame) -> dict:
    working = frame.copy()
    ret = working["Close"].pct_change()
    volume_mean = working["Volume"].rolling(20).mean()
    volume_std = working["Volume"].rolling(20).std().replace(0, np.nan)
    volume_z = (working["Volume"] - volume_mean) / volume_std
    next_1d = working["Close"].pct_change().shift(-1)
    next_5d = working["Close"].pct_change(5).shift(-5)
    forward_returns = pd.concat([ret.shift(-step) for step in range(1, 6)], axis=1)
    forward_vol_5d = forward_returns.std(axis=1) * np.sqrt(252)

    signal = pd.DataFrame(
        {
            "VolumeZ": volume_z,
            "Next1dReturn": next_1d,
            "Next5dReturn": next_5d,
            "Next5dVol": forward_vol_5d,
            "ShockFlag": volume_z >= 2.0,
        },
        index=working.index,
    ).dropna(subset=["VolumeZ"])

    shocks = signal["ShockFlag"] == True
    cards = {
        "Shock Days": int(shocks.sum()),
        "Avg Next 1d": _to_number(signal.loc[shocks, "Next1dReturn"].mean(), 4),
        "Avg Next 5d": _to_number(signal.loc[shocks, "Next5dReturn"].mean(), 4),
        "Avg Next 5d Vol": _to_number(signal.loc[shocks, "Next5dVol"].mean(), 4),
    }

    return {
        "label": _feature_map()["volume_shock"],
        "cards": cards,
        "series": _records(signal),
    }


def _breakout_context_feature(frame: pd.DataFrame) -> dict:
    working = frame.copy()
    rolling_high_20 = working["High"].rolling(20).max().shift(1)
    rolling_high_63 = working["High"].rolling(63).max().shift(1)
    rolling_high_252 = working["High"].rolling(252).max().shift(1)
    rolling_low_20 = working["Low"].rolling(20).min().shift(1)
    rolling_low_63 = working["Low"].rolling(63).min().shift(1)
    rolling_low_252 = working["Low"].rolling(252).min().shift(1)

    range_pos_252 = (working["Close"] - rolling_low_252) / (rolling_high_252 - rolling_low_252)
    compression_20 = rolling_high_20 / rolling_low_20 - 1.0
    context = pd.DataFrame(
        {
            "Dist20dHigh": working["Close"] / rolling_high_20 - 1.0,
            "Dist63dHigh": working["Close"] / rolling_high_63 - 1.0,
            "Dist252dHigh": working["Close"] / rolling_high_252 - 1.0,
            "RangePos252d": range_pos_252,
            "Compression20d": compression_20,
            "Breakout20d": working["Close"] > rolling_high_20,
            "Breakout63d": working["Close"] > rolling_high_63,
        },
        index=working.index,
    ).dropna(how="all")

    cards = {
        "252d Range Pos": _latest_value(context["RangePos252d"], 4),
        "Dist 20d High": _latest_value(context["Dist20dHigh"], 4),
        "Dist 63d High": _latest_value(context["Dist63dHigh"], 4),
        "20d Compression": _latest_value(context["Compression20d"], 4),
    }

    return {
        "label": _feature_map()["breakout_context"],
        "cards": cards,
        "series": _records(context),
    }


def _bucket_average(feature: pd.Series, target: pd.Series, labels: list[str]) -> pd.DataFrame:
    joined = pd.concat([feature.rename("Feature"), target.rename("Target")], axis=1).dropna()
    if joined.empty:
        return pd.DataFrame(columns=["Bucket", "AvgNext1d", "Count"])
    n_buckets = min(len(labels), int(joined["Feature"].nunique()))
    if n_buckets <= 1:
        return pd.DataFrame(
            [{"Bucket": labels[0], "AvgNext1d": joined["Target"].mean(), "Count": int(len(joined))}]
        )
    ranked = joined["Feature"].rank(method="first")
    bucket_labels = labels[:n_buckets]
    buckets = pd.qcut(ranked, q=n_buckets, labels=bucket_labels)
    bucketed = (
        joined.assign(Bucket=buckets)
        .groupby("Bucket")
        .agg(AvgNext1d=("Target", "mean"), Count=("Target", "count"))
        .reset_index()
    )
    bucketed["Bucket"] = bucketed["Bucket"].astype(str)
    return bucketed


def _candle_structure_feature(frame: pd.DataFrame) -> dict:
    working = frame.copy()
    candle_range = (working["High"] - working["Low"]).replace(0, np.nan)
    body = (working["Close"] - working["Open"]).abs() / candle_range
    upper_wick = (working["High"] - working[["Open", "Close"]].max(axis=1)) / candle_range
    lower_wick = (working[["Open", "Close"]].min(axis=1) - working["Low"]) / candle_range
    close_location = (working["Close"] - working["Low"]) / candle_range
    wick_imbalance = lower_wick - upper_wick
    next_1d = working["Close"].pct_change().shift(-1)

    structure = pd.DataFrame(
        {
            "BodyPct": body,
            "UpperWickPct": upper_wick,
            "LowerWickPct": lower_wick,
            "CloseLocation": close_location,
            "WickImbalance": wick_imbalance,
            "Next1dReturn": next_1d,
        },
        index=working.index,
    ).dropna(how="all")

    close_buckets = _bucket_average(structure["CloseLocation"], structure["Next1dReturn"], ["Q1", "Q2", "Q3", "Q4", "Q5"])
    wick_buckets = _bucket_average(structure["WickImbalance"], structure["Next1dReturn"], ["Q1", "Q2", "Q3", "Q4", "Q5"])

    best_close_bucket = close_buckets.loc[close_buckets["AvgNext1d"].idxmax()] if not close_buckets.empty else None
    cards = {
        "Latest Body %": _latest_value(structure["BodyPct"], 4),
        "Latest Close Loc": _latest_value(structure["CloseLocation"], 4),
        "Best Close Bucket": str(best_close_bucket["Bucket"]) if best_close_bucket is not None else "N/A",
        "Best Bucket Avg": _to_number(best_close_bucket["AvgNext1d"], 4) if best_close_bucket is not None else None,
    }

    return {
        "label": _feature_map()["candle_structure"],
        "cards": cards,
        "series": _records(structure),
        "close_location_buckets": _records(close_buckets),
        "wick_buckets": _records(wick_buckets),
    }


def build_graph_feature_payload(
    df: pd.DataFrame,
    *,
    ticker: str,
    benchmark_df: pd.DataFrame | None = None,
    benchmark_ticker: str = DEFAULT_GRAPH_BENCHMARK,
) -> dict:
    frame = _clean_ohlcv(df)
    benchmark = _clean_ohlcv(benchmark_df if benchmark_df is not None else df)

    payload = {
        "ticker": ticker,
        "benchmark": benchmark_ticker,
        "feature_options": GRAPH_FEATURES,
        "rows": int(len(frame)),
        "last_close": _to_number(frame["Close"].iloc[-1], 4) if len(frame) else None,
        "avg_volume": _to_number(frame["Volume"].mean(), 2) if len(frame) else None,
        "data": _records(frame[["Open", "High", "Low", "Close", "Volume"]]),
        "correlation": frame[["Open", "High", "Low", "Close", "Volume"]].corr().round(4).to_dict() if len(frame) else {},
        "features": {
            "relative_strength": _relative_strength_feature(frame, benchmark, ticker, benchmark_ticker),
            "volume_profile": _volume_profile_feature(frame),
            "gap_session": _gap_session_feature(frame),
            "seasonality": _seasonality_feature(frame),
            "volume_shock": _volume_shock_feature(frame),
            "breakout_context": _breakout_context_feature(frame),
            "candle_structure": _candle_structure_feature(frame),
        },
    }
    return payload


__all__ = [
    "DEFAULT_GRAPH_BENCHMARK",
    "GRAPH_FEATURES",
    "build_graph_feature_payload",
]
