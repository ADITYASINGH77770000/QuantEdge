from fastapi.testclient import TestClient

from api.server import app


client = TestClient(app)


def test_compat_ohlcv_route_returns_lowercase_rows():
    res = client.get("/ohlcv", params={"ticker": "GOOG", "start": "2020-01-01"})
    assert res.status_code == 200
    payload = res.json()
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert {"date", "open", "high", "low", "close", "volume"} <= set(row)


def test_compat_metrics_route_returns_numeric_payload():
    res = client.get("/metrics", params={"ticker": "GOOG", "start": "2020-01-01"})
    assert res.status_code == 200
    payload = res.json()
    assert {"sharpe", "sortino", "cagr", "max_drawdown", "win_rate", "var_95", "cvar_95", "ann_vol"} <= set(payload)
    assert all(isinstance(payload[key], (int, float)) for key in payload)
 


def test_audit_route_exposes_extended_statistics():
    res = client.get("/api/audit/GOOG", params={"start": "2020-01-01"})
    assert res.status_code == 200
    payload = res.json()
    assert {"missing_dates", "adf", "acf", "pacf"} <= set(payload)
    assert {"statistic", "p_value", "critical_values", "is_stationary"} <= set(payload["adf"])
    assert isinstance(payload["acf"], list)
    assert isinstance(payload["pacf"], list)


def test_alerts_route_reports_notification_status_without_sending_by_default():
    res = client.get("/api/alerts/GOOG")
    assert res.status_code == 200
    payload = res.json()
    assert {"email_configured", "emails_sent", "notifications_requested", "alerts"} <= set(payload)
    assert payload["notifications_requested"] is False
    assert isinstance(payload["emails_sent"], int)
    assert payload["alerts"]
    assert {"metric", "price", "threshold", "triggered", "insight", "email_sent"} <= set(payload["alerts"][0])


def test_graphs_route_returns_quant_feature_payload():
    res = client.get("/api/graphs/GOOG", params={"start": "2020-01-01", "benchmark": "SPY"})
    assert res.status_code == 200
    payload = res.json()
    assert {"ticker", "benchmark", "feature_options", "features", "rows", "last_close", "avg_volume", "data"} <= set(payload)
    assert payload["benchmark"] == "SPY"
    assert isinstance(payload["feature_options"], list)
    assert payload["feature_options"]
    assert {"relative_strength", "volume_profile", "gap_session", "seasonality", "volume_shock", "breakout_context", "candle_structure"} <= set(payload["features"])
    assert isinstance(payload["features"]["relative_strength"]["relative_series"], list)
    assert isinstance(payload["features"]["volume_profile"]["profile"], list)
