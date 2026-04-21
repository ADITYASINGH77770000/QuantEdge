# core/alerts.py

class AlertEngine:
    def __init__(self, config=None):
        self.config = config or {
            "signal_threshold": 0.8,
            "drawdown_limit": 0.1,
            "var_limit": 0.05,
            "sharpe_min": 1.0,
            "confidence_threshold": 0.85,
            "volume_spike_multiplier": 2.5
        }

    def generate_alerts(self, data):
        alerts = []

        # 1. SIGNAL ALERT
        if data.get("signal_score", 0) > self.config["signal_threshold"]:
            alerts.append({
                "type": "Signal",
                "message": f"Strong signal detected (Score: {data['signal_score']:.2f})",
                "level": "HIGH"
            })

        # 2. REGIME ALERT
        if data.get("current_regime") != data.get("prev_regime"):
            alerts.append({
                "type": "Regime",
                "message": f"Market regime changed: {data['prev_regime']} → {data['current_regime']}",
                "level": "MEDIUM"
            })

        # 3. RISK ALERT
        if data.get("drawdown", 0) > self.config["drawdown_limit"]:
            alerts.append({
                "type": "Risk",
                "message": f"Drawdown exceeded: {data['drawdown']:.2%}",
                "level": "CRITICAL"
            })

        if data.get("var", 0) > self.config["var_limit"]:
            alerts.append({
                "type": "Risk",
                "message": f"VaR exceeded: {data['var']:.2%}",
                "level": "CRITICAL"
            })

        # 4. PERFORMANCE ALERT
        if data.get("sharpe", 0) < self.config["sharpe_min"]:
            alerts.append({
                "type": "Performance",
                "message": f"Sharpe dropped: {data['sharpe']:.2f}",
                "level": "MEDIUM"
            })

        # 5. PREDICTION ALERT
        if data.get("confidence", 0) > self.config["confidence_threshold"]:
            alerts.append({
                "type": "Prediction",
                "message": f"High confidence prediction ({data['confidence']:.2f})",
                "level": "HIGH"
            })

        # 6. TRADE ALERT
        if data.get("action"):
            alerts.append({
                "type": "Trade",
                "message": f"{data['action']} {data['ticker']} | Size: {data['position_size']:.2%}",
                "level": "INFO"
            })

        # 7. PORTFOLIO ALERT
        old_w = data.get("old_weights", {})
        new_w = data.get("new_weights", {})
        for asset in new_w:
            if abs(new_w[asset] - old_w.get(asset, 0)) > 0.05:
                alerts.append({
                    "type": "Portfolio",
                    "message": f"{asset}: {old_w.get(asset,0):.2%} → {new_w[asset]:.2%}",
                    "level": "MEDIUM"
                })

        # 8. ANOMALY ALERT
        if data.get("volume", 0) > data.get("avg_volume", 1) * self.config["volume_spike_multiplier"]:
            alerts.append({
                "type": "Anomaly",
                "message": f"Volume spike detected",
                "level": "HIGH"
            })

        return alerts