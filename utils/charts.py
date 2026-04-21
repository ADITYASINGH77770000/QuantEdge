"""
utils/charts.py
──────────────────────────────────────────────────────────────────────────────
Reusable Plotly chart helper functions — keeps page files thin.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


PALETTE = px.colors.qualitative.Plotly


def candlestick(df: pd.DataFrame, ticker: str) -> go.Figure:
    """OHLCV candlestick with volume bars below."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
    colors = ["green" if c >= o else "red"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
                         marker_color=colors, name="Volume"), row=2, col=1)
    fig.update_layout(title=f"{ticker} — Price & Volume",
                      xaxis_rangeslider_visible=False,
                      template="plotly_dark", height=550)
    return fig


def line_chart(df: pd.DataFrame, title: str = "",
               y_label: str = "Value") -> go.Figure:
    """Multi-series line chart from a DataFrame."""
    fig = px.line(df, title=title, template="plotly_dark")
    fig.update_yaxes(title_text=y_label)
    return fig


def equity_curve(returns: pd.Series, benchmark: pd.Series = None,
                 title: str = "Equity Curve") -> go.Figure:
    """Cumulative return equity curve vs optional benchmark."""
    cum = (1 + returns).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum.index, y=cum.values,
                             name="Strategy", line=dict(color="cyan", width=2)))
    if benchmark is not None:
        bm = (1 + benchmark).cumprod()
        fig.add_trace(go.Scatter(x=bm.index, y=bm.values,
                                 name="Buy & Hold", line=dict(color="gray", dash="dash")))
    fig.update_layout(title=title, template="plotly_dark",
                      yaxis_title="Cumulative Return", height=400)
    return fig


def drawdown_chart(returns: pd.Series) -> go.Figure:
    """Underwater / drawdown chart."""
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    fig = go.Figure(go.Scatter(x=dd.index, y=dd.values,
                               fill="tozeroy", fillcolor="rgba(255,50,50,0.3)",
                               line=dict(color="red"), name="Drawdown"))
    fig.update_layout(title="Drawdown", template="plotly_dark",
                      yaxis_tickformat=".1%", height=300)
    return fig


def efficient_frontier(rets: np.ndarray, vols: np.ndarray,
                       sharpes: np.ndarray,
                       max_sharpe_pt: tuple = None,
                       min_vol_pt: tuple = None) -> go.Figure:
    """Efficient frontier scatter — Monte Carlo simulated portfolios."""
    fig = go.Figure(go.Scatter(
        x=vols, y=rets, mode="markers",
        marker=dict(color=sharpes, colorscale="Viridis",
                    colorbar=dict(title="Sharpe"), size=4, opacity=0.6),
        name="Portfolios"))
    if max_sharpe_pt:
        fig.add_trace(go.Scatter(x=[max_sharpe_pt[0]], y=[max_sharpe_pt[1]],
                                 mode="markers+text", text=["Max Sharpe"],
                                 textposition="top right",
                                 marker=dict(size=14, color="gold", symbol="star"),
                                 name="Max Sharpe"))
    if min_vol_pt:
        fig.add_trace(go.Scatter(x=[min_vol_pt[0]], y=[min_vol_pt[1]],
                                 mode="markers+text", text=["Min Vol"],
                                 textposition="top right",
                                 marker=dict(size=14, color="lime", symbol="diamond"),
                                 name="Min Vol"))
    fig.update_layout(title="Efficient Frontier", template="plotly_dark",
                      xaxis_title="Annualised Volatility",
                      yaxis_title="Annualised Return", height=500)
    return fig


def heatmap(corr: pd.DataFrame, title: str = "Correlation Heatmap") -> go.Figure:
    """Annotated correlation heatmap."""
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    title=title, template="plotly_dark")
    return fig


def metric_card_row(metrics: dict) -> str:
    """Return HTML for a row of metric cards (used with st.markdown)."""
    cards = "".join(
        f"""<div style="background:#1e2130;border-radius:8px;padding:14px 20px;
                        margin:4px;display:inline-block;min-width:130px;text-align:center;">
              <div style="color:#aaa;font-size:12px">{k}</div>
              <div style="color:#fff;font-size:20px;font-weight:bold">{v}</div>
            </div>"""
        for k, v in metrics.items()
    )
    return f'<div style="display:flex;flex-wrap:wrap;gap:4px">{cards}</div>'
