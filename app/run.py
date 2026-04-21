"""
run.py  —  QuantEdge Multi-Page Entry Point
────────────────────────────────────────────────────────────────────────────
Run:  streamlit run run.py

Streamlit auto-discovers all numbered pages in the adjacent quantedge_pages/
folder and renders them as sidebar navigation items.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from app.ui_pages._shared import DARK_CSS
from utils.config import cfg

st.set_page_config(
    page_title="QuantEdge",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(DARK_CSS, unsafe_allow_html=True)

demo_pill = '<span class="qe-demo-pill">DEMO</span>' if cfg.DEMO_MODE else ""
st.sidebar.markdown(
    f'<div class="qe-logo">📊 QuantEdge {demo_pill}</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("Institutional Quant Research Platform")
st.sidebar.divider()

st.markdown("""
<div style="text-align:center; padding:40px 0 20px 0;">
  <div style="font-size:52px; margin-bottom:8px;">📊</div>
  <h1 style="font-size:32px; font-weight:700; color:#e8ecf4; margin:0;">QuantEdge</h1>
  <p style="color:#5a6180; font-size:16px; margin-top:8px;">Institutional Quant Research Platform</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pages",        "11")
c2.metric("Core Modules", "9")
c3.metric("API Routes",   "20+")
c4.metric("Mode",         "Demo" if cfg.DEMO_MODE else "Live")

st.divider()

st.markdown("""
### Navigate using the sidebar ←

| Phase | Modules | Pages |
|-------|---------|-------|
| Foundation | Data, Charts, Auditing, Prediction | Dashboard · Graph Research · Auditing · Prediction |
| Quant Core | Signals, Backtest, Portfolio, Risk, Alerts | Signals · Backtest · Portfolio · Risk · Alerts |
| Research Grade | Factors, Regime | Factors · Regime |

### Quick Start
""")

st.code("""
pip install -r requirements.txt
cp .env.example .env
streamlit run run.py
""", language="bash")

st.divider()
st.caption("QuantEdge — Streamlit · Plotly · yFinance · TensorFlow")
