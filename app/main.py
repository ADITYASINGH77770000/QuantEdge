"""
app/main.py
──────────────────────────────────────────────────────────────────────────────
QuantEdge — main entry point and navigation router.
Run with: streamlit run app/main.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from utils.config import cfg

st.set_page_config(
    page_title="QuantEdge |  Quant Research",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar branding ──────────────────────────────────────────────────────────
st.sidebar.title("📊 QuantEdge")
st.sidebar.caption("Institutional Quant Research Platform")
if cfg.DEMO_MODE:
    st.sidebar.info("🎭 **Demo Mode** — synthetic data, no API keys needed.")
st.sidebar.divider()

# ── Home page ─────────────────────────────────────────────────────────────────
st.title("QuantEdge — Institutional Quant Research Platform")
st.markdown("""
Welcome to **QuantEdge**, upgraded from Finnovix into a hedge-fund-grade research platform.

| Phase | Modules | Status |
|-------|---------|--------|
| Foundation | Dashboard, Graphs, Auditing, Prediction | ✅ Refactored |
| Quant Core | Signals, Backtest, Portfolio, Risk | ✅ New |
| Research Grade | Factors, Regime | ✅ Extreme |
| Polish | Tests, Docker, PDF export, Demo mode | ✅ Complete |

**Navigate** using the sidebar pages ←
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Modules", "11")
col2.metric("Build Phases", "4")
col3.metric("Lines of Code", "3,000+")
col4.metric("Mode", "Demo" if cfg.DEMO_MODE else "Live")

st.divider()
st.subheader("Quick Start")
st.code("""
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env
# Edit .env with your API keys

# 3. Run the app
streamlit run app/main.py
""", language="bash")
