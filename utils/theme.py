"""
utils/theme.py
──────────────────────────────────────────────────────────────────────────────
QuantEdge Cognitive UI Engine
Neuroscience-driven design: attention magnetism, reward loops, flow state.
Call apply_quantedge_theme() at the top of every page.
"""

import streamlit as st


# ── Plotly dark theme config shared across all pages ─────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"

PLOTLY_LAYOUT = dict(
    template=PLOTLY_TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(8,10,18,0.6)",
    font=dict(family="Inter, JetBrains Mono, monospace", color="#c8d6e5"),
    title_font=dict(size=16, color="#e8f4fd"),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(
        gridcolor="rgba(100,120,160,0.15)",
        zerolinecolor="rgba(100,120,160,0.25)",
        tickfont=dict(size=11),
    ),
    yaxis=dict(
        gridcolor="rgba(100,120,160,0.15)",
        zerolinecolor="rgba(100,120,160,0.25)",
        tickfont=dict(size=11),
    ),
)

# ── Colour palette ─────────────────────────────────────────────────────────
COLORS = {
    "signal_buy":    "#00f5a0",
    "signal_sell":   "#ff4757",
    "neutral":       "#ffd32a",
    "accent_blue":   "#0be0ff",
    "accent_purple": "#a55efd",
    "accent_gold":   "#ffd700",
    "dim":           "#546e8a",
    "glow_green":    "rgba(0,245,160,0.7)",
    "glow_red":      "rgba(255,71,87,0.7)",
    "glow_blue":     "rgba(11,224,255,0.7)",
}


# ── Master CSS ────────────────────────────────────────────────────────────────
_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
  --bg-void:      #03050d;
  --bg-deep:      #070b16;
  --bg-panel:     rgba(10,16,30,0.85);
  --bg-glass:     rgba(14,22,42,0.72);
  --border-glow:  rgba(11,224,255,0.18);
  --border-dim:   rgba(80,110,160,0.22);
  --text-primary: #e8f4fd;
  --text-dim:     #7a9abb;
  --text-accent:  #0be0ff;
  --accent-buy:   #00f5a0;
  --accent-sell:  #ff4757;
  --accent-gold:  #ffd700;
  --accent-purp:  #a55efd;
  --glow-buy:     rgba(0,245,160,0.35);
  --glow-sell:    rgba(255,71,87,0.35);
  --glow-blue:    rgba(11,224,255,0.3);
  --glow-gold:    rgba(255,215,0,0.3);
  --font-main:    'Inter', sans-serif;
  --font-mono:    'JetBrains Mono', monospace;
  --radius:       12px;
  --radius-sm:    8px;
  --transition:   all 0.35s cubic-bezier(0.25,0.46,0.45,0.94);
}

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg-void) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-main) !important;
}

/* ── Animated starfield background ── */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  z-index: 0;
  background:
    radial-gradient(ellipse at 20% 20%, rgba(11,224,255,0.04) 0%, transparent 60%),
    radial-gradient(ellipse at 80% 80%, rgba(165,94,253,0.04) 0%, transparent 60%),
    radial-gradient(ellipse at 50% 50%, rgba(0,245,160,0.02) 0%, transparent 70%);
  animation: ambientShift 12s ease-in-out infinite alternate;
  pointer-events: none;
}

@keyframes ambientShift {
  0%   { opacity: 0.6; transform: scale(1); }
  100% { opacity: 1;   transform: scale(1.04); }
}

/* ── Grid fabric overlay ── */
[data-testid="stAppViewContainer"]::after {
  content: '';
  position: fixed;
  inset: 0;
  z-index: 0;
  background-image:
    linear-gradient(rgba(11,224,255,0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(11,224,255,0.04) 1px, transparent 1px);
  background-size: 60px 60px;
  animation: gridDrift 20s linear infinite;
  pointer-events: none;
}

@keyframes gridDrift {
  0%   { transform: translate(0, 0); }
  100% { transform: translate(60px, 60px); }
}

/* ── Main content above bg ── */
[data-testid="stMain"], section.main, .block-container {
  position: relative;
  z-index: 1;
}

.block-container {
  padding-top: 2rem !important;
  max-width: 1400px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(7,11,22,0.98) 0%, rgba(10,16,32,0.98) 100%) !important;
  border-right: 1px solid var(--border-glow) !important;
  backdrop-filter: blur(20px);
}

[data-testid="stSidebar"]::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--accent-buy), var(--text-accent), var(--accent-purp));
  animation: topBarFlow 4s linear infinite;
  background-size: 200% 100%;
}

@keyframes topBarFlow {
  0%   { background-position: 0% 50%; }
  100% { background-position: 200% 50%; }
}

[data-testid="stSidebar"] * { font-family: var(--font-main) !important; }

/* Sidebar nav links */
[data-testid="stSidebarNav"] a {
  border-radius: var(--radius-sm) !important;
  transition: var(--transition) !important;
  margin: 2px 0 !important;
}
[data-testid="stSidebarNav"] a:hover {
  background: var(--bg-glass) !important;
  box-shadow: 0 0 12px var(--glow-blue) !important;
  transform: translateX(4px) !important;
}

/* ── Page headings ── */
h1 {
  font-size: 2rem !important;
  font-weight: 700 !important;
  background: linear-gradient(135deg, var(--text-primary) 0%, var(--text-accent) 60%, var(--accent-purp) 100%);
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  background-clip: text !important;
  letter-spacing: -0.5px !important;
  margin-bottom: 0.2rem !important;
  animation: titleGlow 3s ease-in-out infinite alternate;
}

@keyframes titleGlow {
  0%   { filter: drop-shadow(0 0 8px rgba(11,224,255,0.3)); }
  100% { filter: drop-shadow(0 0 18px rgba(11,224,255,0.6)); }
}

h2 {
  font-size: 1.3rem !important;
  font-weight: 600 !important;
  color: var(--text-primary) !important;
  border-bottom: 1px solid var(--border-glow);
  padding-bottom: 6px;
}

h3 {
  font-size: 1.05rem !important;
  color: var(--text-dim) !important;
  font-weight: 500 !important;
}

/* ── Captions ── */
[data-testid="stCaption"], .stCaption {
  color: var(--text-dim) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.8rem !important;
  letter-spacing: 0.5px;
}

/* ── Streamlit metric widgets ── */
[data-testid="stMetric"] {
  background: var(--bg-glass) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: var(--radius) !important;
  padding: 18px 20px !important;
  transition: var(--transition) !important;
  backdrop-filter: blur(16px) !important;
}
[data-testid="stMetric"]:hover {
  border-color: var(--border-glow) !important;
  box-shadow: 0 0 20px var(--glow-blue), 0 4px 20px rgba(0,0,0,0.4) !important;
  transform: translateY(-2px) !important;
}
[data-testid="stMetricLabel"] {
  color: var(--text-dim) !important;
  font-size: 0.78rem !important;
  text-transform: uppercase !important;
  letter-spacing: 1px !important;
  font-family: var(--font-mono) !important;
}
[data-testid="stMetricValue"] {
  color: var(--text-accent) !important;
  font-size: 1.7rem !important;
  font-weight: 700 !important;
  font-family: var(--font-mono) !important;
}
[data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

/* ── Buttons ── */
[data-testid="stButton"] > button,
button[kind="primary"],
.stButton button {
  background: linear-gradient(135deg, rgba(11,224,255,0.15), rgba(165,94,253,0.15)) !important;
  border: 1px solid var(--border-glow) !important;
  color: var(--text-accent) !important;
  font-family: var(--font-main) !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  letter-spacing: 0.5px !important;
  border-radius: var(--radius) !important;
  padding: 10px 28px !important;
  transition: var(--transition) !important;
  position: relative !important;
  overflow: hidden !important;
  text-transform: uppercase !important;
}
[data-testid="stButton"] > button::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(11,224,255,0.08), rgba(165,94,253,0.08));
  opacity: 0;
  transition: opacity 0.3s;
}
[data-testid="stButton"] > button:hover {
  background: linear-gradient(135deg, rgba(11,224,255,0.25), rgba(165,94,253,0.25)) !important;
  border-color: var(--text-accent) !important;
  box-shadow: 0 0 24px var(--glow-blue), 0 0 48px rgba(11,224,255,0.15) !important;
  transform: translateY(-2px) scale(1.02) !important;
  color: #fff !important;
}
[data-testid="stButton"] > button:active {
  transform: translateY(0) scale(0.98) !important;
}

button[kind="primary"] {
  background: linear-gradient(135deg, rgba(0,245,160,0.2), rgba(11,224,255,0.2)) !important;
  border-color: var(--accent-buy) !important;
  color: var(--accent-buy) !important;
}
button[kind="primary"]:hover {
  box-shadow: 0 0 28px var(--glow-buy), 0 0 56px rgba(0,245,160,0.1) !important;
  color: #fff !important;
}

/* ── Select boxes & inputs ── */
[data-testid="stSelectbox"] > div,
[data-testid="stMultiSelect"] > div,
.stSelectbox > div > div,
.stMultiSelect > div > div {
  background: var(--bg-glass) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  transition: var(--transition) !important;
}
[data-testid="stSelectbox"]:hover > div,
[data-testid="stMultiSelect"]:hover > div {
  border-color: var(--border-glow) !important;
  box-shadow: 0 0 12px var(--glow-blue) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] {
  padding: 4px 0 !important;
}
[data-testid="stSlider"] [role="slider"] {
  background: var(--text-accent) !important;
  box-shadow: 0 0 12px var(--glow-blue) !important;
}
[data-testid="stSlider"] .st-bm {
  background: linear-gradient(90deg, var(--accent-buy), var(--text-accent)) !important;
}

/* ── Number inputs ── */
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
.stTextInput input {
  background: var(--bg-glass) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-mono) !important;
  transition: var(--transition) !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stDateInput"] input:focus {
  border-color: var(--text-accent) !important;
  box-shadow: 0 0 12px var(--glow-blue) !important;
  outline: none !important;
}

/* ── Info / Warning / Success / Error boxes ── */
[data-testid="stAlert"],
.stAlert, .element-container .stAlert {
  border-radius: var(--radius) !important;
  backdrop-filter: blur(12px) !important;
  border-left-width: 3px !important;
  font-size: 0.88rem !important;
}
div[data-testid="stInfo"] {
  background: rgba(11,224,255,0.06) !important;
  border-color: var(--text-accent) !important;
}
div[data-testid="stWarning"] {
  background: rgba(255,215,0,0.06) !important;
  border-color: var(--accent-gold) !important;
}
div[data-testid="stSuccess"] {
  background: rgba(0,245,160,0.06) !important;
  border-color: var(--accent-buy) !important;
}
div[data-testid="stError"] {
  background: rgba(255,71,87,0.08) !important;
  border-color: var(--accent-sell) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button[role="tab"] {
  background: transparent !important;
  border: none !important;
  color: var(--text-dim) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.85rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.5px !important;
  padding: 10px 20px !important;
  text-transform: uppercase !important;
  transition: var(--transition) !important;
  border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[role="tab"]:hover {
  color: var(--text-primary) !important;
  border-bottom-color: rgba(11,224,255,0.4) !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: var(--text-accent) !important;
  border-bottom: 2px solid var(--text-accent) !important;
  background: rgba(11,224,255,0.05) !important;
  border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
  background: var(--bg-glass) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: var(--radius) !important;
  backdrop-filter: blur(12px) !important;
  transition: var(--transition) !important;
}
[data-testid="stExpander"]:hover {
  border-color: var(--border-glow) !important;
}
[data-testid="stExpanderToggleIcon"] { color: var(--text-accent) !important; }
[data-testid="stExpander"] summary {
  color: var(--text-dim) !important;
  font-size: 0.88rem !important;
  font-family: var(--font-mono) !important;
  padding: 12px 16px !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
  border-radius: var(--radius) !important;
  overflow: hidden !important;
  border: 1px solid var(--border-dim) !important;
  box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
}
.dvn-scroller { background: var(--bg-panel) !important; }
.glideDataEditor { background: var(--bg-panel) !important; }

/* ── Plotly charts ── */
.js-plotly-plot, .plotly .svg-container {
  border-radius: var(--radius) !important;
  overflow: hidden !important;
}
[data-testid="stPlotlyChart"] {
  border-radius: var(--radius) !important;
  border: 1px solid var(--border-dim) !important;
  overflow: hidden !important;
  transition: var(--transition) !important;
  box-shadow: 0 4px 30px rgba(0,0,0,0.4) !important;
}
[data-testid="stPlotlyChart"]:hover {
  border-color: var(--border-glow) !important;
  box-shadow: 0 0 30px var(--glow-blue), 0 4px 40px rgba(0,0,0,0.5) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--text-accent) !important; }

/* ── Divider ── */
hr {
  border: none !important;
  height: 1px !important;
  background: linear-gradient(90deg, transparent, var(--border-glow), transparent) !important;
  margin: 1.5rem 0 !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
  background: rgba(165,94,253,0.1) !important;
  border: 1px solid rgba(165,94,253,0.35) !important;
  color: var(--accent-purp) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.82rem !important;
  border-radius: var(--radius-sm) !important;
  padding: 8px 20px !important;
  transition: var(--transition) !important;
}
[data-testid="stDownloadButton"] button:hover {
  background: rgba(165,94,253,0.2) !important;
  box-shadow: 0 0 16px rgba(165,94,253,0.4) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb {
  background: var(--border-glow);
  border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: var(--text-accent); }

/* ── Label text ── */
label, .stSelectbox label, .stSlider label,
.stNumberInput label, .stDateInput label,
.stMultiSelect label, .stCheckbox label {
  color: var(--text-dim) !important;
  font-size: 0.8rem !important;
  font-family: var(--font-mono) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.8px !important;
  font-weight: 500 !important;
}

/* ── Sidebar title ── */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] .sidebar-title {
  font-size: 1.3rem !important;
  background: linear-gradient(135deg, var(--accent-buy), var(--text-accent));
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  background-clip: text !important;
}

/* ── Sidebar caption ── */
[data-testid="stSidebar"] [data-testid="stCaption"] {
  font-size: 0.72rem !important;
  color: var(--text-dim) !important;
}

/* ── Multiselect tags ── */
span[data-baseweb="tag"] {
  background: rgba(11,224,255,0.12) !important;
  border: 1px solid var(--border-glow) !important;
  border-radius: 20px !important;
  color: var(--text-accent) !important;
  font-size: 0.78rem !important;
  font-family: var(--font-mono) !important;
}

/* ── Code blocks ── */
code, pre {
  background: rgba(10,16,30,0.8) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: var(--radius-sm) !important;
  font-family: var(--font-mono) !important;
  color: var(--accent-buy) !important;
}

/* ── Columns spacing ── */
[data-testid="column"] { padding: 0 6px !important; }

/* ── Top bar progress ── */
[data-testid="stStatusWidget"] { display: none !important; }

/* ── Animations on appear ── */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
.block-container > div > div {
  animation: fadeInUp 0.45s ease-out both;
}

/* ── Pulse animation for alerts ── */
@keyframes alertPulse {
  0%, 100% { box-shadow: 0 0 0 0 var(--glow-sell); }
  50%       { box-shadow: 0 0 24px 4px var(--glow-sell); }
}
.qe-alert-pulse { animation: alertPulse 2s ease-in-out infinite; }

/* ── Hover focus effect: sharpen foreground, dim background ── */
.qe-hover-focus:hover { 
  filter: brightness(1.1) !important;
  z-index: 10 !important;
}

/* ── Spinner overlay ── */
[data-testid="stSpinner"] > div {
  border-color: var(--text-accent) transparent transparent transparent !important;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] input[type="checkbox"] {
  accent-color: var(--text-accent) !important;
}

/* ── Radio buttons ── */
[data-testid="stRadio"] label {
  text-transform: none !important;
  font-size: 0.85rem !important;
}

/* ── Sidebar divider ── */
[data-testid="stSidebar"] hr {
  background: linear-gradient(90deg, transparent, rgba(11,224,255,0.2), transparent) !important;
}

/* ── Main content top divider animation ── */
.qe-page-header {
  padding: 0 0 20px 0;
  border-bottom: 1px solid var(--border-glow);
  margin-bottom: 24px;
  position: relative;
}
.qe-page-header::after {
  content: '';
  position: absolute;
  bottom: -1px; left: 0;
  width: 80px; height: 2px;
  background: linear-gradient(90deg, var(--accent-buy), var(--text-accent));
  border-radius: 2px;
}

/* ── Neuroscience metric cards ── */
.qe-metric-card {
  background: var(--bg-glass);
  border: 1px solid var(--border-dim);
  border-radius: var(--radius);
  padding: 18px 20px;
  text-align: center;
  transition: var(--transition);
  backdrop-filter: blur(16px);
  position: relative;
  overflow: hidden;
}
.qe-metric-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent-buy), var(--text-accent), var(--accent-purp));
  opacity: 0;
  transition: opacity 0.3s;
}
.qe-metric-card:hover {
  border-color: var(--border-glow);
  box-shadow: 0 0 24px var(--glow-blue), 0 8px 30px rgba(0,0,0,0.4);
  transform: translateY(-3px);
}
.qe-metric-card:hover::before { opacity: 1; }
.qe-metric-label {
  color: var(--text-dim);
  font-size: 0.72rem;
  font-family: var(--font-mono);
  text-transform: uppercase;
  letter-spacing: 1.2px;
  margin-bottom: 8px;
}
.qe-metric-value {
  color: var(--text-accent);
  font-size: 1.6rem;
  font-weight: 700;
  font-family: var(--font-mono);
  line-height: 1;
}
.qe-metric-value.positive { color: var(--accent-buy) !important; }
.qe-metric-value.negative { color: var(--accent-sell) !important; }

/* ── Signal badge ── */
.qe-badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 0.72rem;
  font-family: var(--font-mono);
  font-weight: 600;
  letter-spacing: 0.8px;
  text-transform: uppercase;
}
.qe-badge-buy  { background: rgba(0,245,160,0.12); border: 1px solid var(--accent-buy);  color: var(--accent-buy); }
.qe-badge-sell { background: rgba(255,71,87,0.12);  border: 1px solid var(--accent-sell); color: var(--accent-sell); }
.qe-badge-hold { background: rgba(255,215,0,0.1);   border: 1px solid var(--accent-gold); color: var(--accent-gold); }

/* ── Regime environment indicator ── */
.qe-regime-bull {
  background: radial-gradient(circle at 50% 50%, rgba(0,245,160,0.08), transparent 70%);
  border: 1px solid rgba(0,245,160,0.2);
  border-radius: var(--radius);
  padding: 16px 20px;
  animation: regimePulse 4s ease-in-out infinite;
}
.qe-regime-bear {
  background: radial-gradient(circle at 50% 50%, rgba(255,71,87,0.08), transparent 70%);
  border: 1px solid rgba(255,71,87,0.2);
  border-radius: var(--radius);
  padding: 16px 20px;
  animation: regimePulseRed 4s ease-in-out infinite;
}
.qe-regime-sideways {
  background: radial-gradient(circle at 50% 50%, rgba(255,215,0,0.05), transparent 70%);
  border: 1px solid rgba(255,215,0,0.2);
  border-radius: var(--radius);
  padding: 16px 20px;
}
@keyframes regimePulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(0,245,160,0.15); }
  50%       { box-shadow: 0 0 30px 4px rgba(0,245,160,0.15); }
}
@keyframes regimePulseRed {
  0%, 100% { box-shadow: 0 0 0 0 rgba(255,71,87,0.15); }
  50%       { box-shadow: 0 0 30px 4px rgba(255,71,87,0.15); }
}

/* ── RL Agent trajectory ── */
.qe-agent-card {
  background: var(--bg-glass);
  border: 1px solid rgba(165,94,253,0.25);
  border-radius: var(--radius);
  padding: 18px;
  position: relative;
  overflow: hidden;
}
.qe-agent-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--accent-purp), var(--text-accent));
  animation: agentBeam 2s linear infinite;
  background-size: 200% 100%;
}
@keyframes agentBeam {
  0%   { background-position: -100% 0; }
  100% { background-position: 200% 0; }
}

/* ── Prediction confidence cloud ── */
.qe-confidence-cloud {
  background: rgba(11,224,255,0.04);
  border: 1px dashed rgba(11,224,255,0.2);
  border-radius: var(--radius);
  padding: 14px;
  font-size: 0.82rem;
  color: var(--text-dim);
  font-family: var(--font-mono);
}

/* ── Risk gravity pit ── */
.qe-risk-high {
  background: radial-gradient(ellipse at center, rgba(255,71,87,0.12) 0%, transparent 70%);
  border: 1px solid rgba(255,71,87,0.3);
  border-radius: var(--radius);
  padding: 14px;
}

/* ── Loading skeleton ── */
@keyframes skeletonPulse {
  0%   { opacity: 0.4; }
  50%  { opacity: 0.8; }
  100% { opacity: 0.4; }
}

/* ── Toast notification ── */
.qe-toast {
  position: fixed;
  bottom: 24px; right: 24px;
  background: var(--bg-glass);
  border: 1px solid var(--border-glow);
  border-radius: var(--radius);
  padding: 14px 20px;
  backdrop-filter: blur(20px);
  z-index: 9999;
  animation: toastSlideIn 0.4s cubic-bezier(0.25,0.46,0.45,0.94);
  min-width: 260px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5), 0 0 20px var(--glow-blue);
}
@keyframes toastSlideIn {
  from { transform: translateX(120%); opacity: 0; }
  to   { transform: translateX(0);   opacity: 1; }
}
</style>
"""

_JS = """
<script>
(function() {
  // ── Hover focus effect: contextual blur ──────────────────────────────────
  function initHoverFocus() {
    const charts = document.querySelectorAll('[data-testid="stPlotlyChart"]');
    charts.forEach(chart => {
      chart.addEventListener('mouseenter', () => {
        charts.forEach(c => { if (c !== chart) c.style.opacity = '0.55'; });
      });
      chart.addEventListener('mouseleave', () => {
        charts.forEach(c => { c.style.opacity = '1'; });
      });
    });
  }

  // ── Ripple on button click ───────────────────────────────────────────────
  function initRipple() {
    document.querySelectorAll('button').forEach(btn => {
      btn.addEventListener('click', function(e) {
        const ripple = document.createElement('span');
        const rect = btn.getBoundingClientRect();
        ripple.style.cssText = `
          position:absolute;left:${e.clientX-rect.left}px;top:${e.clientY-rect.top}px;
          width:0;height:0;border-radius:50%;
          background:rgba(11,224,255,0.35);
          transform:translate(-50%,-50%);
          animation:rippleAnim 0.6s ease-out forwards;
          pointer-events:none;
        `;
        if (!document.getElementById('qe-ripple-style')) {
          const s = document.createElement('style');
          s.id = 'qe-ripple-style';
          s.textContent = '@keyframes rippleAnim{to{width:200px;height:200px;opacity:0;}}';
          document.head.appendChild(s);
        }
        btn.style.position = 'relative';
        btn.style.overflow = 'hidden';
        btn.appendChild(ripple);
        setTimeout(() => ripple.remove(), 700);
      });
    });
  }

  // ── Particle burst on metric hover ──────────────────────────────────────
  function initMetricBurst() {
    document.querySelectorAll('[data-testid="stMetric"]').forEach(el => {
      el.addEventListener('mouseenter', function() {
        for (let i = 0; i < 5; i++) {
          const p = document.createElement('div');
          p.style.cssText = `
            position:absolute;width:4px;height:4px;border-radius:50%;
            background:rgba(11,224,255,${0.4+Math.random()*0.6});
            left:${40+Math.random()*20}%;top:${40+Math.random()*20}%;
            animation:particleBurst${i} 0.8s ease-out forwards;
            pointer-events:none;z-index:999;
          `;
          if (!document.getElementById('qe-particle-style')) {
            const s = document.createElement('style');
            s.id = 'qe-particle-style';
            s.textContent = 
              [0,1,2,3,4].map(n =>
                `@keyframes particleBurst${n}{to{transform:translate(${(Math.random()-0.5)*60}px,${-40-Math.random()*40}px);opacity:0;}}`
              ).join('');
            document.head.appendChild(s);
          }
          el.style.position = 'relative';
          el.appendChild(p);
          setTimeout(() => p.remove(), 900);
        }
      });
    });
  }

  // ── Run after Streamlit renders ──────────────────────────────────────────
  function init() {
    initHoverFocus();
    initRipple();
    initMetricBurst();
  }

  // Retry until DOM is ready
  let attempts = 0;
  const interval = setInterval(() => {
    if (document.querySelector('[data-testid="stMain"]') || ++attempts > 20) {
      clearInterval(interval);
      init();
      // Re-run on navigation
      const observer = new MutationObserver(() => {
        setTimeout(init, 300);
      });
      observer.observe(document.body, { childList: true, subtree: true });
    }
  }, 300);
})();
</script>
"""


def apply_quantedge_theme():
    """Inject the full QuantEdge cognitive UI theme into the current page."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(_JS,  unsafe_allow_html=True)


def qe_metric_cards(metrics: dict, cols: int = 4) -> None:
    """
    Render a row of neuroscience-styled metric cards.
    metrics: dict of {label: value_string}
    """
    positive_keywords = ["cagr", "sharpe", "sortino", "return", "win", "ic"]
    negative_keywords = ["drawdown", "var", "cvar", "loss", "vol"]

    cards_html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0;">'
    for label, value in metrics.items():
        lk = label.lower()
        css_class = ""
        if any(k in lk for k in positive_keywords):
            css_class = "positive"
        elif any(k in lk for k in negative_keywords):
            css_class = "negative"

        cards_html += f"""
        <div class="qe-metric-card">
          <div class="qe-metric-label">{label}</div>
          <div class="qe-metric-value {css_class}">{value}</div>
        </div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


def qe_section_header(title: str, subtitle: str = "") -> None:
    """Render a styled section header with optional subtitle."""
    html = f'''
    <div class="qe-page-header">
      <div style="color:var(--text-dim);font-family:var(--font-mono);
                  font-size:0.72rem;text-transform:uppercase;letter-spacing:2px;
                  margin-bottom:6px;">
        QuantEdge · Research Terminal
      </div>
      <div style="font-size:1.7rem;font-weight:700;
                  background:linear-gradient(135deg,#e8f4fd 0%,#0be0ff 60%,#a55efd 100%);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;letter-spacing:-0.5px;">
        {title}
      </div>
      {"" if not subtitle else f'<div style="color:var(--text-dim);font-size:0.85rem;margin-top:6px;font-family:var(--font-mono);">{subtitle}</div>'}
    </div>'''
    st.markdown(html, unsafe_allow_html=True)


def qe_regime_box(regime: str, recommendation: str = "") -> None:
    """Render a regime environment indicator box."""
    regime_lower = regime.lower()
    if "bull" in regime_lower:
        css = "qe-regime-bull"
        icon = "📈"
        color = "var(--accent-buy)"
    elif "bear" in regime_lower:
        css = "qe-regime-bear"
        icon = "📉"
        color = "var(--accent-sell)"
    else:
        css = "qe-regime-sideways"
        icon = "↔"
        color = "var(--accent-gold)"

    html = f'''
    <div class="{css}" style="margin:12px 0;">
      <div style="font-size:0.72rem;font-family:var(--font-mono);text-transform:uppercase;
                  letter-spacing:1.5px;color:var(--text-dim);margin-bottom:8px;">
        Current Market Regime
      </div>
      <div style="font-size:1.3rem;font-weight:700;color:{color};">
        {icon} {regime}
      </div>
      {"" if not recommendation else f'<div style="margin-top:10px;font-size:0.84rem;color:var(--text-dim);font-family:var(--font-mono);">{recommendation}</div>'}
    </div>'''
    st.markdown(html, unsafe_allow_html=True)


def apply_plotly_theme(fig, title: str = "", height: int = 500):
    """Apply the QuantEdge dark theme to any Plotly figure."""
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=height,
        title=dict(text=title, font=dict(size=15, color="#e8f4fd"), x=0.01),
    )
    return fig
