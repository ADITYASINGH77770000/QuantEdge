import streamlit as st

st.set_page_config(
    page_title="QuantEdge",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚡ QuantEdge")

    ticker = st.selectbox("Select Stock", ["GOOG", "NVDA", "AMZN", "META"])
    date_range = st.selectbox("Date Range", ["1M", "6M", "1Y", "5Y"])
    model = st.selectbox("Model", ["LSTM", "Linear", "XGBoost"])

    run = st.button("🚀 Run Analysis")

# ---------------- TOP METRICS ----------------
st.markdown("## 📊 Market Snapshot")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Price", "₹1,420", "+2.3%")
col2.metric("Volume", "3.2M")
col3.metric("Volatility", "1.8%")
col4.metric("Signal", "BUY")

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Technical", "🤖 AI Insights"])

with tab1:
    st.subheader("Price Chart")
    st.line_chart([1, 2, 3, 4, 5])

with tab2:
    st.subheader("Indicators")
    st.line_chart([5, 4, 3, 2, 1])

with tab3:
    st.subheader("AI Assistant")
    st.chat_message("assistant").write(
        "📊 Market looks bullish. Consider RSI + MACD confirmation."
    )