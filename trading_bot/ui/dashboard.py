import streamlit as st

st.set_page_config(page_title="Trading Bot", page_icon="📈", layout="wide")

st.title("🤖 Bot de Trading - Dashboard")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Capital", "$10,000")
with col2:
    st.metric("Trades Hoje", "0")
with col3:
    st.metric("PnL", "$0.00")

st.info("✨ Dashboard em desenvolvimento. Use run_bot.py para executar o bot.")