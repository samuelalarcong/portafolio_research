import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from incrementalvar import compute_incremental_var
from portfolio_main import CONF_LEVEL

# ---------- PAGE SETUP ----------
st.set_page_config(
    page_title="Portfolio Incremental VaR",
    layout="wide"
)

st.title("📊 Portfolio Incremental VaR Dashboard")

# ---------- COMPUTE DATA ----------
with st.spinner("Computing portfolio optimization and Incremental VaR..."):
    inc_var_df, port_daily, weights_full = compute_incremental_var()

# ---------- PORTFOLIO RISK METRICS ----------
losses = -port_daily
VaR = np.quantile(losses, CONF_LEVEL)
ES  = losses[losses >= VaR].mean()

st.subheader("📉 Portfolio Risk Metrics")
col1, col2, col3 = st.columns(3)

col1.metric(f"VaR {int(CONF_LEVEL*100)}%", f"{VaR:.2%}")
col2.metric(f"ES {int(CONF_LEVEL*100)}%", f"{ES:.2%}")
col3.metric("Observations", f"{len(port_daily)} days")

# ---------- INCREMENTAL VAR TABLE ----------
st.subheader("📋 Incremental VaR by Asset")

st.dataframe(
    inc_var_df.style.format({
        "Weight_in_Full": "{:.2%}",
        "VaR_Full": "{:.2%}",
        "VaR_Without": "{:.2%}",
        "Incremental_VaR": "{:.2%}",
    }),
    use_container_width=True
)

# ---------- BUBBLE PLOT ----------
st.subheader("🫧 Incremental VaR Bubble Plot")

inc_var_sorted = inc_var_df.sort_values("Incremental_VaR")

fig, ax = plt.subplots(figsize=(14, 8))

y_pos = np.arange(len(inc_var_sorted))
bubble_sizes = inc_var_sorted["Weight_in_Full"] * 6000

colors = inc_var_sorted["Incremental_VaR"].apply(
    lambda x: "red" if x > 0 else "green"
)

ax.scatter(
    inc_var_sorted["Incremental_VaR"],
    y_pos,
    s=bubble_sizes,
    c=colors,
    alpha=0.7,
    edgecolors="black",
    linewidth=1
)

# Labels
ax.set_yticks(y_pos)
ax.set_yticklabels(inc_var_sorted["Ticker"])

ax.axvline(0, linestyle="--", color="black", linewidth=1)

ax.set_xlabel("Incremental VaR")
ax.set_title("Incremental VaR Contribution\nRed = Risk Contributor | Green = Diversifier")

ax.grid(True, linestyle="--", alpha=0.4)

st.pyplot(fig)

# ---------- FOOTNOTE ----------
st.caption(
    "Incremental VaR is computed by re-optimizing the portfolio "
    "after removing each asset using a Markowitz mean–variance framework."
)
