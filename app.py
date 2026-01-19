import matplotlib
matplotlib.use("Agg")

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from incrementalvar import compute_incremental_var

st.set_page_config(page_title="Incremental VaR Dashboard", layout="wide")

st.title("📊 Incremental VaR – Markowitz Portfolio")

with st.spinner("Computing portfolio & incremental VaR..."):
    inc_var_df, port_daily, weights = compute_incremental_var()

# ---------- TABLE ----------
st.subheader("Incremental VaR by Asset")
st.dataframe(
    inc_var_df.style.format({
        "Weight": "{:.2%}",
        "VaR_Full_%": "{:.2f}",
        "VaR_Without_%": "{:.2f}",
        "Incremental_VaR_%": "{:.2f}",
    }),
    use_container_width=True
)

# ---------- BUBBLE PLOT ----------
st.subheader("Incremental VaR Contribution")

fig, ax = plt.subplots(figsize=(12, 8))

colors = inc_var_df["Incremental_VaR_%"].apply(
    lambda x: "red" if x > 0 else "green"
)

ax.scatter(
    inc_var_df["Incremental_VaR_%"],
    range(len(inc_var_df)),
    s=inc_var_df["Weight"] * 6000,
    c=colors,
    alpha=0.7,
    edgecolors="black"
)

ax.set_yticks(range(len(inc_var_df)))
ax.set_yticklabels(inc_var_df["Ticker"])
ax.axvline(0, color="black", linestyle="--")

ax.set_xlabel("Incremental VaR (%)")
ax.set_title("Red = Risk Contributor | Green = Diversifier")

st.pyplot(fig)
