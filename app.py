import streamlit as st
import matplotlib.pyplot as plt
from portfolio_main import fetch_prices, optimize_markowitz, optimize_litterman
from incrementalvar import compute_incremental_var

st.title("📊 Portfolio Analysis Dashboard")

# --- Fetch prices
prices, returns = fetch_prices()

# --- Compute portfolios
w_markowitz = optimize_markowitz(returns)
w_litterman = optimize_litterman(returns)

# --- Bar plot of weights comparison
st.subheader("Portfolio Weights Comparison")
fig, ax = plt.subplots(figsize=(12, 6))
w_markowitz.plot(kind="bar", alpha=0.6, label="Markowitz", ax=ax)
w_litterman.plot(kind="bar", alpha=0.6, label="Black-Litterman", ax=ax)
ax.set_ylabel("Weight")
ax.set_title("Weights per Asset by Approach")
ax.legend()
st.pyplot(fig)

# --- Compute IVaR for Markowitz
st.subheader("Incremental VaR (Markowitz Portfolio)")
inc_var_df, port_daily, weights_full = compute_incremental_var()
st.dataframe(inc_var_df)

# --- Bubble plot for IVaR
st.subheader("IVaR Bubble Plot")
fig2, ax2 = plt.subplots(figsize=(14, 8))
colors = inc_var_df["Incremental_VaR"].apply(lambda x: "red" if x > 0 else "green")
y_pos = range(len(inc_var_df))
ax2.scatter(inc_var_df["Incremental_VaR"], y_pos,
            s=inc_var_df["Weight_in_Full"]*10000, alpha=0.7, c=colors, edgecolors="black")
for i, row in inc_var_df.iterrows():
    offset = 0.02 if row["Incremental_VaR"] >= 0 else -0.02
    ha = "left" if row["Incremental_VaR"] >= 0 else "right"
    ax2.text(row["Incremental_VaR"] + offset, i, row["Ticker"], va="center", ha=ha)
ax2.axvline(0, color="black", linestyle="--")
ax2.set_yticks(y_pos)
ax2.set_yticklabels(inc_var_df["Ticker"])
ax2.set_xlabel("Incremental VaR")
ax2.set_title("Incremental VaR per Asset (Markowitz Portfolio)")
st.pyplot(fig2)
