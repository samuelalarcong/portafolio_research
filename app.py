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

# --- Horizontal bar plots side by side
st.subheader("Portfolio Weights Comparison (Markowitz vs Black-Litterman)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

# Left: Markowitz
w_markowitz.sort_values().plot(kind="barh", ax=ax1, color="skyblue")
ax1.set_title("Markowitz Portfolio")
ax1.set_xlabel("Weight")
ax1.grid(True, linestyle="--", alpha=0.3)

# Right: Black-Litterman
w_litterman.sort_values().plot(kind="barh", ax=ax2, color="lightgreen")
ax2.set_title("Black-Litterman Portfolio")
ax2.set_xlabel("Weight")
ax2.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# --- Compute IVaR for Markowitz
st.subheader("Incremental VaR (Markowitz Portfolio)")
inc_var_df, port_daily, weights_full = compute_incremental_var()
st.dataframe(inc_var_df)

# --- Bubble plot for IVaR
st.subheader("IVaR Bubble Plot")
n_tickers = len(inc_var_df)
fig2, ax2 = plt.subplots(figsize=(max(10, n_tickers*0.5), 8))

colors = inc_var_df["Incremental_VaR"].apply(lambda x: "red" if x > 0 else "green")
y_pos = range(n_tickers)

# Adjust bubble size
bubble_sizes = inc_var_df["Weight_in_Full"] * 50000 / n_tickers

ax2.scatter(inc_var_df["Incremental_VaR"], y_pos,
            s=bubble_sizes, alpha=0.7, c=colors, edgecolors="black")

for i, row in inc_var_df.iterrows():
    offset = 0.02 if row["Incremental_VaR"] >= 0 else -0.02
    ha = "left" if row["Incremental_VaR"] >= 0 else "right"
    ax2.text(row["Incremental_VaR"] + offset, i, row["Ticker"], va="center", ha=ha)

ax2.axvline(0, color="black", linestyle="--")
ax2.set_yticks(y_pos)
ax2.set_yticklabels(inc_var_df["Ticker"])
ax2.set_xlabel("Incremental VaR")
ax2.set_title("Incremental VaR per Asset (Markowitz Portfolio)")
plt.tight_layout()
st.pyplot(fig2)




