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

