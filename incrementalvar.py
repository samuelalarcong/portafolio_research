import pandas as pd
from portfolio_main import fetch_prices, optimize_portfolio, calculate_var


def compute_incremental_var():
    prices, returns = fetch_prices()

    # ----- FULL PORTFOLIO -----
    weights_full = optimize_portfolio(returns)
    port_daily = (returns * weights_full).sum(axis=1)
    VaR_full = calculate_var(port_daily)

    results = []

    for ticker in weights_full.index:
        reduced = returns.drop(columns=[ticker])

        weights_ex = optimize_portfolio(reduced)
        port_ex = (reduced * weights_ex).sum(axis=1)
        VaR_ex = calculate_var(port_ex)

        results.append({
            "Ticker": ticker,
            "Weight": weights_full[ticker],
            "VaR_Full_%": VaR_full * 100,
            "VaR_Without_%": VaR_ex * 100,
            "Incremental_VaR_%": (VaR_full - VaR_ex) * 100
        })

    inc_var_df = pd.DataFrame(results).sort_values(
        "Incremental_VaR_%", ascending=False
    )

    return inc_var_df, port_daily, weights_full
