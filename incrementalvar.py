import pandas as pd
from portfolio_main import optimize_portfolio, calculate_var, compute_portfolio_daily_returns, GAMMA

def compute_incremental_var():
    # Get full portfolio
    port_daily, weights_full, returns = compute_portfolio_daily_returns()
    tickers_used = returns.columns.tolist()

    # Full portfolio VaR
    VaR_full = calculate_var(weights_full, tickers_used, returns)

    incremental_var_results = []
    active_tickers = weights_full[weights_full > 1e-6].index.tolist()

    for ticker in active_tickers:
        tickers_excluded = [t for t in tickers_used if t != ticker]
        weights_excluded = optimize_portfolio(tickers_excluded, returns)
        if weights_excluded is None:
            continue
        VaR_excluded = calculate_var(weights_excluded, tickers_excluded, returns)
        incremental_var = VaR_full - VaR_excluded
        incremental_var_results.append({
            "Ticker": ticker,
            "Weight_in_Full": weights_full[ticker],
            "VaR_Full": VaR_full,
            "VaR_Without": VaR_excluded,
            "Incremental_VaR": incremental_var
        })

    inc_var_df = pd.DataFrame(incremental_var_results).sort_values("Incremental_VaR", ascending=False)
    return inc_var_df, port_daily, weights_full
