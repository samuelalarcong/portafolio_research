import pandas as pd
import numpy as np
from portfolio_main import fetch_prices, optimize_markowitz, GAMMA

CONF_LEVEL = 0.95

def compute_incremental_var():
    prices, returns = fetch_prices()
    weights_full = optimize_markowitz(returns, gamma=GAMMA)
    port_daily = (returns * weights_full).sum(axis=1)
    
    VaR_full = np.quantile(-port_daily, CONF_LEVEL)
    
    incremental_var_results = []
    active_tickers = weights_full[weights_full > 1e-6].index.tolist()
    
    for ticker in active_tickers:
        tickers_excluded = [t for t in active_tickers if t != ticker]
        weights_excluded = optimize_markowitz(returns[tickers_excluded], gamma=GAMMA)
        VaR_excluded = np.quantile(-(returns[tickers_excluded] * weights_excluded).sum(axis=1), CONF_LEVEL)
        incremental_var_results.append({
            "Ticker": ticker,
            "Weight_in_Full": weights_full[ticker],
            "VaR_Full": VaR_full,
            "VaR_Without": VaR_excluded,
            "Incremental_VaR": VaR_full - VaR_excluded
        })
    inc_var_df = pd.DataFrame(incremental_var_results).sort_values("Incremental_VaR", ascending=False)
    return inc_var_df, port_daily, weights_full
