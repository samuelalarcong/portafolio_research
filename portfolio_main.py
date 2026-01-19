import os
import pandas as pd
import numpy as np
import cvxpy as cp
from sqlalchemy import create_engine, text

# ---------- CONFIG ----------
TARGET_ASSETS = [
    "AVGO","267260-KR","APP","3017","MA","NVDA","META","WISE",
    "TW","TTWO","SNPS","AAPL","PANW","CASH-KRW","CASH-USD",
]

MIN_HISTORY_DAYS = 504       # Require 2 years of data
CONF_LEVEL = 0.95
GAMMA = 5.0

# ---------- FUNCTIONS ----------

def fetch_prices(target_assets=TARGET_ASSETS, start_date=None, end_date=None, min_history_days=MIN_HISTORY_DAYS):
    """Fetch price data from DB for target assets."""
    if start_date is None:
        start_date = pd.Timestamp.today() - pd.DateOffset(years=2)

    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASSWORD")

    if not all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS]):
        raise RuntimeError("Missing DB credentials.")

    engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}", pool_pre_ping=True)

    with engine.begin() as conn:
        query = """
            SELECT symbol, date, close_price
            FROM security_prices
            WHERE symbol = ANY(:syms)
              AND date >= :start_date
        """
        if end_date:
            query += " AND date <= :end_date"
        query += " ORDER BY date"

        prices_raw = pd.read_sql(text(query),
                                 conn,
                                 params={"syms": target_assets, "start_date": start_date, "end_date": end_date},
                                 parse_dates=["date"])
    if prices_raw.empty:
        raise RuntimeError("No price history found.")

    prices = prices_raw.pivot(index="date", columns="symbol", values="close_price").sort_index().ffill().dropna(how="any")

    enough = prices.count() >= min_history_days
    prices = prices.loc[:, enough]
    if prices.shape[1] == 0:
        raise RuntimeError("None of the selected assets have enough history.")

    returns = prices.pct_change().dropna()
    return prices, returns

def optimize_portfolio(tickers_list, returns_df, gamma=GAMMA):
    """Markowitz mean-variance optimization"""
    rets_subset = returns_df[tickers_list]
    mu = rets_subset.mean().values * 252
    Sigma = np.cov(rets_subset.values, rowvar=False) * 252
    n = len(mu)

    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - gamma * cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None

    weights = np.maximum(w.value, 0)
    weights = weights / weights.sum()
    return weights

def calculate_var(weights, tickers_list, returns_df, conf_level=CONF_LEVEL):
    """Calculate historical VaR for given weights"""
    w_series = pd.Series(weights, index=tickers_list)
    port_daily = (returns_df[tickers_list] * w_series).sum(axis=1)
    losses = -port_daily
    VaR = np.quantile(losses, conf_level)
    return VaR

def compute_portfolio_daily_returns():
    """Compute full optimized portfolio daily returns"""
    prices, returns = fetch_prices()
    weights = optimize_portfolio(returns.columns.tolist(), returns)
    if weights is None:
        raise RuntimeError("Optimization failed.")
    w_series = pd.Series(weights, index=returns.columns)
    port_daily = (returns * w_series).sum(axis=1)
    return port_daily, w_series, returns
