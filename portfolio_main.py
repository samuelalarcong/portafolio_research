import os
import pandas as pd
import numpy as np
import cvxpy as cp
from sqlalchemy import create_engine, text

# ---------- CONFIG ----------
TARGET_ASSETS = [
    "AVGO", "267260-KR", "APP", "3017", "MA", "NVDA",
    "META", "WISE", "TW", "TTWO", "SNPS", "AAPL",
    "PANW", "CASH-KRW", "CASH-USD",
]
MIN_HISTORY_DAYS = 504
CONF_LEVEL = 0.95
GAMMA = 5.0

# ---------- DB CONNECTION ----------
def get_engine():
    DB_HOST = os.getenv("DB_HOST", "54.90.6.56")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    DB_NAME = os.getenv("DB_NAME", "portfolio_db")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASS = os.getenv("DB_PASSWORD", "DevelopSql2023!")
    return create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}", pool_pre_ping=True)

# ---------- FETCH PRICES ----------
def fetch_prices():
    engine = get_engine()
    with engine.begin() as conn:
        query = """
            SELECT symbol, date, close_price
            FROM security_prices
            WHERE symbol = ANY(:syms)
            ORDER BY date
        """
        prices_raw = pd.read_sql(text(query), conn, params={"syms": TARGET_ASSETS}, parse_dates=["date"])
    prices = prices_raw.pivot(index="date", columns="symbol", values="close_price")
    prices = prices.sort_index().ffill().dropna(how="any")
    # Keep only assets with enough history
    enough = prices.count() >= MIN_HISTORY_DAYS
    prices = prices.loc[:, enough]
    returns = prices.pct_change().dropna()
    return prices, returns

# ---------- OPTIMIZE PORTFOLIO ----------
def optimize_markowitz(returns, gamma=GAMMA):
    mu = returns.mean().values * 252
    Sigma = np.cov(returns.values, rowvar=False) * 252
    n = len(mu)
    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - gamma * cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)
    weights = np.maximum(w.value, 0)
    weights /= weights.sum()
    return pd.Series(weights, index=returns.columns)

# ---------- SIMPLE BLACK-LITTERMAN ----------
def optimize_litterman(returns, tau=0.05):
    # Simplified: shrink expected returns toward mean (neutral view)
    mu = returns.mean().values * 252
    mu_bl = tau * mu + (1 - tau) * np.mean(mu)
    Sigma = np.cov(returns.values, rowvar=False) * 252
    n = len(mu_bl)
    w = cp.Variable(n)
    objective = cp.Maximize(mu_bl @ w - GAMMA * cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)
    weights = np.maximum(w.value, 0)
    weights /= weights.sum()
    return pd.Series(weights, index=returns.columns)
