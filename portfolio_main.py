import os
import numpy as np
import pandas as pd
import cvxpy as cp
from sqlalchemy import create_engine, text

# ---------- CONFIG ----------
TARGET_ASSETS = [
    "AVGO", "267260-KR", "APP", "3017", "MA", "NVDA", "META",
    "WISE", "TW", "TTWO", "SNPS", "AAPL", "PANW", "CASH-KRW", "CASH-USD"
]

MIN_HISTORY_DAYS = 504
CONF_LEVEL = 0.95
GAMMA = 5.0


def fetch_prices(start_date=None, end_date=None):
    DB_HOST = os.getenv("DB_HOST", "54.90.6.56")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "portfolio_db")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASS = os.getenv("DB_PASSWORD", "DevelopSql2023!")

    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        pool_pre_ping=True,
    )

    if start_date is None:
        start_date = pd.Timestamp.today() - pd.DateOffset(years=2)

    with engine.begin() as conn:
        query = """
            SELECT symbol, date, close_price
            FROM security_prices
            WHERE symbol = ANY(:syms)
              AND date >= :start_date
        """
        if end_date:
            query += " AND date <= :end_date"

        prices_raw = pd.read_sql(
            text(query),
            conn,
            params={"syms": TARGET_ASSETS, "start_date": start_date},
            parse_dates=["date"],
        )

    prices = (
        prices_raw
        .pivot(index="date", columns="symbol", values="close_price")
        .sort_index()
        .ffill()
        .dropna()
    )

    prices = prices.loc[:, prices.count() >= MIN_HISTORY_DAYS]
    returns = prices.pct_change().dropna()

    return prices, returns


def optimize_portfolio(returns_df, gamma=GAMMA):
    mu = returns_df.mean().values * 252
    Sigma = np.cov(returns_df.values, rowvar=False) * 252
    n = len(mu)

    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - gamma * cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1, w >= 0]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)

    weights = np.maximum(w.value, 0)
    weights /= weights.sum()

    return pd.Series(weights, index=returns_df.columns)


def calculate_var(port_daily, conf_level=CONF_LEVEL):
    losses = -port_daily
    return np.quantile(losses, conf_level)
