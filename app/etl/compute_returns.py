import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql+psycopg2://quant:quant@localhost:5432/warehouse"
)

query = """
SELECT
    a.ticker,
    p.trade_date,
    p.close
FROM daily_prices p
JOIN assets a
ON p.asset_id = a.asset_id
ORDER BY ticker, trade_date
"""

df = pd.read_sql(query, engine)

prices = df.pivot(
    index="trade_date",
    columns="ticker",
    values="close"
)

returns = prices.pct_change().dropna()

returns_long = returns.stack().reset_index()
returns_long.columns = ['trade_date', 'ticker', 'daily_return']

assets = pd.read_sql("SELECT * FROM assets", engine)

returns_long = returns_long.merge(assets, on='ticker')
returns_long = returns_long[['asset_id', 'trade_date', 'daily_return']]

returns_long.to_sql(
    'daily_returns',
    engine,
    if_exists='append',
    index=False
)
