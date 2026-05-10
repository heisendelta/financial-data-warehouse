import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = (
    "postgresql+psycopg2://quant:quant@localhost:5432/warehouse"
)
engine = create_engine(DATABASE_URL)

ticker = 'AAPL'

df = pd.read_csv(f'data/raw/{ticker.lower()}.csv')
df = df.iloc[2:] # get rid of multiindex columns
df.columns = ['trade_date', 'close', 'high', 'low', 'open', 'volume']


df.to_sql(
    'raw_prices',
    engine,
    if_exists='append',
    index=False
)

print(f'Loaded data of [{ticker}] into raw_prices.')
