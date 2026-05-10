import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = (
    "postgresql+psycopg2://quant:quant@localhost:5432/warehouse"
)
engine = create_engine(DATABASE_URL)

df = pd.read_csv(f'data/raw/all_prices.csv')
# df = df.iloc[2:] # get rid of multiindex columns

df = df.rename(columns={
    "Date": "trade_date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume"
})

df = df[[ "ticker", "trade_date", "open", "high", "low", "close", "volume" ]]

print(df.head())

df.to_sql(
    'raw_prices',
    engine,
    if_exists='append',
    index=False
)

print(f'Loaded data of all_prices.csv into raw_prices.')
