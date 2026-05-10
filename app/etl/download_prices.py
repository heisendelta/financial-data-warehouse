import yfinance as yf

ticker = 'AAPL'
df = yf.download(ticker, period='1y')

print(df.head())

df.to_csv(f'data/raw/{ticker.lower()}.csv')
