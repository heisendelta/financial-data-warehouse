import yfinance as yf
import pandas as pd

tickers = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOG']

all_data = []

for ticker in tickers:
    print(f'Downloading {ticker}')

    df = yf.download(ticker, period='2y')
    df = df.reset_index()
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

    df['ticker'] = ticker

    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)
combined_df.to_csv('data/raw/all_prices.csv', index=False)

print(combined_df.head())
