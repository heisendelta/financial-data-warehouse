from sqlalchemy import create_engine

from app.portfolio.strategies.minimum_variance import (
    MinimumVariancePortfolio
)
portfolio = MinimumVariancePortfolio

DATABASE_URL = (
    "postgresql+psycopg2://quant:quant@localhost:5432/warehouse"
)

engine = create_engine(DATABASE_URL)

strategy = portfolio(engine)

results = strategy.run()

print(results)
