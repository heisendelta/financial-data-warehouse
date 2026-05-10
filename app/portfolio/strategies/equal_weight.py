import numpy as np

from app.portfolio.base import PortfolioStrategy


class EqualWeightPortfolio(PortfolioStrategy):

    def __init__(self, engine):

        super().__init__(
            engine,
            strategy_name="equal_weight"
        )

    def compute_weights(self, train_returns):

        n = train_returns.shape[1]

        return np.ones(n) / n