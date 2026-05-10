from portfolio.base import PortfolioStrategy
from portfolio.helper import opt_port


class MinimumVariancePortfolio(PortfolioStrategy):

    def __init__(self, engine):

        super().__init__(
            engine,
            strategy_name="minimum_variance"
        )

    def compute_weights(self, train_returns):

        cov = train_returns.cov().values

        return opt_port(cov).flatten()