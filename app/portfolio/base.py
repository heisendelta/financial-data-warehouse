import numpy as np
import pandas as pd

from sqlalchemy import text


class PortfolioStrategy:

    def __init__(
        self,
        engine,
        strategy_name,
        train_window=252,
        test_window=21,
    ):

        self.engine = engine

        self.strategy_name = strategy_name

        self.train_window = train_window
        self.test_window = test_window

        self.returns = None

        self.portfolio_returns = None
        self.weights_history = []

        self.metrics = {}

    # ---------------------------------------------------
    # DATA LOADING
    # ---------------------------------------------------

    def load_returns(self):

        query = """
        SELECT
            a.ticker,
            r.trade_date,
            r.daily_return

        FROM daily_returns r

        JOIN assets a
        ON r.asset_id = a.asset_id

        ORDER BY trade_date
        """

        df = pd.read_sql(query, self.engine)

        returns = df.pivot(
            index="trade_date",
            columns="ticker",
            values="daily_return"
        )

        returns = returns.dropna(axis=1)

        self.returns = returns

        return returns

    # ---------------------------------------------------
    # STRATEGY METHOD (overridden by children)
    # ---------------------------------------------------

    def compute_weights(self, train_returns):

        raise NotImplementedError

    # ---------------------------------------------------
    # BACKTEST ENGINE
    # ---------------------------------------------------

    def backtest(self):

        if self.returns is None:
            self.load_returns()

        returns = self.returns

        portfolio_returns = []

        for start in range(
            0,
            len(returns) - self.train_window - self.test_window,
            self.test_window
        ):

            train = returns.iloc[
                start : start + self.train_window
            ]

            test = returns.iloc[
                start + self.train_window :
                start + self.train_window + self.test_window
            ]

            w = self.compute_weights(train)

            self.weights_history.append(w)

            step_returns = test.values @ w

            portfolio_returns.extend(step_returns)

        self.portfolio_returns = np.array(portfolio_returns)

        return self.portfolio_returns

    # ---------------------------------------------------
    # METRICS
    # ---------------------------------------------------

    def compute_metrics(self):

        r = self.portfolio_returns

        sharpe = np.mean(r) / np.std(r)

        volatility = np.std(r)

        cumulative_return = np.prod(1 + r) - 1

        self.metrics = { # serialize as native python float
            "sharpe": float(sharpe),
            "volatility": float(volatility),
            "cumulative_return": float(cumulative_return),
        }

        return self.metrics

    # ---------------------------------------------------
    # SAVE RUN
    # ---------------------------------------------------

    def save_run(self):

        query = text("""
        INSERT INTO portfolio_runs (
            strategy,
            train_window,
            test_window,
            sharpe,
            volatility,
            cumulative_return
        )
        VALUES (
            :strategy,
            :train_window,
            :test_window,
            :sharpe,
            :volatility,
            :cumulative_return
        )
        RETURNING run_id
        """)

        with self.engine.begin() as conn:

            run_id = conn.execute(
                query,
                {
                    "strategy": self.strategy_name,
                    "train_window": self.train_window,
                    "test_window": self.test_window,
                    "sharpe": self.metrics["sharpe"],
                    "volatility": self.metrics["volatility"],
                    "cumulative_return": self.metrics["cumulative_return"],
                }
            ).scalar()

        return run_id

    # ---------------------------------------------------
    # FULL PIPELINE
    # ---------------------------------------------------

    def run(self):

        self.load_returns()

        self.backtest()

        self.compute_metrics()

        run_id = self.save_run()

        return {
            "run_id": run_id,
            "metrics": self.metrics,
        }
