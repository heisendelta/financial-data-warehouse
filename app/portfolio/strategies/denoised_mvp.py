from portfolio.base import PortfolioStrategy
from portfolio.helper import estimate_cov, denoise_cov, opt_port

class DenoisedMVP(PortfolioStrategy):

    def __init__(self, engine):

        super().__init__(
            engine,
            strategy_name='denoised_mvp'
        )

    def compute_weights(self, train_returns):

        cov = estimate_cov(train_returns)

        q = train_returns.shape[0] / train_returns.shape[1]

        cov_d = denoise_cov(
            cov,
            q=q,
            b_width=0.01,
            type=1
        )

        return opt_port(cov_d)
