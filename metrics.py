import numpy as np
import pandas as pd

class PortfolioMetrics:
    def __init__(self, returns, weights=None, risk_free_rate=0.03):
        """
        Initializes the PortfolioMetrics class.

        Parameters:
        - returns (array-like): Daily returns of the portfolio.
        - weights (array-like, optional): Daily portfolio weights (N x M) for turnover calculation.
        - risk_free_rate (float): Annualized risk-free rate (default: 3%).
        """
        self.rets = np.asarray(returns, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64) if weights is not None else None
        self.rf = float(risk_free_rate)  # Ensure risk-free rate is float
        self.annualized_return = float(self.calc_annualized_total_return())
        self.annualized_volatility = float(self.calc_annualized_volatility())
        self.annualized_downside_dev = float(self.calc_annualized_downside_deviation())
        self.max_drawdown = float(self.calc_max_drawdown())
    
    def calc_annualized_total_return(self):
        cumulative_returns = (1 + self.rets).cumprod()
        total_return = cumulative_returns[-1] - 1
        return float((1 + total_return) ** (252 / len(self.rets)) - 1)

    def calc_annualized_volatility(self):
        return float(self.rets.std() * np.sqrt(252))

    def calc_annualized_downside_deviation(self):
        return float(self.rets[self.rets < 0].std() * np.sqrt(252))

    def calc_max_drawdown(self):
        cumulative_returns = (1 + self.rets).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        return float(drawdowns.min())

    def calc_annualized_calmar_ratio(self):
        return float(self.annualized_return / abs(self.max_drawdown)) if self.max_drawdown != 0 else np.nan

    def calc_annualized_sortino_ratio(self):
        return float((self.annualized_return - self.rf) / self.annualized_downside_dev) if self.annualized_downside_dev != 0 else np.nan

    def calc_annualized_sharpe_ratio(self):
        return float((self.annualized_return - self.rf) / self.annualized_volatility) if self.annualized_volatility != 0 else np.nan

    def calc_annualized_portfolio_turnover(self):
        if self.weights is None:
            return np.nan  # Return NaN if weights are not provided
        daily_turnover = pd.DataFrame(self.weights).diff().abs().sum(axis=1).mean()
        return float(daily_turnover * 252)  # Annualized turnover

    def calc_annualized_sharpe_ratio(self):
        return float((self.annualized_return - self.rf) / self.annualized_volatility) if self.annualized_volatility != 0 else np.nan

    
    def get_metrics(self):
        return {
            "Annualized Return": self.annualized_return,
            "Annualized Volatility": self.annualized_volatility,
            "Annualized Downside Deviation": self.annualized_downside_dev,
            "Max Drawdown": self.max_drawdown,
            "Annualized Calmar Ratio": self.calc_annualized_calmar_ratio(),
            "Annualized Sortino Ratio": self.calc_annualized_sortino_ratio(),
            "Annualized Sharpe Ratio": self.calc_annualized_sharpe_ratio(),
            "Annualized Portfolio Turnover": self.calc_annualized_portfolio_turnover(),
            "Annualized Sharpe Ratio": self.calc_annualized_sharpe_ratio(),
        }

