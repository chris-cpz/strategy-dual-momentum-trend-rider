#!/usr/bin/env python3
"""
Dual Momentum Trend Rider - Trend_following, breakout Trading Strategy

Strategy Type: trend_following, breakout
Description: This strategy combines cross-sectional momentum with time-series trend following to capture persistent market inefficiencies across multiple asset classes.

We identify assets that have both:

Strong relative momentum (outperforming their peers), and

A positive price trend versus their own historical behavior (above moving average or similar trend filter).

The core belief is that assets with strong momentum and confirmed trends tend to continue their path due to institutional flow inertia, behavioral biases, and slow-moving macro forces.

Methodology:
Universe Selection:

Liquid assets (e.g., major equity indices, FX pairs, commodities, or liquid ETFs).

Filter out illiquid names based on minimum average daily volume.

Momentum Signal (Cross-Sectional):

Calculate 12-month total return (excluding last 1 month) for each asset.

Rank assets by momentum score.

Select the top 30% percentile for long exposure.

Trend Filter (Time-Series):

Apply a 200-day Simple Moving Average (SMA) filter.

Only go long if the asset is above its 200-day SMA (i.e., in a confirmed uptrend).

Position Sizing:

Equal-weight across all qualified assets (or use risk-parity weighting if desired).

Implement volatility scaling to normalize risk across positions.

Rebalancing Frequency:

Monthly (on the first trading day of each month).

Optional: Weekly signal check for extreme moves (stop-loss triggers).

Risk Management:

5% trailing stop-loss per position.

Max portfolio drawdown cap (e.g., 15%) for systematic de-leveraging.

Execution Logic:

Staggered limit orders around VWAP to minimize slippage.

Optional transaction cost model to avoid excessive turnover.

Optional Enhancements:
Alternative Momentum Horizons: Blend 3M and 6M momentum for signal robustness.

Machine Learning Overlay: Use a classifier to predict signal quality (optional for future iterations).

Factor Neutralization: Apply beta or sector neutralization depending on backtest results.
Created: 2025-08-29T17:15:34.916Z

WARNING: This is a template implementation. Thoroughly backtest before live trading.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DualMomentumTrendRiderStrategy:
    """
    Dual Momentum Trend Rider Implementation
    
    Strategy Type: trend_following, breakout
    Risk Level: Monitor drawdowns and position sizes carefully
    """
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.positions = {}
        self.performance_metrics = {}
        logger.info(f"Initialized Dual Momentum Trend Rider strategy")
        
    def get_default_config(self):
        """Default configuration parameters"""
        return {
            'max_position_size': 0.05,  # 5% max position size
            'stop_loss_pct': 0.05,      # 5% stop loss
            'lookback_period': 20,       # 20-day lookback
            'rebalance_freq': 'daily',   # Rebalancing frequency
            'transaction_costs': 0.001,  # 0.1% transaction costs
        }
    
    def load_data(self, symbols, start_date, end_date):
        """Load market data for analysis"""
        try:
            import yfinance as yf
            data = yf.download(symbols, start=start_date, end=end_date)
            logger.info(f"Loaded data for {len(symbols)} symbols")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

# =============================================================================
# USER'S STRATEGY IMPLEMENTATION
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Sample data generation for demonstration
def generate_sample_data(num_assets=8, num_days=800, seed=42):
    np.random.seed(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=num_days)
    tickers = ["Asset_%d" % i for i in range(num_assets)]
    prices = np.cumprod(1 + 0.0005 * np.random.randn(num_days, num_assets) + 0.0002, axis=0) * 100
    volumes = np.abs(1000000 + 100000 * np.random.randn(num_days, num_assets))
    price_df = pd.DataFrame(prices, index=dates, columns=tickers)
    volume_df = pd.DataFrame(volumes, index=dates, columns=tickers)
    return price_df, volume_df

# Dual Momentum Trend Rider Strategy Class
class DualMomentumTrendRider:
    def __init__(self, price_df, volume_df, 
                 min_avg_volume=500000, 
                 momentum_lookback=252, 
                 momentum_skip=21, 
                 trend_lookback=200, 
                 top_pct=0.3, 
                 rebalance_freq="M", 
                 stop_loss_pct=0.05, 
                 max_drawdown_cap=0.15, 
                 vol_lookback=20, 
                 risk_free_rate=0.0):
        # Initialize parameters
        self.price_df = price_df
        self.volume_df = volume_df
        self.min_avg_volume = min_avg_volume
        self.momentum_lookback = momentum_lookback
        self.momentum_skip = momentum_skip
        self.trend_lookback = trend_lookback
        self.top_pct = top_pct
        self.rebalance_freq = rebalance_freq
        self.stop_loss_pct = stop_loss_pct
        self.max_drawdown_cap = max_drawdown_cap
        self.vol_lookback = vol_lookback
        self.risk_free_rate = risk_free_rate
        self.asset_list = price_df.columns.tolist()
        self.signals = None
        self.positions = None
        self.returns = None
        self.portfolio_value = None
        self.max_portfolio_value = None
        self.stop_loss_prices = None
        self.drawdown = None

    def universe_selection(self):
        # Filter assets by minimum average daily volume
        avg_vol = self.volume_df.rolling(window=60, min_periods=1).mean()
        liquid_assets = avg_vol.columns[(avg_vol.iloc[-1] >= self.min_avg_volume)].tolist()
        logging.info("Selected liquid assets: %s" % liquid_assets)
        return liquid_assets

    def compute_momentum(self, assets):
        # Calculate 12-month total return (excluding last 1 month)
        shifted = self.price_df[assets].shift(self.momentum_skip)
        past = self.price_df[assets].shift(self.momentum_skip + self.momentum_lookback)
        momentum = (shifted / past) - 1
        return momentum

    def compute_trend(self, assets):
        # Compute 200-day SMA
        sma = self.price_df[assets].rolling(window=self.trend_lookback, min_periods=1).mean()
        trend = self.price_df[assets] > sma
        return trend

    def compute_volatility(self, assets):
        # Compute rolling volatility for volatility scaling
        returns = self.price_df[assets].pct_change()
        vol = returns.rolling(window=self.vol_lookback, min_periods=1).std()
        return vol

    def generate_signals(self):
        # Generate signals for each rebalance date
        rebalance_dates = self.price_df.resample(self.rebalance_freq).first().index
        signals = pd.DataFrame(index=self.price_df.index, columns=self.asset_list)
        stop_loss_prices = pd.DataFrame(index=self.price_df.index, columns=self.asset_list)
        last_stop_loss = {}
        for date in rebalance_dates:
            if date not in self.price_df.index:
                continue
            # Universe selection
            assets = self.universe_selection()
            # Momentum
            momentum = self.compute_momentum(assets)
            mom_today = momentum.loc[date]
            # Rank by momentum
            mom_rank = mom_today.rank(ascending=False, method="min")
            top_n = int(np.ceil(len(assets) * self.top_pct))
            top_assets = mom_rank[mom_rank <= top_n].index.tolist()
            # Trend filter
            trend = self.compute_trend(assets)
            trend_today = trend.loc[date]
            qualified = [a for a in top_assets if trend_today[a]]
            # Set signals
            signals.loc[date, :] = 0
            for a in qualified:
                signals.loc[date, a] = 1
                # Set stop-loss price
                price = self.price_df.loc[date, a]
                stop_loss_prices.loc[date, a] = price * (1 - self.stop_loss_pct)
                last_stop_loss[a] = price * (1 - self.stop_loss_pct)
            # Carry forward stop-loss for open positions
            for a in self.asset_list:
                if signals.loc[date, a] != 1 and a in last_stop_loss:
                    stop_loss_prices.loc[date, a] = last_stop_loss[a]
        # Forward fill signals and stop-loss prices
        signals = signals.fillna(method="ffill").fillna(0)
        stop_loss_prices = stop_loss_prices.fillna(method="ffill")
        self.signals = signals
        self.stop_loss_prices = stop_loss_prices

    def apply_stop_loss(self):
        # Apply 5% trailing stop-loss per position
        signals = self.signals.copy()
        stop_loss_prices = self.stop_loss_prices.copy()
        for a in self.asset_list:
            in_position = False
            for i in range(1, len(signals)):
                if signals.iloc[i, signals.columns.get_loc(a)] == 1:
                    if not in_position:
                        # Entering position, set stop-loss
                        stop_loss_prices.iloc[i, signals.columns.get_loc(a)] = self.price_df.iloc[i, self.price_df.columns.get_loc(a)] * (1 - self.stop_loss_pct)
                        in_position = True
                    else:
                        # Update trailing stop-loss if new high
                        price = self.price_df.iloc[i, self.price_df.columns.get_loc(a)]
                        prev_price = self.price_df.iloc[i-1, self.price_df.columns.get_loc(a)]
                        prev_stop = stop_loss_prices.iloc[i-1, signals.columns.get_loc(a)]
                        new_stop = max(prev_stop, price * (1 - self.stop_loss_pct))
                        stop_loss_prices.iloc[i, signals.columns.get_loc(a)] = new_stop
                        # Check stop-loss breach
                        if price < prev_stop:
                            signals.iloc[i, signals.columns.get_loc(a)] = 0
                            in_position = False
                else:
                    in_position = False
        self.signals = signals
        self.stop_loss_prices = stop_loss_prices

    def position_sizing(self):
        # Equal-weighted with volatility scaling
        positions = pd.DataFrame(index=self.price_df.index, columns=self.asset_list)
        vol = self.compute_volatility(self.asset_list)
        target_vol = 0.02 # Target 2% volatility per position
        for i in range(len(self.price_df)):
            date = self.price_df.index[i]
            sig = self.signals.iloc[i]
            if sig.sum() == 0:
                positions.iloc[i] = 0
                continue
            # Volatility scaling
            inv_vol = 1 / (vol.iloc[i] + 1e-8)
            inv_vol = inv_vol * sig
            if inv_vol.sum() == 0:
                weights = sig / sig.sum()
            else:
                weights = inv_vol / inv_vol.sum()
            weights = weights * sig
            positions.iloc[i] = weights
        self.positions = positions.fillna(0)

    def backtest(self):
        # Calculate daily returns
        returns = self.price_df.pct_change().fillna(0)
        strat_returns = (self.positions.shift(1) * returns).sum(axis=1)
        # Apply max portfolio drawdown cap
        portfolio_value = (1 + strat_returns).cumprod()
        max_portfolio_value = portfolio_value.cummax()
        drawdown = (portfolio_value - max_portfolio_value) / max_portfolio_value
        deleverage = drawdown < -self.max_drawdown_cap
        strat_returns[deleverage] = 0
        portfolio_value = (1 + strat_returns).cumprod()
        self.returns = strat_returns
        self.portfolio_value = portfolio_value
        self.max_portfolio_value = max_portfolio_value
        self.drawdown = drawdown

    def performance_metrics(self):
        # Calculate Sharpe ratio, max drawdown, CAGR
        ann_factor = 252
        mean_return = self.returns.mean() * ann_factor
        std_return = self.returns.std() * np.sqrt(ann_factor)
        sharpe = (mean_return - self.risk_free_rate) / (std_return + 1e-8)
        max_dd = self.drawdown.min()
        total_days = (self.portfolio_value.index[-1] - self.portfolio_value.index[0]).days
        cagr = (self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0]) ** (365.25 / total_days) - 1
        return {"Sharpe": sharpe, "Max Drawdown": max_dd, "CAGR": cagr}

    def run(self):
        try:
            logging.info("Generating signals...")
            self.generate_signals()
            logging.info("Applying stop-loss logic...")
            self.apply_stop_loss()
            logging.info("Calculating position sizing...")
            self.position_sizing()
            logging.info("Running backtest...")
            self.backtest()
            metrics = self.performance_metrics()
            print("Performance Metrics:")
            for k, v in metrics.items():
                print("%s: %.4f" % (k, v))
            # Plot results
            plt.figure(figsize=(12,6))
            plt.plot(self.portfolio_value.index, self.portfolio_value.values, label="Strategy Equity Curve")
            plt.title("Dual Momentum Trend Rider - Equity Curve")
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value")
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception as e:
            logging.error("Error in strategy run: %s" % str(e))

# Main execution block
if __name__ == "__main__":
    # Generate sample data
    price_df, volume_df = generate_sample_data(num_assets=8, num_days=800)
    # Instantiate and run strategy
    strategy = DualMomentumTrendRider(price_df, volume_df)
    strategy.run()

# =============================================================================
# STRATEGY EXECUTION AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Example usage and testing
    strategy = DualMomentumTrendRiderStrategy()
    print(f"Strategy '{strategyName}' initialized successfully!")
    
    # Example data loading
    symbols = ['SPY', 'QQQ', 'IWM']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    print(f"Loading data for symbols: {symbols}")
    data = strategy.load_data(symbols, start_date, end_date)
    
    if data is not None:
        print(f"Data loaded successfully. Shape: {data.shape}")
        print("Strategy ready for backtesting!")
    else:
        print("Failed to load data. Check your internet connection.")
