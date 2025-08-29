# DUAL MOMENTUM TREND RIDER

## Trend_following, breakout Trading Strategy

[![GitHub Stars](https://img.shields.io/github/stars/chris-cpz/strategy-dual-momentum-trend-rider?style=for-the-badge&logo=github)](https://github.com/chris-cpz/strategy-dual-momentum-trend-rider)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/downloads/)

**Description**: This strategy combines cross-sectional momentum with time-series trend following to capture persistent market inefficiencies across multiple asset classes.

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

**Strategy Type**: Trend_following, breakout | **Implementation Date**: 8/29/2025 | **Status**: Development

---

## Overview

This repository contains a trend_following, breakout trading strategy implementation based on quantitative analysis and systematic market research. The strategy is designed to identify and exploit market inefficiencies while maintaining robust risk management protocols.

### Key Features

- Quantitative signal generation using mathematical models
- Risk-adjusted portfolio construction with dynamic position sizing
- Real-time execution framework with transaction cost optimization
- Comprehensive performance analytics and reporting
- Multi-layered risk controls and position limits
- Extensive backtesting with statistical validation

### Target Performance Metrics

| Metric | Target |
|--------|--------|
| Sharpe Ratio | > 1.5 |
| Maximum Drawdown | < 15% |
| Volatility Target | 12-18% annualized |
| Rebalancing | Daily/Weekly |
| Minimum Capital | $100,000+ |

---


## Research Hypothesis

**Primary Hypothesis (H₁)**: The trend_following, breakout trading approach will generate risk-adjusted returns superior to passive market exposure.

---


## Theoretical Framework

### Market Efficiency & Behavioral Finance Foundation
The strategy exploits systematic deviations from market efficiency.

---


## Research Methodology

### Phase I: Strategy Development Framework
Comprehensive hypothesis formation and testing protocol.

---

## Implementation

### Prerequisites

- Python 3.8+
- 16GB+ RAM recommended
- Stable internet connection for data feeds

### Installation

```bash
# Clone repository
git clone https://github.com/chris-cpz/strategy-dual-momentum-trend-rider.git
cd strategy-dual-momentum-trend-rider

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file with your API keys:

```bash
# Required API Keys
ALPHA_VANTAGE_API_KEY=your_key_here
IEX_CLOUD_API_KEY=your_key_here
QUANDL_API_KEY=your_key_here
```

### Quick Start

```python
from strategy import DualMomentumTrendRiderStrategy

# Initialize strategy
strategy = DualMomentumTrendRiderStrategy()

# Load data and run backtest
data = strategy.load_data(['SPY', 'QQQ'], '2020-01-01', '2023-12-31')
results = strategy.backtest(data)
strategy.generate_report(results)
```

---

## Backtesting Framework

### Validation Process

- **Historical Analysis**: 10+ years of market data
- **Walk-Forward Testing**: 12-month optimization with 6-month validation
- **Monte Carlo Simulation**: 10,000+ scenarios for statistical significance
- **Stress Testing**: Performance during market crashes and volatility spikes

### Transaction Cost Modeling

- Realistic bid-ask spreads and market impact
- Dynamic slippage based on order size and liquidity
- Optimal execution strategies (TWAP, VWAP)

---


## Risk Assessment & Critical Limitations

### **MODEL RISK** 🔴
Overfitting and parameter instability concerns.

---


## Performance Monitoring Framework

| **Metric** | **Target** | **Calculation** |
|------------|-----------|----------------|
| **Sharpe Ratio** | > 1.0 | Risk-adjusted returns |

---


## Data Infrastructure Requirements

| **Data Category** | **Frequency** | **Provider** |
|-------------------|---------------|-------------|
| **Market Data** | Real-time/Daily | Multiple vendors |

---

## Technical Architecture

### System Components

```
Data Layer → Signal Generation → Portfolio Construction → Execution
    ↓              ↓                    ↓               ↓
Market Data    Feature Eng.      Risk Management   Order Management
Alt Data       ML Models         Position Sizing   Performance Tracking
Fundamentals   Signal Combine    Regime Detection  Reporting
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Data Processing | pandas, numpy | High-performance data manipulation |
| Machine Learning | scikit-learn | Advanced signal generation |
| Backtesting | zipline, pyfolio | Institutional backtesting |
| Execution | Interactive Brokers API | Real-time trading |
| Visualization | plotly, matplotlib | Performance reporting |

---

## Risk Management

### Risk Controls

- Position size limits (5% maximum per position)
- Stop-loss mechanisms (5% default)
- Volatility targeting
- Drawdown limits
- Sector concentration limits

### Performance Monitoring

- Real-time P&L tracking
- Risk metric calculations
- Performance attribution
- Automated alerts for limit breaches

---

## Academic References

1. **Jegadeesh, N., & Titman, S. (1993)**. "Returns to Buying Winners and Selling Losers." Journal of Finance, 48(1), 65-91.
2. **Fama, E. F., & French, K. R. (2012)**. "Size, value, and momentum in international stock returns." Journal of Financial Economics, 105(3), 457-472.
3. **Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013)**. "Value and momentum everywhere." Journal of Finance, 68(3), 929-985.

---

## Legal Disclaimer

**IMPORTANT**: This software is for educational and research purposes only. Past performance does not guarantee future results. Trading involves substantial risk of loss and is not suitable for all investors.

### Risk Warnings

- You may lose some or all of your invested capital
- Quantitative models may fail during market stress
- Execution timing and slippage may impact returns
- Regulatory requirements may affect implementation
- No guarantee of profitability or risk control

---

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## Support

- **Documentation**: [Wiki](https://github.com/chris-cpz/strategy-dual-momentum-trend-rider/wiki)
- **Issues**: [GitHub Issues](https://github.com/chris-cpz/strategy-dual-momentum-trend-rider/issues)
- **Discussions**: [GitHub Discussions](https://github.com/chris-cpz/strategy-dual-momentum-trend-rider/discussions)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built with precision for quantitative trading research*

*Last Updated: 8/29/2025 | Version: 1.0.0*
