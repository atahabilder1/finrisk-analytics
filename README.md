# FinRisk Analytics

**Volatility Forecasting & Value-at-Risk Modeling Using ML & Econometrics**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)

---

## 🎯 Overview

FinRisk Analytics is an institutional-grade risk modeling platform that combines classical econometric methods (GARCH, EGARCH) with modern machine learning (LSTM, XGBoost) to forecast volatility and calculate Value-at-Risk across multiple asset classes.

**Key Features:**
- 📊 Multi-model volatility forecasting (GARCH, EGARCH, LSTM)
- ⚠️ Value-at-Risk calculation (5 methods)
- 📈 Expected Shortfall (CVaR) analysis
- 🔍 Statistical backtesting (Kupiec test)
- 🔗 Correlation regime detection
- 📱 Interactive Streamlit dashboard
- 📉 Crisis period stress testing

---

## 🏗️ Project Structure
```
finrisk-analytics/
├── configs/              # Configuration files
├── data/                 # Data storage
│   ├── raw/             # Raw market data
│   ├── processed/       # Cleaned data
│   └── features/        # Engineered features
├── notebooks/           # Jupyter notebooks (5 total)
├── src/                 # Source code
│   ├── data/           # Data handling
│   ├── models/         # GARCH & LSTM models
│   ├── risk/           # VaR calculation & backtesting
│   ├── correlation/    # Correlation analysis
│   └── utils/          # Utilities
├── dashboard/          # Streamlit dashboard
├── results/            # Outputs
│   ├── plots/         # Visualizations
│   ├── reports/       # Risk reports
│   └── models/        # Trained models
└── tests/             # Unit tests
```

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/aniktahabilder/finrisk-analytics.git
cd finrisk-analytics
```

### 2. Set Up Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Settings

Edit `configs/config.yaml` to customize:
- Date ranges
- Asset universe
- Model parameters
- VaR confidence levels

### 4. Run Analysis

**Option A: Jupyter Notebooks**
```bash
jupyter notebook

# Run notebooks in order:
# 01_volatility_eda.ipynb
# 02_garch_models.ipynb
# 03_lstm_volatility.ipynb
# 04_var_calculation.ipynb
# 05_portfolio_risk.ipynb
```

**Option B: Interactive Dashboard**
```bash
streamlit run dashboard/app.py
```

---

## 📊 Asset Universe

**15 Assets across 4 Classes:**

| Category | Assets | Purpose |
|----------|--------|---------|
| **Equities** | SPY, XLK, XLF, XLV, XLE, XLI | Market + sector risk |
| **Fixed Income** | TLT, IEF | Duration risk |
| **Alternatives** | GLD, BTC, ETH, SOL | Diversification risk |
| **Macro** | VIX | Volatility indicator |

**Deep-dive case studies:** SPY, BTC, ETH, TLT

---

## 🔧 Methodology

### Volatility Models

**Classical Econometrics:**
- **GARCH(1,1)** - Baseline volatility clustering
- **EGARCH** - Asymmetric volatility (leverage effect)
- **GJR-GARCH** - Captures negative shock impact

**Machine Learning:**
- **LSTM** - Sequential time-series patterns
- **LSTM-GARCH Hybrid** - Combined approach
- **XGBoost** - Feature-based volatility

### Value-at-Risk Methods

1. **Historical Simulation** - Empirical distribution
2. **Parametric (Normal)** - Gaussian assumption
3. **Parametric (Student-t)** - Fat tails
4. **Cornish-Fisher** - Skewness/kurtosis adjusted
5. **Filtered Historical Simulation** - GARCH + historical

### Risk Metrics

- Value-at-Risk (95%, 99% confidence)
- Expected Shortfall (CVaR)
- Rolling correlations
- Regime-dependent correlations
- Portfolio VaR (multi-asset)

### Backtesting

- VaR violation counting
- Kupiec test (statistical validation)
- Traffic light approach (Basel framework)
- Crisis period analysis (March 2020, 2022)

---

## 📈 Target Metrics

| Metric | Target |
|--------|--------|
| **Volatility Forecast RMSE** | < 2% |
| **VaR Coverage (95%)** | 4-6% violations |
| **Kupiec Test** | Pass at 5% |
| **Expected Shortfall Error** | < 10% |
| **Dashboard Load Time** | < 3 seconds |

---

## 📱 Dashboard Features

**Interactive Streamlit App:**
- Real-time volatility charts
- VaR breach timeline visualization
- Correlation heatmaps (by regime)
- Portfolio risk calculator
- Model comparison view
- Crisis scenario analysis

**Launch:**
```bash
streamlit run dashboard/app.py
```

---

## 🔗 Related Projects

This is part of a three-project portfolio:

1. [Market Intelligence ML](https://github.com/aniktahabilder/market-intelligence-ml) - Price prediction
2. **FinRisk Analytics** ← *You are here*
3. [AlphaRL Portfolio](https://github.com/aniktahabilder/alpharl-portfolio) - RL optimization

**Integration:** This project provides volatility forecasts and VaR estimates that feed into Project 3's RL agent for risk-aware portfolio optimization.

---

## 📚 Dependencies

**Core:**
- pandas, numpy, scipy
- scikit-learn, xgboost, tensorflow

**Econometrics:**
- arch (GARCH models)
- statsmodels

**Visualization:**
- matplotlib, seaborn, plotly
- streamlit (dashboard)

**Financial Data:**
- yfinance, ccxt

See `requirements.txt` for complete list.

---

## 🎯 Use Cases

**Risk Management:**
- Portfolio risk monitoring
- VaR calculation for regulatory compliance
- Stress testing under crisis scenarios
- Early warning system for volatility spikes

**Quantitative Research:**
- Volatility model comparison
- Risk factor attribution
- Correlation regime analysis

**Portfolio Management:**
- Risk budgeting
- Dynamic hedging strategies
- Capital allocation decisions

---

## 🧪 Testing
```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

---

## 📝 To-Do

- [ ] Implement data fetching pipeline
- [ ] Fit GARCH models on all assets
- [ ] Train LSTM volatility models
- [ ] Calculate VaR using all methods
- [ ] Run Kupiec backtesting
- [ ] Analyze correlation regimes
- [ ] Build interactive dashboard
- [ ] Generate risk reports
- [ ] Crisis period analysis
- [ ] Write technical documentation

---

## 🤝 Contributing

This is a personal portfolio project. Feedback and suggestions are welcome!

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## 👤 Author

**Anik Tahabilder**
- PhD Student, Computer Science, Wayne State University
- Research: Multimodal AI, Blockchain Security, Quantitative Finance
- GitHub: [@aniktahabilder](https://github.com/aniktahabilder)

---

## 🙏 Acknowledgments

- GARCH methodology from Engle (2001) and Bollerslev (1986)
- VaR backtesting framework based on Basel Committee guidelines
- Built as part of comprehensive ML/Finance portfolio

---

## 📞 Contact

For questions or collaboration:
- GitHub: [aniktahabilder](https://github.com/aniktahabilder)
- Email: [your-email]

---

**⭐ If you find this project useful, please consider giving it a star!**
