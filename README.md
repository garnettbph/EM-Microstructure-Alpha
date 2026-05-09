# Emerging Market Microstructure: Alpha in Illiquid Environments

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-Enabled-orange.svg)
![Quant](https://img.shields.io/badge/Domain-Quantitative_Finance-success.svg)

## Project Overview
Many standard cross-sectional machine learning predictions break down entirely when applied to emerging markets. This project explores the development of a high-frequency (minute-level) statistical arbitrage strategy specifically constrained by emerging market liquidity.

**The Core Thesis:** *Alpha in high-frequency trading is an illusion without proper execution engineering.* This project proves mathematically that standard machine learning models fail in high-friction environments, and demonstrates how to rescue weak alpha using Time-Weighted Average Price (TWAP) algorithms and passive limit orders.

---

## Repository Structure

```text
EM-Microstructure-Alpha/
│
├── README.md                           # Project documentation and findings
├── requirements.txt                    # Python dependencies
├── notebooks/
│   └── EM_Microstructure_Analysis.ipynb  # Complete research notebook
├── src/                                
│   ├── __init__.py
│   ├── data_generator.py               # Stochastic EM market simulator
│   ├── features.py                     # Microstructure alpha extraction
│   └── backtesters.py                  # Custom execution engines
└── images/                             
    ├── feature_importance.png          
    ├── naive_execution.png       
    ├── twap_execution.png          
    └── hourly_timeframe.png
```
---

## The Data Problem: Why a Custom Simulator?
Free financial APIs (like Yahoo Finance) are inadequate for modeling emerging market microstructure. They lack Level-2 order book data, and limit historical 1-minute data to few days.

To solve this, I engineered a Stochastic Market Simulator (src/data_generator.py) that models realistic Emerging Market behavior using:

- Poisson Processes: To model sparse trade arrivals and generate realistic zero-volume minutes.

- Geometric Brownian Motion: For the underlying true asset price.

- Discrete Tick Constraints & Bid-Ask Bounce: To enforce realistic exchange tick sizes and simulate the friction of crossing the spread.
---

## Feature Engineering
I extracted specialized microstructure features to capture order book dynamics:
- Order Flow Imbalance (OFI) Proxy: Using the Tick Rule to determine buyer-initiated vs. seller-initiated volume.
- Amihud Illiquidity Measure: To penalize assets that move aggressively on low volume.
- Roll's Spread Estimator: To dynamically approximate the effective bid-ask spread and model execution slippage
---

## The Three Experiments
### Experiment 1: The Illusion of Alpha (Naive Execution)

I trained a LightGBM cross-sectional ranking model on 1-minute data. Standard, frictionless backtesters showed massive profits. However, when passed through a custom execution engine that enforced maximum volume participation (5%) and penalized spread-crossing (Roll_Spread), the alpha was destroyed.

- Result: -17.60% Return | -3.27 Sharpe Ratio (images/naive_execution.png)
- Takeaway: The bid-ask spread in emerging markets is larger than the short-term alpha predicted by the ML model. Aggressive market orders guarantee a steady bleed of capital.

### Experiment 2: Execution Engineering (TWAP + Maker Orders)
To rescue the strategy, I built a secondary execution algorithm (src/backtesters.py). Instead of taking liquidity, the engine slices target positions into 5-minute blocks (TWAP) and uses passive limit orders. This acts as a market maker, earning the spread rather than paying it, and reduces exchange fees.

- Result: +5.89% Return | 1.75 Sharpe Ratio (images/twap_execution.png)
- Takeaway: By eliminating the spread penalty, the faint predictive signal from the LightGBM model survived friction. Execution mechanics matter just as much as the predictive model.

### Experiment 3: The Danger of Timeframe Shifting
A common "fix" for high-frequency noise is to down-sample data. I aggregated the data to 1-Hour bars to see if a lower trading frequency could bypass the spread costs.

- Result: -64.27% Return | -44.19 Sharpe Ratio (images/hourly_timeframe.png)
- Takeaway: Catastrophic failure. Aggregating the data smoothed out the microstructure anomalies (bid-ask bounce, order flow imbalance) that the model relied on. The market was reduced to an unpredictable random walk, resulting in massive losses from spread payments. Alpha is highly timeframe-specific.
---

## How to Run the Code
1. Clone the repository:
```bash
git clone https://github.com/garnettbph/EM-Microstructure-Alpha.git
cd EM-Microstructure-Alpha
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Analysis:
Open notebooks/EM_Microstructure_Analysis.ipynb and run all cells to generate the synthetic data, train the LightGBM models, and run the three execution backtests. For a fully automated run of the data pipeline, model training, and execution backtests, simply run the main script from the root directory:
```bash
python main.py
```
---

## Conclusion
This project demonstrates that in illiquid, emerging market environments, Alpha is entirely dependent on execution capacity. A cross-sectional machine learning model is effectively useless unless paired with an execution algorithm (like TWAP) designed to navigate sparse volume and bid-ask spread friction without incurring excessive market impact.
