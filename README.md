# 🧮 Computational Finance Dashboard (CompFinLearn)

This project was developed during my master's studies as part of the **Computational Finance** course at CAU Kiel.
It is the companion of my [Risk Management Dashboard](https://github.com/jofella/RiskLearnDashboard): an interactive
review of the lecture's numerical methods and an exercise in building modular **Streamlit** applications.

---

## 📖 Project Overview

The lecture revolves around four tasks — pricing European options, pricing American options, hedging, and calibration —
and solves them again and again with different numerical tools. The dashboard turns each tool into an interactive page:
change a parameter and watch prices, errors, convergence rates and hedges respond, with the theory and a data-driven
interpretation next to every chart.

- Theory follows the lecture notes *Computational Finance* by Prof. Dr. Mathias Vetter (CAU Kiel, Summer Term 2026)
- Exercises follow the weekly Python tutorials (Weeks 2–10)
- All numerics are implemented from scratch in NumPy/SciPy in `util/`, separated from the Streamlit front end

---

## ⚙️ Features

### Chapter 1 — Binomial Trees
- Interactive CRR tree (hover nodes for $S$, $V$ and hedge ratio), random path through the tree, early-exercise nodes
- Convergence to Black-Scholes in $M$ and $S(0)$: saw-tooth error of CRR vs the strike-anchored tree ($ud = c$)
- No-arbitrage bounds and put-call parity across volatilities
- Down-and-out barrier calls: price vs barrier, slow convergence, discrete vs continuous monitoring
- American puts: early-exercise premium, even/odd oscillation, Richardson extrapolation $2V_{4M} - V_{2M}$

### Chapter 2 — Continuous Market Models
- Black-Scholes prices and Greeks (Delta, Gamma, Vega, Theta, Rho, Speed) across maturities
- Pricing by integration for arbitrary payoffs (power calls, straddles, digitals) incl. delta by integration and a speed comparison with trees
- Perpetual American put: free-boundary ODE via `solve_ivp` vs closed form
- Delta hedging a DAX call (course data, Feb 2023 – Feb 2024) with adjustable rebalancing; simulated hedging error vs number of trades and vs volatility misspecification
- Heston model: exact (non-central χ²) variance simulation, Gil-Pelaez density vs lognormal, effect of ρ, σ̃, λ on the implied volatility smile

### Chapter 3 — Integral Transforms
- Laplace transform pricing with an interactive damping constant $R$ (call / call − S(0) / put depending on the strip)
- FFT pricing of all strikes at once, timing and accuracy vs adaptive quadrature
- Heston deltas via the transform
- Implied volatility smile of S&P 500 calls (course data), "fit it yourself" sliders, Nelder-Mead calibration of Heston to implied volatilities or prices

### Chapter 4 — Monte Carlo Simulation
- Convergence trajectories with confidence funnel, $n^{-1/2}$ error rate, chooser option
- Random numbers: linear congruential generators, inversion (exact and interpolated), acceptance/rejection with optimal proposal, Box-Muller, Cholesky on the course `stocks.csv`
- Variance reduction: antithetic variables (and when they fail), control variates for a Parisian option, importance sampling by exponential tilting in the NIG model (incl. the infinite-variance trap)
- Sensitivities: finite differences (bias-variance trade-off, common random numbers), pathwise and likelihood-ratio estimators, digital options
- SDE schemes: Euler vs Milstein, measured strong orders; Heston variance with Euler (negative values) vs exact simulation
- Longstaff-Schwartz: regression step, exercise boundary vs CRR, in-/out-of-sample bias, basis and exercise-date sensitivity

### Chapter 5 — Finite Differences
- Heat-equation transform, explicit scheme with live stability check (λ ≤ ½) and wrong boundary conditions
- θ-schemes (implicit, Crank-Nicolson) with convergence study and Thomas algorithm
- American puts by projection: price surface and free boundary

### Chapter 6 — Machine Learning
- Deep hedging with a NumPy neural network (hand-written backpropagation + Adam) in Black-Scholes and Heston markets, compared with the BS delta hedge

---

## 🗂️ Project Structure

```
OptionPricingDashboard/
│
├── app/
│   ├── main.py                        # Navigation entrypoint (st.navigation)
│   └── pages/
│       ├── home.py
│       ├── 1_Binomial_Trees.py        # Ch. 1
│       ├── 2_1_Black_Scholes.py       # Ch. 2
│       ├── 2_2_Delta_Hedging.py
│       ├── 2_3_Heston_Model.py
│       ├── 3_1_Laplace_FFT.py         # Ch. 3
│       ├── 3_2_Calibration.py
│       ├── 4_1_MC_Basics.py           # Ch. 4
│       ├── 4_2_Random_Numbers.py
│       ├── 4_3_Variance_Reduction.py
│       ├── 4_4_Sensitivities.py
│       ├── 4_5_SDE_Simulation.py
│       ├── 4_6_Longstaff_Schwartz.py
│       ├── 5_Finite_Differences.py    # Ch. 5
│       └── 6_Deep_Hedging.py          # Ch. 6
│
├── util/
│   ├── binomial.py                    # CRR / anchored trees, barriers, American options
│   ├── black_scholes.py               # Closed forms, Greeks, integration, implied volatility
│   ├── heston.py                      # Characteristic function, Laplace / FFT pricing, simulation
│   ├── monte_carlo.py                 # SDE schemes, sensitivity estimators, Longstaff-Schwartz
│   ├── finite_differences.py          # θ-scheme for the heat equation, Thomas algorithm
│   ├── deep_hedging.py                # NumPy MLP + Adam for deep hedging
│   └── data_utils.py                  # Cached data loading
│
├── data/
│   ├── DAX_index.csv                  # DAX daily closes 2000–2024 (dates)
│   ├── DAX_sample.csv                 # Course file: 255 DAX closes for the hedging tutorial
│   ├── options_SP500.csv              # Course file: S&P 500 calls, 06 Sep 2022, expiry 30 Dec 2022
│   └── stocks.csv                     # Course file: four price series
│
├── sources/                           # Lecture notes and tutorial notebooks (not tracked)
└── requirements.txt
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/jofella/OptionPricingDashboard.git
cd OptionPricingDashboard
python -m venv venv
venv\Scripts\activate          # Windows  (macOS / Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app/main.py
```

---

**Disclaimer:** Academic project for educational purposes. Not intended for financial advice or live trading.
