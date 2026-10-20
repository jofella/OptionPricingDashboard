import streamlit as st

st.title("CompFinLearn Dashboard")
st.markdown("""
An **interactive learning dashboard** for numerical methods in option pricing, written in Python.
Content follows the *'Computational Finance'* curriculum at **CAU Kiel** and the accompanying weekly tutorials.
""")
st.info("""
**Academic Resource** — Course material by **Prof. Dr. Mathias Vetter**:
[University of Kiel Lecture Notes](https://www.math.uni-kiel.de/finmath)
""", icon="📖")

st.markdown(r"""
### The Four Basic Tasks
The lecture revolves around four questions that come back in every chapter, each time answered with a
different numerical tool:

1. **European options** — compute $V(0) = e^{-rT}E_Q[f(S(T))]$.
2. **American options** — solve the optimal stopping problem $\sup_\tau E_Q[e^{-r\tau} g(S(\tau))]$.
3. **Hedging** — compute the replicating strategy, i.e. the *Greeks* (mostly $\Delta = \partial V/\partial S$).
4. **Calibration** — choose model parameters so that model prices match quoted market prices.

No single method wins everywhere: trees are simple, transforms are fast, Monte Carlo is flexible,
PDE methods are accurate, and neural networks scale to settings where everything else breaks.
""")

st.divider()
st.subheader("📚 Curriculum Coverage")

with st.expander("Chapter 1 — Binomial Trees"):
    st.markdown("""
    - **CRR Model:** moment matching for $u, d, q$; interactive tree with option values and hedge ratios
    - **Convergence:** CRR vs Black-Scholes, odd/even oscillation, strike-anchored (Leisen-Reimer type) tree
    - **Exotics:** down-and-out barrier call on the recombining tree
    - **American options:** early exercise region, Richardson extrapolation $2V_{4M} - V_{2M}$
    """)

with st.expander("Chapter 2 — Continuous Market Models"):
    st.markdown("""
    - **Black-Scholes:** closed form, Greeks, no-arbitrage bounds, put-call parity
    - **Pricing by integration:** arbitrary payoffs (power calls), speed vs binomial trees
    - **Perpetual American put:** free-boundary ODE solved numerically vs closed form
    - **Delta hedging:** replicating a DAX call (Feb 2023 – Jan 2024), discrete rebalancing error
    - **Heston model:** stochastic variance paths, Gil-Pelaez density, fat tails & skew
    """)

with st.expander("Chapter 3 — Integral Transforms"):
    st.markdown("""
    - **Laplace transform pricing:** damped payoff transforms, choice of $R$
    - **Fast Fourier Transform:** all strikes at once, speed comparison
    - **Hedging via transforms:** Heston delta across strikes
    - **Implied volatility & calibration:** volatility smile, Heston fit by Nelder-Mead
    """)

with st.expander("Chapter 4 — Monte Carlo Simulation"):
    st.markdown("""
    - **Basics:** convergence rate $1/\\sqrt{N}$, confidence intervals, chooser option
    - **Random numbers:** inversion (exact & interpolated), acceptance/rejection with optimal proposal
    - **Variance reduction:** antithetic variables, control variates (Parisian option), importance sampling (NIG model)
    - **Sensitivities:** finite differences (bias-variance trade-off), pathwise, likelihood ratio
    - **SDE simulation:** Euler vs Milstein (strong order), exact Heston variance vs Euler
    - **Longstaff-Schwartz:** regression-based American put pricing
    """)

with st.expander("Chapter 5 — Finite Differences"):
    st.markdown("""
    - **Heat-equation transformation** of the Black-Scholes PDE
    - **Explicit scheme:** stability condition $\\lambda \\le 1/2$ and what happens when it is violated
    - **Implicit / Crank-Nicolson ($\\theta$-scheme):** unconditional stability, tridiagonal solver
    - **American puts:** projection onto the obstacle, free boundary
    """)

with st.expander("Chapter 6 — Machine Learning"):
    st.markdown("""
    - **Deep hedging:** train a neural network hedge on simulated paths and compare with the BS delta,
      in a complete (Black-Scholes) and an incomplete (Heston) market
    """)

st.divider()
st.caption("""
**Disclaimer:** Academic project for educational purposes. Not intended for financial advice or live trading.
""")
