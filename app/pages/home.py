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

### What makes a numerical method good?
- **Accuracy** — how close is the answer, and how does the error $\varepsilon(N) \approx CN^{-p}$ shrink with the
  computational effort $N$? A higher rate $p$ wins eventually, a smaller constant $C$ may win for practical $N$.
- **Speed** — operations and memory needed for a given accuracy.
- **Implementation effort** — a method needed once does not justify a month of optimisation.
- **Generality** — does it extend to other models and payoffs (path dependence, early exercise, many assets)?
""")

with st.expander("🧭 How to use this dashboard"):
    st.markdown(r"""
    Every page follows the same pattern:

    1. **Theory** at the top, in the notation of the lecture notes.
    2. **📖 Intuition** boxes explain the idea in plain language, often with a small numerical example.
    3. **🎓 Deep dive** boxes contain derivations and background for exam preparation.
    4. **Interactive sections**: change parameters with the widgets — all charts and the
       **Interpretation** texts below them are recomputed from the actual numbers.
    5. **💡 Key takeaways** and **📝 Test yourself** at the end of each page: exam-style questions with answers that
       can be revealed one by one.

    Suggested route: start with the binomial tree (Ch. 1) and Black-Scholes (Ch. 2), then compare how the same European
    call is priced by transforms (Ch. 3), Monte Carlo (Ch. 4) and finite differences (Ch. 5). American options appear in
    Ch. 1, 4 and 5 — comparing the three approaches is a good exam exercise.
    """)

with st.expander("🔤 Notation used throughout"):
    st.markdown(r"""
    | Symbol | Meaning |
    |---|---|
    | $S(t)$, $B(t) = e^{rt}$ | stock price and bank account (bond) |
    | $r$, $\sigma$ | risk-free interest rate, volatility |
    | $T$, $K$ | maturity, strike |
    | $Q$ | equivalent martingale (risk-neutral) measure |
    | $f$, $g$ | payoff of a European / American option |
    | $V(t) = v(t, S(t))$ | option price and pricing function |
    | $\varphi = (\varphi_0, \varphi_1)$ | hedging strategy: bonds and shares |
    | $\Phi$, $\varphi$ (density) | standard normal cdf and density |
    | $\gamma(t)$, $\kappa$, $\lambda$, $\widetilde\sigma$, $\rho$ | Heston variance and its parameters (long-run level $\kappa/\lambda$) |
    | $\chi(u) = E_Q[e^{iu\log S(T)}]$ | characteristic function of the log price |
    | $M$, $N$, $m$ | number of tree steps / simulations / time steps (depending on the chapter) |
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
