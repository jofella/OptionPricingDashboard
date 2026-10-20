import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from util.black_scholes import bs_price, bs_greeks
from util.monte_carlo import gbm_terminal, delta_fd, delta_pathwise, delta_lr, digital_delta_exact
from util.ui import takeaways, test_yourself

st.title("🧭 Monte Carlo Sensitivities (Greeks)")
st.markdown(r"""
Hedging requires derivatives of option prices such as $\Delta = \frac{\partial}{\partial s}E[Z(s)]$ with
$Z(s) = e^{-rT}f\big(S(T; s)\big)$, $S(T; s) = s\,e^{(r-\sigma^2/2)T + \sigma\sqrt T X}$. Section 4.4 presents three
Monte Carlo approaches:

| Method | Estimator of $\Delta$ | Requirement | Bias |
|---|---|---|---|
| **Finite differences** | $\frac{1}{N}\sum \frac{Z_n(s + h/2) - Z_n(s - h/2)}{h}$ | none | $O(h^2)$ |
| **Infinitesimal perturbation** (pathwise) | $\frac1N\sum \frac{\partial Z_n}{\partial s} = \frac1N\sum e^{-rT}f'(S_n(T))\frac{S_n(T)}{s}$ | $f$ (Lipschitz) continuous | none |
| **Likelihood ratio** | $\frac1N\sum e^{-rT}f(S_n(T))\,\frac{\partial}{\partial s}\log g_s(S_n(T))$ | density $g_s$ of $S(T)$ known | none |

For geometric Brownian motion the score is $\frac{\partial}{\partial s}\log g_s = \frac{X}{s\sigma\sqrt T}$.
The pathwise method differentiates the **payoff**, the likelihood ratio method differentiates the **density**.
""")

with st.expander("📖 Intuition: which method when?"):
    st.markdown(r"""
    - **Pathwise** is usually the most accurate, but it needs a derivative of the payoff. For a digital
      option $1_{\{S(T) \ge K\}}$ the pathwise derivative is $0$ almost surely — the estimator returns $0$,
      which is plainly wrong. Discontinuous payoffs break it.
    - **Likelihood ratio** never touches the payoff, so it works for digitals and barriers. The price is
      variance: the score $X/(s\sigma\sqrt T)$ is unbounded and multiplies the whole payoff.
    - **Finite differences** always work but have a bias-variance trade-off in $h$. Using the **same
      random numbers** for $s \pm h/2$ (common random numbers) is essential — otherwise the variance
      explodes like $1/h^2$.
    """)

st.write("---")
c = st.columns(6)
S0 = c[0].number_input("S(0)", value=100.0, step=5.0)
K = c[1].number_input("Strike K", value=100.0, step=5.0)
T = c[2].number_input("T", value=1.0, min_value=0.1, step=0.25)
r = c[3].number_input("r", value=0.02, step=0.01, format="%.3f")
sigma = c[4].number_input("σ", value=0.30, min_value=0.01, step=0.05)
N = c[5].select_slider("Simulations N", [1000, 10000, 50000, 200000], value=50000)

# ── 1. Comparison ────────────────────────────────────────────────────
st.header("1. Three Estimators Side by Side")
payoff = st.radio("Payoff", ["European call", "Digital call 1{S(T) ≥ K}"], horizontal=True)
digital = payoff.startswith("Digital")
h = st.select_slider("Finite-difference step h", [1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0], value=0.5)

rng = np.random.default_rng(20)
X = rng.standard_normal(N)
if digital:
    exact = digital_delta_exact(S0, K, T, r, sigma)
    fd_samples = np.exp(-r * T) * ((gbm_terminal(S0 + h / 2, T, r, sigma, X) >= K).astype(float)
                                   - (gbm_terminal(S0 - h / 2, T, r, sigma, X) >= K)) / h
else:
    exact = bs_greeks(S0, K, T, r, sigma)["Delta"]
    fd_samples = delta_fd(S0, K, T, r, sigma, X, h)
ip_samples = delta_pathwise(S0, K, T, r, sigma, X, digital=digital)
lr_samples = delta_lr(S0, K, T, r, sigma, X, digital=digital)

rows = []
for name, smp in (("Finite differences (CRN)", fd_samples), ("Pathwise (IP)", ip_samples), ("Likelihood ratio", lr_samples)):
    est, sd = smp.mean(), smp.std(ddof=1)
    rows.append([name, est, est - exact, sd**2, 1.96 * sd / np.sqrt(N)])
df = pd.DataFrame(rows, columns=["Method", "Estimate", "Error", "Var per sample", "95% half-width"]).set_index("Method")
c1, c2 = st.columns([1, 2])
c1.metric("Exact delta", f"{exact:.5f}")
c2.dataframe(df.style.format({"Estimate": "{:.5f}", "Error": "{:+.5f}", "Var per sample": "{:.4f}", "95% half-width": "{:.5f}"}),
             use_container_width=True)
if digital:
    st.markdown(f"""
    **Interpretation:** The pathwise estimator returns exactly 0 — the payoff is flat almost everywhere,
    and the jump at $K$ (where all the sensitivity lives) is invisible to a pathwise derivative.
    The likelihood ratio estimator works because it differentiates the smooth density instead.
    Finite differences catch the jump only on the few paths with $S(T)$ within $h$ of $K$; for small $h$
    these become rare and the variance per sample grows like $1/h$
    ({df.loc['Finite differences (CRN)', 'Var per sample']:.1f} at $h = {h}$).
    """)
else:
    st.markdown(f"""
    **Interpretation:** All three are unbiased up to Monte Carlo noise. The pathwise estimator has the
    smallest variance ({df.loc['Pathwise (IP)', 'Var per sample']:.3f} per sample) because it uses the
    exact derivative of each path. The likelihood ratio estimator is
    {df.loc['Likelihood ratio', 'Var per sample'] / df.loc['Pathwise (IP)', 'Var per sample']:.1f}× noisier: the score
    multiplies the full payoff. With common random numbers the finite-difference estimator converges
    to the pathwise one as $h \\to 0$.
    """)
with st.expander("🎓 Deep dive: deriving the pathwise and likelihood ratio estimators"):
    st.markdown(r"""
    Write the price as $z(s) = E[Z(s)]$ with $Z(s) = e^{-rT}f\big(s\,e^{(r - \sigma^2/2)T + \sigma\sqrt T X}\big)$.

    **Pathwise (infinitesimal perturbation).** Differentiate *inside* the expectation:
    $z'(s) = E\big[\frac{\partial}{\partial s}Z(s)\big] = E\big[e^{-rT}f'(S(T))\frac{S(T)}{s}\big]$, because $\partial S(T)/\partial s = S(T)/s$.
    Interchanging derivative and expectation requires $Z$ to be (Lipschitz) continuous in $s$. For a call
    $f' = 1_{\{S_T > K\}}$, giving the estimator used here.

    **Likelihood ratio.** Move the parameter into the **density** instead: $z(s) = \int e^{-rT}f(y)\,g_s(y)\,dy$, where
    $g_s$ is the density of $S(T)$. Differentiating the density,

    $$z'(s) = \int e^{-rT}f(y)\,\frac{\partial_s g_s(y)}{g_s(y)}\,g_s(y)\,dy = E\big[e^{-rT}f(S(T))\,\partial_s\log g_s(S(T))\big].$$

    For the lognormal density, $\partial_s\log g_s = X/(s\sigma\sqrt T)$ — the **score function**. No derivative of $f$ is
    needed, so discontinuous payoffs are fine; the price is that the score is unbounded, which inflates the variance.
    """)

st.write("---")

# ── 2. FD bias-variance ──────────────────────────────────────────────
st.header("2. The Bias-Variance Trade-off of Finite Differences")
st.markdown(r"""
The central difference has bias $\frac{h^2}{24}z'''(s) + O(h^4)$, where $z''' $ is the **speed**
(third derivative of the price). Its variance depends on how the two prices are simulated:

- **Common random numbers (CRN):** the same $X_n$ for $s \pm h/2$; for a Lipschitz payoff the variance
  stays bounded as $h \to 0$.
- **Independent draws:** $\operatorname{Var} \approx 2\operatorname{Var}(Z)/h^2$ — explodes for small $h$.

The mean squared error is $\text{MSE}(h) = \text{bias}(h)^2 + \operatorname{Var}/N$. Below the bias is computed
exactly from the closed-form price.
""")


@st.cache_data
def mse_curves(S0, K, T, r, sigma, N, digital):
    rng = np.random.default_rng(20)
    X1, X2 = rng.standard_normal(N), rng.standard_normal(N)
    hs = np.logspace(-3, np.log10(1.5 * S0), 70)  # h < 2 S0 keeps S0 - h/2 > 0
    if digital:
        price = lambda s: np.exp(-r * T) * norm.cdf((np.log(s / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)))
        pay = lambda s, x: np.exp(-r * T) * (gbm_terminal(s, T, r, sigma, x) >= K)
        exact = digital_delta_exact(S0, K, T, r, sigma)
    else:
        price = lambda s: bs_price(s, K, T, r, sigma)
        pay = lambda s, x: np.exp(-r * T) * np.maximum(gbm_terminal(s, T, r, sigma, x) - K, 0)
        exact = bs_greeks(S0, K, T, r, sigma)["Delta"]
    bias = (price(S0 + hs / 2) - price(S0 - hs / 2)) / hs - exact
    var_crn = np.array([np.var((pay(S0 + h / 2, X1) - pay(S0 - h / 2, X1)) / h) for h in hs])
    var_ind = np.array([np.var((pay(S0 + h / 2, X1) - pay(S0 - h / 2, X2)) / h) for h in hs])
    return hs, bias, var_crn / N, var_ind / N


hs, bias, v_crn, v_ind = mse_curves(S0, K, T, r, sigma, N, digital)
ip_var = np.var(ip_samples) / N
lr_var = np.var(lr_samples) / N
fig = go.Figure()
fig.add_trace(go.Scatter(x=hs, y=bias**2 + v_crn, name="FD with common random numbers", line=dict(color="crimson", width=2.5)))
fig.add_trace(go.Scatter(x=hs, y=bias**2 + v_ind, name="FD with independent draws", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=hs, y=bias**2, name="bias² only", line=dict(color="grey", dash="dot")))
if not digital:
    fig.add_hline(y=ip_var, line_color="steelblue", line_dash="dash", annotation_text="pathwise (IP)",
                  annotation_position="bottom left")
fig.add_hline(y=lr_var, line_color="seagreen", line_dash="dash", annotation_text="likelihood ratio",
              annotation_position="top left")
mse_crn = bias**2 + v_crn
y_lo = np.log10(max(min(mse_crn.min(), ip_var, lr_var) / 20, 1e-12))
y_hi = np.log10(max((bias**2 + v_ind).max(), mse_crn.max()) * 3)
fig.update_layout(xaxis_type="log", yaxis_type="log", xaxis_title="h", yaxis_title="MSE", height=450,
                  yaxis_range=[y_lo, y_hi], legend=dict(x=0.02, y=0.98))
st.plotly_chart(fig, use_container_width=True)
best = hs[np.argmin(mse_crn)]
cross = hs[np.argmax(bias**2 > v_crn)] if np.any(bias**2 > v_crn) else None
cross_txt = rf" (here from $h \approx {cross:.2g}$ on)" if cross is not None else ""
st.markdown(f"""
**Interpretation:** For large $h$ the squared bias (grey) dominates{cross_txt} — the
difference quotient averages the slope over too wide an interval. For small $h$ the variance takes over: with independent draws
(orange) the MSE explodes like $1/h^2$, with common random numbers (red)
{'it levels off at the pathwise variance — the CRN estimator *becomes* the pathwise estimator.' if not digital else 'it still grows like 1/h because the digital payoff jumps.'}
The best step for CRN here is $h \\approx {best:.3g}$. Unbiased methods (dashed lines) avoid the trade-off entirely.
""")
with st.expander("🎓 Deep dive: bias and optimal step size of finite differences"):
    st.markdown(r"""
    Taylor expansion around $s$ gives
    $z(s \pm \tfrac h2) = z \pm \tfrac h2 z' + \tfrac{h^2}{8}z'' \pm \tfrac{h^3}{48}z''' + \dots$, so the central difference has

    $$\frac{z(s + h/2) - z(s - h/2)}{h} = z'(s) + \frac{h^2}{24}z'''(s) + O(h^4).$$

    (A forward difference would have bias of order $h$ — much worse.) The mean squared error is
    $\text{MSE}(h) = \big(\tfrac{h^2}{24}z'''\big)^2 + \operatorname{Var}/N$:

    - **Independent draws:** $\operatorname{Var} \approx 2\operatorname{Var}(Z)/h^2$. Balancing $h^4$ against $1/(Nh^2)$ gives
      $h^* \propto N^{-1/6}$ and $\text{MSE} \propto N^{-2/3}$ — slower than the usual $N^{-1}$.
    - **Common random numbers, Lipschitz payoff:** the variance stays bounded, so $h$ can be taken small and the MSE
      approaches $\operatorname{Var}/N$ — as good as the pathwise method.
    - **Common random numbers, discontinuous payoff (digital):** the difference is non-zero only on paths with $S_T$
      within $O(h)$ of $K$, which gives $\operatorname{Var} \propto 1/h$ and again a genuine trade-off.
    """)

st.write("---")

# ── 3. LR across strikes ─────────────────────────────────────────────
st.header("3. Likelihood Ratio Variance Across Strikes")
st.markdown(r"""
The likelihood ratio estimator for the digital call is $e^{-rT}1_{\{S(T)\ge K\}}\frac{X}{s\sigma\sqrt T}$.
How noisy it is depends on where the strike sits relative to the distribution of $S(T)$.
""")


@st.cache_data
def lr_vs_K(S0, T, r, sigma, N):
    rng = np.random.default_rng(20)
    X = rng.standard_normal(N)
    Ks = np.linspace(0.1 * S0, 3 * S0, 200)
    var_lr = np.array([np.var(delta_lr(S0, k, T, r, sigma, X, digital=True)) for k in Ks])
    rel = np.array([np.sqrt(v / N) / digital_delta_exact(S0, k, T, r, sigma) for v, k in zip(var_lr, Ks)])
    return Ks, var_lr, rel


Ks, var_lr, rel = lr_vs_K(S0, T, r, sigma, N)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=Ks, y=var_lr, name="Var per sample", line=dict(color="seagreen")))
fig2.add_trace(go.Scatter(x=Ks, y=rel, name="relative std. error", yaxis="y2", line=dict(color="crimson", dash="dash")))
fig2.update_layout(xaxis_title="Strike K", yaxis_title="Variance of LR sample",
                   yaxis2=dict(title="Std. error / true delta", overlaying="y", side="right", type="log"),
                   height=400, legend=dict(x=0.6, y=1))
st.plotly_chart(fig2, use_container_width=True)
i_min = int(np.argmin(rel))
st.markdown(f"""
**Interpretation:** For small strikes almost every path pays 1, so the estimator is essentially the score
$e^{{-rT}}X/(s\\sigma\\sqrt T)$ itself: its variance is maximal (≈ $e^{{-2rT}}/(s\\sigma)^2T^{{-1}} = {np.exp(-2 * r * T) / (S0 * sigma) ** 2 / T:.5f}$)
while the true delta is almost zero — the estimator averages large positive and negative scores to
get a tiny number. As $K$ grows, fewer paths pay and the absolute variance falls. What matters is the
**relative** error (dashed, log scale): it is smallest near $K \\approx {Ks[i_min]:.0f}$, where the delta itself
is largest, and blows up for deep in-the-money and deep out-of-the-money digitals.
""")

with st.expander("🎓 Deep dive: second-order Greeks (gamma)"):
    st.markdown(r"""
    - **Finite differences:** $\frac{z(s+h) - 2z(s) + z(s-h)}{h^2}$; even with common random numbers its variance explodes
      for a call, because the second difference of the kinked payoff is non-zero only near the strike.
    - **Pathwise:** fails for a call — the pathwise first derivative is a step function, whose derivative is 0 almost
      everywhere.
    - **Likelihood ratio:** works, using the second derivative of the density.
    - **Mixed:** apply the pathwise method once and the likelihood ratio once
      ("LR-pathwise"), usually the most efficient choice for gamma.

    General rule: pathwise where the payoff is smooth enough, likelihood ratio where it is not, and finite
    differences (with common random numbers!) when nothing else is available.
    """)

takeaways([
    "Finite differences need common random numbers and a carefully chosen $h$ (bias $O(h^2)$ vs. variance).",
    "Pathwise derivatives are unbiased and low-variance but require a continuous payoff — they fail for digitals.",
    "Likelihood ratio differentiates the density (score function), works for any payoff, but has higher variance.",
    "Choose the method by payoff smoothness; for gamma combine pathwise and likelihood ratio.",
])

test_yourself("greeks", [
    (r"Derive the pathwise delta estimator of a European call in the Black-Scholes model.",
     r"$\partial_s e^{-rT}(S_T - K)^+ = e^{-rT}1_{\{S_T > K\}}\,\partial_s S_T = e^{-rT}1_{\{S_T > K\}}S_T/s$; average over simulations."),
    (r"Why does the pathwise method fail for a digital option, and which method works instead?",
     r"The payoff $1_{\{S_T \ge K\}}$ has derivative 0 almost everywhere, so the pathwise estimator is identically 0. The likelihood ratio method (or finite differences) works."),
    (r"What is the bias of the central finite difference, and why is it better than the forward difference?",
     r"$\frac{h^2}{24}z'''(s) + O(h^4)$; the forward difference has bias $\frac h2 z''(s) + O(h^2)$, one order worse, because the even terms do not cancel."),
    (r"Why do common random numbers reduce the variance of the finite-difference estimator so drastically?",
     r"With the same $X_n$ the two prices are highly positively correlated, so the variance of their difference is small (of order $h^2$ for Lipschitz payoffs) instead of $2\operatorname{Var}(Z)$."),
])
