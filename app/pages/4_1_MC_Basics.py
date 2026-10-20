import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from util.black_scholes import bs_price, chooser_price
from util.monte_carlo import gbm_terminal, mc_summary

st.title("🎲 Monte Carlo Simulation — The Basics")
st.markdown(r"""
Option prices are expectations, $V(0) = E_Q\big[e^{-rT}f(S(T))\big] = E[h(X)]$, so they can be
approximated by **averaging simulations** (Section 4.1):

$$\widehat V_N = \frac1N\sum_{n=1}^N h(X_n), \qquad X_1, X_2, \ldots \text{ i.i.d.}$$

- **Law of large numbers:** $\widehat V_N \to V(0)$ almost surely.
- **Central limit theorem:** $\sqrt N(\widehat V_N - V(0)) \Rightarrow \mathcal N(0, \sigma^2)$ with
  $\sigma^2 = \operatorname{Var}(h(X))$, so the error is of order $\sigma/\sqrt N$.
- **Confidence interval:** with the sample variance $\widehat\sigma_N^2$, an approximate 95% interval is

$$\Big[\widehat V_N - 1.96\sqrt{\widehat\sigma_N^2/N},\; \widehat V_N + 1.96\sqrt{\widehat\sigma_N^2/N}\Big].$$

""")

with st.expander("📖 Why Monte Carlo when it converges so slowly?"):
    st.markdown(r"""
    An error of order $N^{-1/2}$ means **100× more simulations for one more correct digit** — slow
    compared to quadrature in one dimension. The crucial point is that the rate $N^{-1/2}$ does
    **not depend on the dimension** of $X$. A deterministic grid with $N$ points in $d$ dimensions has
    only $N^{1/d}$ points per axis, so its error deteriorates like $N^{-k/d}$. For path-dependent
    options ($d$ = number of time steps) or baskets, Monte Carlo is often the only feasible method.

    The other lever is the constant $\sigma^2$: **variance reduction** (next pages) lowers $\sigma^2$
    instead of increasing $N$.
    """)

st.write("---")
c = st.columns(5)
S0 = c[0].number_input("S(0)", value=120.0, step=5.0)
K = c[1].number_input("Strike K", value=100.0, step=5.0)
T = c[2].number_input("Maturity T", value=1.0, min_value=0.1, step=0.25)
r = c[3].number_input("r", value=0.02, step=0.01, format="%.3f")
sigma = c[4].number_input("σ", value=0.30, min_value=0.01, step=0.05)
true_val = bs_price(S0, K, T, r, sigma)

# ── 1. Convergence trajectories ──────────────────────────────────────
st.header("1. Convergence of the Estimator")
c1, c2, c3 = st.columns(3)
M = c1.slider("Number of independent runs", 5, 50, 20)
N = c2.select_slider("Simulations per run N", [1000, 5000, 10000, 50000], value=10000)
seed = c3.number_input("Seed", value=20, step=1)


@st.cache_data
def trajectories(S0, K, T, r, sigma, M, N, seed):
    rng = np.random.default_rng(int(seed))
    X = rng.standard_normal((M, N))
    pay = np.exp(-r * T) * np.maximum(gbm_terminal(S0, T, r, sigma, X) - K, 0)
    n = np.arange(1, N + 1)
    est = np.cumsum(pay, axis=1) / n
    var_true = np.var(np.exp(-r * T) * np.maximum(gbm_terminal(S0, T, r, sigma, rng.standard_normal(2_000_000)) - K, 0))
    return est, var_true


est, var_true = trajectories(S0, K, T, r, sigma, M, N, seed)
n = np.arange(1, N + 1)
idx = np.unique(np.geomspace(100, N, 400).astype(int)) - 1
band = 1.96 * np.sqrt(var_true / n[idx])
inside = np.mean(np.abs(est[:, -1] - true_val) <= 1.96 * np.sqrt(var_true / N))

fig = go.Figure()
for k in range(M):
    fig.add_trace(go.Scatter(x=n[idx], y=est[k, idx], mode="lines", line=dict(color="steelblue", width=0.8),
                             opacity=0.6, showlegend=(k == 0), name="MC trajectories"))
fig.add_trace(go.Scatter(x=n[idx], y=true_val + band, line=dict(color="black", dash="dash"), name="95% band"))
fig.add_trace(go.Scatter(x=n[idx], y=true_val - band, line=dict(color="black", dash="dash"), showlegend=False))
fig.add_hline(y=true_val, line_color="crimson", annotation_text="Black-Scholes")
fig.update_layout(xaxis_title="Number of simulations n", yaxis_title="Estimate of V(0)", height=450, legend=dict(x=0.75, y=1))
st.plotly_chart(fig, use_container_width=True)

m = st.columns(4)
m[0].metric("True price (BS)", f"{true_val:.4f}")
m[1].metric("Variance of discounted payoff", f"{var_true:.1f}")
m[2].metric(f"95% half-width at N={N:,}", f"{1.96 * np.sqrt(var_true / N):.4f}")
m[3].metric("Runs ending inside the band", f"{inside:.0%}")

st.markdown(f"""
**Interpretation:** Every run wanders early on and then settles down, but even after {N:,} simulations
the estimates still scatter by about ±{1.96 * np.sqrt(var_true / N):.2f} around the true value
{true_val:.4f}. The dashed funnel $V(0) \\pm 1.96\\sqrt{{\\sigma^2/n}}$ narrows only like $1/\\sqrt n$.
About 95% of the runs should end inside it (here {inside:.0%} of {M}). The variance of the
discounted payoff, {var_true:.0f}, is large because the call payoff is unbounded — this constant is
what variance reduction attacks.
""")

errs = np.abs(est[:, idx] - true_val)
rmse = np.sqrt(np.mean(errs**2, axis=0))
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=n[idx], y=rmse, name="RMSE across runs", line=dict(color="crimson")))
fig2.add_trace(go.Scatter(x=n[idx], y=np.sqrt(var_true / n[idx]), name="σ/√n", line=dict(color="black", dash="dash")))
fig2.update_layout(xaxis_type="log", yaxis_type="log", xaxis_title="n", yaxis_title="Error", height=330,
                   title="Error decays like n^(−1/2): slope −1/2 on log-log axes")
st.plotly_chart(fig2, use_container_width=True)
st.write("---")

# ── 2. Chooser option ────────────────────────────────────────────────
st.header("2. Pricing a Chooser Option")
st.markdown(r"""
A **chooser option** lets the holder decide at time $t^* \in (0, T)$ whether it becomes a European call
or a put with strike $K$ and maturity $T$. Monte Carlo handles this naturally: simulate $S(t^*)$, compare
the Black-Scholes call and put values at $t^*$, then simulate $S(T)$ and pay the chosen payoff.
By put-call parity the choice is "call" iff $S(t^*) > Ke^{-r(T-t^*)}$, and the exact price is

$$V(0) = V_{\text{call}}(S(0), K, T) + V_{\text{put}}\big(S(0), Ke^{-r(T-t^*)}, t^*\big).$$

""")
c1, c2 = st.columns(2)
t_star = c1.slider("Choosing time t*", 0.05, float(T) - 0.05, min(0.5, float(T) - 0.05), step=0.05)
N_ch = c2.select_slider("Paths", [1000, 10000, 100000], value=10000)
rng = np.random.default_rng(int(seed))
S_star = gbm_terminal(S0, t_star, r, sigma, rng.standard_normal(N_ch))
S_T = gbm_terminal(S_star, T - t_star, r, sigma, rng.standard_normal(N_ch))
choose_call = bs_price(S_star, K, T, r, sigma, True, t=t_star) > bs_price(S_star, K, T, r, sigma, False, t=t_star)
pay = np.where(choose_call, np.maximum(S_T - K, 0), np.maximum(K - S_T, 0)) * np.exp(-r * T)
est_ch, var_ch, eps_ch = mc_summary(pay)
exact = chooser_price(S0, K, T, t_star, r, sigma)

m = st.columns(4)
m[0].metric("MC chooser price", f"{est_ch:.4f}")
m[1].metric("95% CI", f"[{est_ch - eps_ch:.3f}, {est_ch + eps_ch:.3f}]")
m[2].metric("Exact price", f"{exact:.4f}")
m[3].metric("Paths choosing the call", f"{choose_call.mean():.1%}")

sub = slice(0, min(N_ch, 3000))
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=S_star[sub][choose_call[sub]], y=S_T[sub][choose_call[sub]], mode="markers",
                          marker=dict(size=3, color="steelblue"), name="chose call"))
fig3.add_trace(go.Scatter(x=S_star[sub][~choose_call[sub]], y=S_T[sub][~choose_call[sub]], mode="markers",
                          marker=dict(size=3, color="crimson"), name="chose put"))
fig3.add_vline(x=K * np.exp(-r * (T - t_star)), line_dash="dash", line_color="black", annotation_text="K·exp(−r(T−t*))")
fig3.add_hline(y=K, line_dash="dot", line_color="grey")
fig3.update_layout(xaxis_title="S(t*)", yaxis_title="S(T)", height=420)
st.plotly_chart(fig3, use_container_width=True)
st.markdown(f"""
**Interpretation:** The decision boundary is the vertical line $S(t^*) = Ke^{{-r(T-t^*)}}$. Paths in the
lower-right (chose call, ended below $K$) and upper-left (chose put, ended above $K$) are the
**regret** cases: the choice was right given the information at $t^*$ but turned out worthless.
The chooser is worth more than either the call ({bs_price(S0, K, T, r, sigma):.3f}) or the put
({bs_price(S0, K, T, r, sigma, False):.3f}) alone, but less than the straddle (their sum), because the
holder must commit at $t^*$. The exact price {exact:.4f} lies {'inside' if abs(est_ch - exact) <= eps_ch else 'outside'}
the 95% confidence interval.
""")
