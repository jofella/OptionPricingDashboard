import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.stats import ncx2

from util.monte_carlo import gbm_paths, coarsen
from util.heston import simulate_variance_exact, simulate_variance_euler
from util.ui import takeaways, test_yourself

st.title("〽️ Simulating Stochastic Differential Equations")
st.markdown(r"""
Path-dependent options (barriers, Asians, Parisians, American exercise) need whole paths, not only
$S(T)$. For a general SDE $dX(t) = a(X(t))\,dt + b(X(t))\,dW(t)$ we discretise time with
$\Delta t = T/m$ and step forward (Section 4.5):

**Euler scheme**

$$X_{i+1} = X_i + a(X_i)\Delta t + b(X_i)\Delta W_i$$

**Milstein scheme** — adds the next term of the Itô-Taylor expansion:

$$X_{i+1} = X_i + a(X_i)\Delta t + b(X_i)\Delta W_i + \tfrac12 b(X_i)b'(X_i)\big((\Delta W_i)^2 - \Delta t\big)$$

For geometric Brownian motion $a(x) = rx$, $b(x) = \sigma x$, so the Milstein correction is
$\frac12\sigma^2X_i\big((\Delta W_i)^2 - \Delta t\big)$, and the exact solution
$S(t_i) = S(0)\exp\big((r - \sigma^2/2)t_i + \sigma W(t_i)\big)$ is available for comparison.
""")

with st.expander("📖 Strong vs weak convergence"):
    st.markdown(r"""
    - **Strong order $\gamma$:** $E|X_m - X(T)| \le C\Delta t^\gamma$ — the simulated *path* is close to the true
      path driven by the same Brownian motion. Relevant for path-dependent payoffs. Euler: $\gamma = 1/2$,
      Milstein: $\gamma = 1$.
    - **Weak order $\beta$:** $|E f(X_m) - E f(X(T))| \le C\Delta t^\beta$ — only the *distribution* matters.
      Relevant for European payoffs. Both Euler and Milstein have $\beta = 1$.

    For a plain European option the extra effort of Milstein therefore buys little; for path-dependent
    products and for multilevel methods its better strong order pays off.
    """)

with st.expander("🎓 Deep dive: where the Milstein correction comes from (Itô-Taylor expansion)"):
    st.markdown(r"""
    Over one step, $X(t + \Delta t) = X(t) + \int_t^{t+\Delta t}a(X_s)\,ds + \int_t^{t+\Delta t}b(X_s)\,dW_s$. Euler freezes
    the integrands at time $t$. To do better, expand $b(X_s)$ itself with Itô's formula:
    $b(X_s) \approx b(X_t) + b'(X_t)b(X_t)(W_s - W_t)$. Plugging this in,

    $$\int_t^{t+\Delta t}b(X_s)\,dW_s \approx b(X_t)\Delta W + b'(X_t)b(X_t)\int_t^{t+\Delta t}(W_s - W_t)\,dW_s.$$

    The remaining stochastic integral can be computed with Itô's formula: $\int_t^{t+\Delta t}(W_s - W_t)\,dW_s = \tfrac12\big((\Delta W)^2 - \Delta t\big)$.
    This term has standard deviation of order $\Delta t$ — the same order as the drift term, so Euler's neglect of it
    limits Euler to strong order $1/2$. Including it gives Milstein with strong order 1.

    **Trick for GBM:** simulating $\log S$ instead of $S$ makes the SDE have constant coefficients; the Euler scheme for
    $\log S$ is then *exact*. Transforming to coordinates with additive noise is often the best "scheme".
    """)

st.write("---")
c = st.columns(4)
S0 = c[0].number_input("S(0)", value=120.0, step=5.0)
r = c[1].number_input("r", value=0.02, step=0.01, format="%.3f")
sigma = c[2].slider("σ", 0.05, 1.0, 0.30, step=0.05)
T = c[3].number_input("T", value=1.0, min_value=0.1, step=0.25)

# ── 1. One path ──────────────────────────────────────────────────────
st.header("1. One Path, Three Schemes")
c1, c2 = st.columns(2)
m_path = c1.select_slider("Time steps m", [10, 25, 50, 100, 250, 1000], value=50)
seed = c2.number_input("Seed", value=20, step=1)
rng = np.random.default_rng(int(seed))
dW = rng.standard_normal((1, m_path)) * np.sqrt(T / m_path)
t_grid = np.linspace(0, T, m_path + 1)
S_ex = gbm_paths(S0, T, r, sigma, dW, "exact")[0]
S_eu = gbm_paths(S0, T, r, sigma, dW, "euler")[0]
S_mi = gbm_paths(S0, T, r, sigma, dW, "milstein")[0]

fig = make_subplots(rows=1, cols=2, subplot_titles=("Paths driven by the same Brownian increments", "Relative error vs exact solution"))
fig.add_trace(go.Scatter(x=t_grid, y=S_ex, name="Exact", line=dict(color="black", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=t_grid, y=S_eu, name="Euler", line=dict(color="steelblue", dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=t_grid, y=S_mi, name="Milstein", line=dict(color="crimson", dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=t_grid, y=(S_eu - S_ex) / S_ex, name="Euler", line=dict(color="steelblue"), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=t_grid, y=(S_mi - S_ex) / S_ex, name="Milstein", line=dict(color="crimson"), showlegend=False), row=1, col=2)
fig.update_xaxes(title_text="t")
fig.update_layout(height=400, legend=dict(x=0, y=1))
st.plotly_chart(fig, use_container_width=True)
st.markdown(f"""
**Interpretation:** All three paths look alike because they share the same Brownian increments — this
is what *strong* convergence is about. The Euler error accumulates in a random-walk fashion
(final relative error {(S_eu[-1] - S_ex[-1]) / S_ex[-1]:+.2%}), while the Milstein correction removes
the leading error term ({(S_mi[-1] - S_ex[-1]) / S_ex[-1]:+.3%}). Increase $\\sigma$ to make the
difference more dramatic: the neglected term $\\frac12\\sigma^2 S((\\Delta W)^2 - \\Delta t)$ scales with $\\sigma^2$.
""")
with st.expander("📖 Intuition: why the errors are measured with the same Brownian motion"):
    st.markdown(r"""
    Strong convergence compares the **simulated path** with the **true path driven by the same noise**. To measure it
    we need the true path and the scheme on a common Brownian motion: we simulate the Brownian increments on the
    finest grid and sum them to get the increments of coarser grids. Using fresh random numbers on each grid would
    compare two unrelated paths and the error would never go to zero — that would be a comparison of distributions
    (weak convergence) instead.
    """)

st.write("---")

# ── 2. Strong order ──────────────────────────────────────────────────
st.header("2. Measuring the Strong Order of Convergence")
st.markdown(r"""
We simulate $N$ fine Brownian paths with $2^{10}$ steps and obtain coarser paths by **summing increments**
(so every resolution is driven by the same Brownian motion). The strong error
$E|S_m - S(T)|$ is plotted against $\Delta t$ on log-log axes, where the slope is the order.
""")


@st.cache_data
def strong_errors(S0, r, sigma, T, N=4000, seed=1):
    rng = np.random.default_rng(seed)
    fine = 2**10
    dW = rng.standard_normal((N, fine)) * np.sqrt(T / fine)
    S_T = gbm_paths(S0, T, r, sigma, dW.sum(axis=1, keepdims=True), "exact")[:, -1]
    ms = 2 ** np.arange(2, 11)
    eu, mi = [], []
    for m in ms:
        dWc = coarsen(dW, fine // m)
        eu.append(np.mean(np.abs(gbm_paths(S0, T, r, sigma, dWc, "euler")[:, -1] - S_T)))
        mi.append(np.mean(np.abs(gbm_paths(S0, T, r, sigma, dWc, "milstein")[:, -1] - S_T)))
    return T / ms, np.array(eu), np.array(mi)


dts, err_eu, err_mi = strong_errors(S0, r, sigma, T)
slope_eu = np.polyfit(np.log(dts), np.log(err_eu), 1)[0]
slope_mi = np.polyfit(np.log(dts), np.log(err_mi), 1)[0]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=dts, y=err_eu, mode="lines+markers", name=f"Euler (slope {slope_eu:.2f})", line=dict(color="steelblue")))
fig2.add_trace(go.Scatter(x=dts, y=err_mi, mode="lines+markers", name=f"Milstein (slope {slope_mi:.2f})", line=dict(color="crimson")))
fig2.add_trace(go.Scatter(x=dts, y=err_eu[-1] * np.sqrt(dts / dts[-1]), name="∝ Δt^½", line=dict(color="steelblue", dash="dot")))
fig2.add_trace(go.Scatter(x=dts, y=err_mi[-1] * dts / dts[-1], name="∝ Δt", line=dict(color="crimson", dash="dot")))
fig2.update_layout(xaxis_type="log", yaxis_type="log", xaxis_title="Δt", yaxis_title="E|S_m − S(T)|", height=420)
st.plotly_chart(fig2, use_container_width=True)
st.markdown(f"""
**Interpretation:** The fitted slopes are {slope_eu:.2f} for Euler and {slope_mi:.2f} for Milstein,
matching the theoretical strong orders 1/2 and 1. Halving $\\Delta t$ reduces the Milstein error by a
factor 2 but the Euler error only by $\\sqrt2 \\approx 1.41$. At the finest grid Milstein is
{err_eu[-1] / err_mi[-1]:.0f}× more accurate for the same number of steps.
""")
with st.expander("🎓 Deep dive: why strong vs. weak order matters — multilevel Monte Carlo"):
    st.markdown(r"""
    For a European price the total error of an SDE-based Monte Carlo estimate has two parts:

    $$\text{MSE} = \underbrace{\big(E[f(X_m)] - E[f(X_T)]\big)^2}_{\text{bias}^2 \;\sim\; \Delta t^{2\beta}} + \underbrace{\frac{\operatorname{Var}f(X_m)}{N}}_{\text{statistical error}}.$$

    With weak order $\beta = 1$, an accuracy $\varepsilon$ needs $\Delta t \sim \varepsilon$ and $N \sim \varepsilon^{-2}$, i.e. a cost of
    order $\varepsilon^{-3}$. **Multilevel Monte Carlo** (Giles, 2008) writes $E[f(X_{\text{fine}})]$ as a telescoping sum of
    corrections between consecutive grids that share the same Brownian motion. The corrections have small variance
    *because of strong convergence*, so most samples can be taken on cheap coarse grids. With Milstein (strong order 1)
    the cost drops to $\varepsilon^{-2}$ — the same as if the exact solution were available.

    The same "fine minus coarse on shared noise" idea also appears in Richardson extrapolation of the
    discretisation bias, $2E[f(X_{\Delta t/2})] - E[f(X_{\Delta t})]$.
    """)

st.write("---")

# ── 3. Heston variance ───────────────────────────────────────────────
st.header("3. The Heston Variance: Why Euler Breaks Down")
st.markdown(r"""
For $d\gamma = (\kappa - \lambda\gamma)\,dt + \widetilde\sigma\sqrt\gamma\,dW$ the Euler step
$\gamma_{i+1} = \gamma_i + (\kappa - \lambda\gamma_i)\Delta t + \widetilde\sigma\sqrt{\gamma_i}\,\Delta W_i$ can produce
**negative variances**, after which $\sqrt{\gamma}$ is undefined. Common fixes are *full truncation*
(use $\gamma^+$ in drift and diffusion) and *reflection* ($|\gamma|$). The **exact** scheme samples
the known transition law: $\gamma(t+\Delta t) \mid \gamma(t) \sim Y/(2c)$, $Y$ non-central $\chi^2$ — no bias at all.
""")
c = st.columns(5)
gam0 = c[0].number_input("γ(0)", value=0.09, step=0.01, format="%.3f")
kappa = c[1].number_input("κ", value=0.09, step=0.01, format="%.3f")
lamb = c[2].number_input("λ", value=2.5, step=0.25)
sig_t = c[3].slider("σ̃ (vol of vol)", 0.1, 1.5, 0.6, step=0.05)
m_h = c[4].select_slider("Steps m", [12, 52, 252], value=52)
N_h = 20000


@st.cache_data
def heston_var(gam0, kappa, lamb, sig_t, T, m, N):
    out = {}
    out["Exact"] = simulate_variance_exact(gam0, kappa, lamb, sig_t, T, m, N, np.random.default_rng(1))
    for fix in ("none", "full truncation", "reflection"):
        out[f"Euler ({fix})"] = simulate_variance_euler(gam0, kappa, lamb, sig_t, T, m, N, np.random.default_rng(1), fix)
    return out


sims = heston_var(gam0, kappa, lamb, sig_t, T, m_h, N_h)
mean_exact = kappa / lamb + (gam0 - kappa / lamb) * np.exp(-lamb * T)
rows = []
for name, g in sims.items():
    rows.append([name, np.mean(g[:, -1]), np.mean(g[:, -1]) - mean_exact, np.mean((g < 0).any(axis=1)), np.mean(g < 0)])
df = pd.DataFrame(rows, columns=["Scheme", "Mean γ(T)", "Bias", "Paths with γ < 0", "Share of negative values"]).set_index("Scheme")
st.dataframe(df.style.format({"Mean γ(T)": "{:.5f}", "Bias": "{:+.5f}", "Paths with γ < 0": "{:.1%}", "Share of negative values": "{:.2%}"}),
             use_container_width=True)
st.caption(f"Exact mean E[γ(T)] = κ/λ + (γ(0) − κ/λ)e^(−λT) = {mean_exact:.5f}.   Feller: 2κ − σ̃² = {2 * kappa - sig_t**2:+.3f}")

c_ = 2 * lamb / ((1 - np.exp(-lamb * T)) * sig_t**2)
df_ = 4 * kappa / sig_t**2
nc = 2 * c_ * np.exp(-lamb * T) * gam0
xg = np.linspace(1e-5, np.percentile(sims["Exact"][:, -1], 99.5), 400)
fig3 = make_subplots(rows=1, cols=2, subplot_titles=("Sample paths: exact (black) vs plain Euler (blue)", "Distribution of γ(T)"))
tt = np.linspace(0, T, m_h + 1)
for k in range(6):
    fig3.add_trace(go.Scatter(x=tt, y=sims["Exact"][k], line=dict(color="black", width=1), showlegend=False), row=1, col=1)
    fig3.add_trace(go.Scatter(x=tt, y=sims["Euler (none)"][k], line=dict(color="steelblue", width=1), showlegend=False), row=1, col=1)
fig3.add_hline(y=0, line_color="crimson", row=1, col=1)
fig3.add_trace(go.Histogram(x=sims["Euler (full truncation)"][:, -1], histnorm="probability density", nbinsx=80,
                            name="Euler (full truncation)", marker_color="steelblue", opacity=0.5), row=1, col=2)
fig3.add_trace(go.Histogram(x=sims["Exact"][:, -1], histnorm="probability density", nbinsx=80, name="Exact scheme",
                            marker_color="grey", opacity=0.5), row=1, col=2)
fig3.add_trace(go.Scatter(x=xg, y=ncx2.pdf(2 * c_ * xg, df_, nc) * 2 * c_, name="True density", line=dict(color="crimson")), row=1, col=2)
# for 4κ/σ̃² < 2 the true density is infinite at 0, so scale the axes to the histogram instead
hist_h, hist_e = np.histogram(sims["Exact"][:, -1], bins=80, density=True)
fig3.update_yaxes(range=[0, 1.3 * np.sort(hist_h)[-2]], row=1, col=2)
fig3.update_xaxes(range=[min(0.0, np.percentile(sims["Euler (full truncation)"][:, -1], 0.5)), xg[-1]], row=1, col=2)
fig3.update_layout(height=420, barmode="overlay", legend=dict(x=0.6, y=1))
st.plotly_chart(fig3, use_container_width=True)
neg = df.loc["Euler (none)", "Paths with γ < 0"]
st.markdown(f"""
**Interpretation:** With $\\widetilde\\sigma = {sig_t}$ and only {m_h} steps per year, {neg:.0%} of the plain Euler
paths hit negative variance — in a real implementation the next step would take the square root of a
negative number. The variance process near zero has a large diffusion relative to its level, which the
linear Euler step cannot resolve. The exact scheme samples the true transition density (red line) and
is unbiased for any step size, at the cost of drawing non-central $\\chi^2$ variables. Note that *full truncation*
does not prevent negative states — it only uses $\\gamma^+$ inside drift and diffusion, so the share of negative
values stays similar while the scheme remains well defined; *reflection* removes them but adds a clear upward bias.
{'The Feller condition is violated, so even the true process touches zero; Euler fixes then introduce visible bias.' if 2 * kappa < sig_t**2 else 'Lower κ or raise σ̃ to violate the Feller condition and see the Euler fixes become biased.'}
""")

takeaways([
    "Euler freezes the coefficients over a step: strong order 1/2, weak order 1.",
    "Milstein adds $\\frac12 bb'((\\Delta W)^2 - \\Delta t)$ from the Itô-Taylor expansion: strong order 1.",
    "Strong convergence matters for path-dependent payoffs and multilevel Monte Carlo; weak convergence for European prices.",
    "Square-root processes (Heston variance) break naive Euler; exact simulation via the non-central $\\chi^2$ law has no bias.",
])

test_yourself("sde", [
    (r"Write down the Euler and the Milstein step for $dS = rS\,dt + \sigma S\,dW$.",
     r"Euler: $S_{i+1} = S_i(1 + r\Delta t + \sigma\Delta W_i)$. Milstein adds $\frac12\sigma^2S_i((\Delta W_i)^2 - \Delta t)$, since $bb' = \sigma S\cdot\sigma$."),
    (r"Define strong and weak order of convergence. What are they for Euler and Milstein?",
     r"Strong: $E|X_m - X(T)| \le C\Delta t^\gamma$ (pathwise). Weak: $|E f(X_m) - E f(X(T))| \le C\Delta t^\beta$ (in distribution). Euler: $\gamma = 1/2$, $\beta = 1$; Milstein: $\gamma = 1$, $\beta = 1$."),
    (r"Why can the plain Euler scheme for the Heston variance produce invalid values, and how does exact simulation avoid this?",
     r"The Gaussian increment $\widetilde\sigma\sqrt{\gamma_i}\Delta W$ is unbounded, so $\gamma_{i+1}$ can be negative and $\sqrt{\gamma}$ undefined. Exact simulation draws from the true transition law (scaled non-central $\chi^2$), which is non-negative."),
    (r"How can GBM be simulated without any discretisation error?",
     r"Simulate $\log S$, which has constant coefficients: $\log S_{i+1} = \log S_i + (r - \sigma^2/2)\Delta t + \sigma\Delta W_i$ is exact."),
])
