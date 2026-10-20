import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.stats import norm

from util.heston import (heston_char, gil_pelaez_density, price_from_density, heston_fft_prices,
                         heston_implied_vols, simulate_variance_exact, simulate_heston_paths)

st.title("🌪️ The Heston Stochastic Volatility Model")
st.markdown(r"""
Empirically, volatility is **not constant**: it clusters, mean-reverts and is negatively correlated
with returns. The Heston model replaces the constant $\sigma^2$ of Black-Scholes by a stochastic
variance process $\gamma(t) = \sigma^2(t)$. Under the pricing measure $Q$ (Section 2.5):

$$dS(t) = rS(t)\,dt + \sqrt{\gamma(t)}\,S(t)\,dW^Q(t),$$

$$d\gamma(t) = \big(\kappa - \lambda\gamma(t)\big)\,dt + \widetilde\sigma\sqrt{\gamma(t)}\,d\widetilde W(t), \qquad d\langle W^Q, \widetilde W\rangle_t = \rho\,dt.$$

| Parameter | Meaning |
|---|---|
| $\gamma(0)$ | current variance |
| $\kappa/\lambda$ | long-run variance level that $\gamma$ reverts to |
| $\lambda$ | speed of mean reversion |
| $\widetilde\sigma$ | volatility of variance ("vol of vol") |
| $\rho$ | correlation between stock and variance shocks (leverage effect) |

The lecture notes treat the uncorrelated case $\rho = 0$ for ease of exposition; the tutorials use the
full model. There is **no closed-form density** of $S(T)$, so prices come from the PDE, from Monte Carlo,
or — most efficiently — from the **characteristic function**, which *is* known in closed form:

$$\chi(u) = E_Q\big[e^{iu\log S(T)}\big] = \exp\big(\psi_0 + iu\log S(0) + \psi_2\,\gamma(0)\big).$$

""")

with st.expander("📖 Intuition: how do the parameters change the return distribution?"):
    st.markdown(r"""
    - **Vol of vol $\widetilde\sigma$** mixes calm and turbulent regimes: the terminal return becomes a
      *mixture* of normals with different variances → **fat tails** (excess kurtosis) → a symmetric
      **smile** in implied volatility.
    - **Correlation $\rho < 0$**: when the stock falls, variance rises, so large down-moves happen in
      high-volatility states → a **fat left tail** → negative skew → the downward-sloping **skew** seen
      in equity index options.
    - **Mean reversion $\lambda$** controls how long a volatility shock persists; strong mean reversion
      makes the model behave like Black-Scholes with $\sigma^2 = \kappa/\lambda$ for long maturities.
    - The **Feller condition** $2\kappa \ge \widetilde\sigma^2$ guarantees that $\gamma$ never hits zero.
    """)

st.write("---")
c = st.columns(8)
S0 = c[0].number_input("S(0)", value=120.0, step=5.0)
r = c[1].number_input("r", value=0.02, step=0.01, format="%.3f")
T = c[2].number_input("T", value=1.0, min_value=0.1, step=0.25)
gam0 = c[3].number_input("γ(0)", value=0.09, min_value=0.001, step=0.01, format="%.3f")
kappa = c[4].number_input("κ", value=0.09, min_value=0.001, step=0.01, format="%.3f")
lamb = c[5].number_input("λ", value=2.5, min_value=0.01, step=0.25)
sig_t = c[6].number_input("σ̃ (vol of vol)", value=0.3, min_value=0.01, step=0.05)
rho = c[7].number_input("ρ", value=-0.5, min_value=-0.99, max_value=0.99, step=0.1)
params = (gam0, kappa, lamb, sig_t, rho)

feller = 2 * kappa - sig_t**2
m = st.columns(4)
m[0].metric("Current vol √γ(0)", f"{np.sqrt(gam0):.2%}")
m[1].metric("Long-run vol √(κ/λ)", f"{np.sqrt(kappa / lamb):.2%}")
m[2].metric("Half-life of vol shocks", f"{np.log(2) / lamb:.2f} yrs")
m[3].metric("Feller 2κ − σ̃²", f"{feller:+.3f}", "satisfied" if feller >= 0 else "violated",
            delta_color="normal" if feller >= 0 else "inverse")
st.write("---")

# ── 1. Paths ─────────────────────────────────────────────────────────
st.header("1. Simulated Paths")
st.markdown(r"""
The variance can be simulated **without discretisation bias**: given $\gamma(t_i)$, the next value is
distributed as $Y/(2c)$ with $c = \dfrac{2\lambda}{(1 - e^{-\lambda\Delta t})\widetilde\sigma^2}$ and $Y$
non-central $\chi^2$ with $4\kappa/\widetilde\sigma^2$ degrees of freedom and non-centrality
$2ce^{-\lambda\Delta t}\gamma(t_i)$. The stock is simulated alongside with a log-Euler step.
""")
seed = st.number_input("Seed", value=20, step=1)


@st.cache_data
def paths(S0, r, T, params, seed):
    rng = np.random.default_rng(int(seed))
    gam = simulate_variance_exact(params[0], params[1], params[2], params[3], T, 1000, 10, rng)
    S, g2 = simulate_heston_paths(S0, T, r, *params, 500, 5, np.random.default_rng(int(seed) + 1))
    return gam, S, g2


gam, S_p, g_p = paths(S0, r, T, params, seed)
fig = make_subplots(rows=1, cols=2, subplot_titles=("Variance γ(t): exact simulation (10 paths)",
                                                   "Stock S(t) and its volatility √γ(t) (5 paths)"),
                    specs=[[{}, {"secondary_y": True}]])
tg = np.linspace(0, T, gam.shape[1])
for k in range(gam.shape[0]):
    fig.add_trace(go.Scatter(x=tg, y=gam[k], mode="lines", line=dict(width=1), showlegend=False), row=1, col=1)
fig.add_hline(y=kappa / lamb, line_dash="dash", line_color="black", annotation_text="κ/λ", row=1, col=1)
tp = np.linspace(0, T, S_p.shape[1])
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
for k in range(S_p.shape[0]):
    fig.add_trace(go.Scatter(x=tp, y=S_p[k], mode="lines", line=dict(width=1.5, color=colors[k]), showlegend=False),
                  row=1, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=tp, y=np.sqrt(g_p[k]), mode="lines", line=dict(width=1, dash="dot", color=colors[k]),
                             showlegend=False), row=1, col=2, secondary_y=True)
fig.update_yaxes(title_text="√γ (dotted)", secondary_y=True, row=1, col=2)
fig.update_layout(height=420)
st.plotly_chart(fig, use_container_width=True)
st.markdown(f"""
**Interpretation:** All variance paths are pulled towards the long-run level $\\kappa/\\lambda = {kappa / lamb:.4f}$
with half-life {np.log(2) / lamb:.2f} years, and stay strictly positive{' (the Feller condition holds)' if feller >= 0 else ' — but since the Feller condition is violated they can touch zero'}.
On the right, with $\\rho = {rho}$, falling stock prices (solid) tend to coincide with rising
volatility (dotted) — the **leverage effect** that makes crashes more likely than rallies of the same size.
""")
st.write("---")

# ── 2. Density ───────────────────────────────────────────────────────
st.header("2. The Density of log S(T) via Gil-Pelaez")
st.markdown(r"""
The density of $X(T) = \log S(T)$ is recovered from the characteristic function by the
**Gil-Pelaez inversion formula**

$$f(x) = \frac{1}{\pi}\int_0^\infty \operatorname{Re}\big(e^{-iux}\chi(u)\big)\,du,$$

after which options can be priced by integration as in Black-Scholes:
$V(0) = e^{-rT}\int g(e^x)f(x)\,dx$ (evaluated with Simpson's rule on the sampled density).
We compare with a **lognormal** density whose variance equals the expected integrated variance
$\bar\sigma^2 T = E\int_0^T\gamma(t)\,dt$.
""")


@st.cache_data
def density(S0, r, T, params):
    g0, kap, lam = params[:3]
    sig_bar2 = kap / lam + (g0 - kap / lam) * (1 - np.exp(-lam * T)) / (lam * T)
    sd = np.sqrt(sig_bar2 * T)
    mu = np.log(S0) + (r - 0.5 * sig_bar2) * T
    x = np.linspace(mu - 6 * sd, mu + 5 * sd, 500)
    char = lambda u: heston_char(u, S0, T, r, *params)
    dens = gil_pelaez_density(x, char, u_max=60.0)
    return x, dens, norm.pdf(x, mu, sd), np.sqrt(sig_bar2)


x, dens, ln_dens, sig_bar = density(S0, r, T, params)
log_scale = st.checkbox("Log scale (look at the tails)", value=False)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=x, y=np.maximum(dens, 1e-12), name="Heston (Gil-Pelaez)", line=dict(color="crimson", width=2.5)))
fig2.add_trace(go.Scatter(x=x, y=ln_dens, name=f"Lognormal, σ̄ = {sig_bar:.2%}", line=dict(color="steelblue", dash="dash")))
fig2.update_layout(xaxis_title="x = log S(T)", yaxis_title="density", yaxis_type="log" if log_scale else "linear",
                   height=400, legend=dict(x=0, y=1))
st.plotly_chart(fig2, use_container_width=True)

K_test = st.slider("Strike for the price check", float(0.5 * S0), float(1.5 * S0), float(round(S0 / 1.2)))
p_dens = price_from_density(x, dens, lambda s: np.maximum(s - K_test, 0), r, T)
p_fft = heston_fft_prices(S0, np.array([K_test]), T, r, *params)[0]
mean_x = np.sum(x * dens) / np.sum(dens)
skew = np.sum((x - mean_x) ** 3 * dens) / np.sum(dens) / (np.sum((x - mean_x) ** 2 * dens) / np.sum(dens)) ** 1.5
m = st.columns(3)
m[0].metric("Call price from density", f"{p_dens:.4f}")
m[1].metric("Call price via FFT (Ch. 3)", f"{p_fft:.4f}")
m[2].metric("Skewness of log S(T)", f"{skew:+.3f}")
st.markdown(f"""
**Interpretation:** Both distributions have (almost) the same variance, but the Heston density is
**skewed** (skewness {skew:+.3f}) and has a heavier tail on the side favoured by $\\rho$ — switch
to log scale to see that the lognormal tail decays much faster. The option price obtained by
integrating the payoff against the Gil-Pelaez density agrees with the FFT price from Chapter 3 to
{abs(p_dens - p_fft):.1e}. The detour via the density costs one integral **per point $x$**,
which is why Chapter 3 prices directly from the characteristic function.
""")
st.write("---")

# ── 3. Smile ─────────────────────────────────────────────────────────
st.header("3. What Heston Does to Implied Volatility")
st.markdown(r"""
If the market followed Black-Scholes, inverting the Black-Scholes formula at every strike would
return the same $\sigma$. Heston prices instead produce a **smile/skew**. Vary one parameter and keep
the others at the values above.
""")
which = st.radio("Vary", ["ρ (correlation)", "σ̃ (vol of vol)", "λ (mean reversion)"], horizontal=True)
K_grid = np.linspace(0.6 * S0, 1.6 * S0, 60)


@st.cache_data
def smile_family(S0, r, T, params, which):
    out = []
    if which.startswith("ρ"):
        vals = [-0.9, -0.5, 0.0, 0.5, 0.9]
        mk = lambda v: (params[0], params[1], params[2], params[3], v)
    elif which.startswith("σ̃"):
        vals = [0.05, 0.2, 0.4, 0.6, 0.8]
        mk = lambda v: (params[0], params[1], params[2], v, params[4])
    else:
        vals = [0.5, 1.0, 2.5, 5.0, 10.0]
        mk = lambda v: (params[0], params[1] * v / params[2], v, params[3], params[4])  # keep κ/λ fixed
    for v in vals:
        out.append((v, heston_implied_vols(S0, K_grid, T, r, mk(v))))
    return out


fig3 = go.Figure()
for (v, iv), col in zip(smile_family(S0, r, T, params, which), ["#08306b", "#2171b5", "#6baed6", "#fd8d3c", "#d94801"]):
    fig3.add_trace(go.Scatter(x=K_grid / S0, y=iv, name=f"{which.split()[0]} = {v}", line=dict(color=col, width=2)))
fig3.update_layout(xaxis_title="Moneyness K / S(0)", yaxis_title="Black-Scholes implied volatility", height=420)
st.plotly_chart(fig3, use_container_width=True)
st.markdown("""
**Interpretation:** Negative $\\rho$ tilts the curve into a **downward skew** (expensive puts, cheap
calls) — the typical shape of equity index options; positive $\\rho$ produces the reverse skew seen in
some commodity markets. Increasing the vol of vol $\\widetilde\\sigma$ bends the curve into a **smile**
(fatter tails on both sides). Faster mean reversion $\\lambda$ (with $\\kappa/\\lambda$ fixed) flattens
everything, because volatility shocks die out before maturity. These are exactly the levers the
calibration in Chapter 3 pulls to fit market data.
""")
