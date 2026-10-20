import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.optimize import minimize

from util.black_scholes import bs_price, implied_vol
from util.heston import heston_fft_prices
from util.data_utils import load_sp500_options

st.title("🎯 Implied Volatility and Model Calibration")
st.markdown(r"""
A model is only useful for pricing exotic products or hedging if it reproduces the prices of liquid
instruments. **Calibration** chooses the parameters $\theta$ so that model prices $V_i(\theta)$ match
observed market prices $\pi_i$ (Section 3.3):

$$\hat\theta = \arg\min_\theta \sum_{i=1}^n \big(V_i(\theta) - \pi_i\big)^2 \qquad\text{or, preferably,}\qquad \hat\theta = \arg\min_\theta \sum_{i=1}^n \big(\mathrm{IV}(V_i(\theta)) - \mathrm{IV}(\pi_i)\big)^2 .$$

### Implied volatility
For fixed $S(0), K, T, r$ the Black-Scholes price is strictly increasing in $\sigma$, so every
admissible price corresponds to exactly one **implied volatility** $\mathrm{IV}(\pi)$ with
$V_{BS}(\sigma = \mathrm{IV}(\pi)) = \pi$. It is a **dimensionless quote** of how expensive an option
is: a price change of 10 USD may be due to the underlying moving, but a 2-point IV change means the option
got genuinely more expensive. If Black-Scholes were correct, all IVs would be equal.
""")

with st.expander("📖 Why calibrate to implied volatilities rather than prices?"):
    st.markdown(r"""
    Prices of deep in-the-money options are large and those of far out-of-the-money options tiny.
    A least-squares fit to **prices** therefore concentrates on the expensive ITM options and nearly
    ignores the wings — which carry most of the information about skew and tails. Implied volatilities
    put all strikes on a **common scale**, so every option contributes comparably to the objective.

    Calibration in practice is hard because the objective is **non-convex**: Heston parameters
    partly compensate each other (e.g. high $\widetilde\sigma$ with weak correlation vs low $\widetilde\sigma$
    with strong correlation), so different starting points can lead to different local minima —
    especially with a **single maturity**, which does not pin down the mean-reversion speed.
    """)

st.write("---")

# ── Data ─────────────────────────────────────────────────────────────
st.header("1. Market Data")
df = load_sp500_options()
source = "course"
uploaded = st.file_uploader("Optional: upload option quotes (CSV with columns Strike, Midpoint, Underlying, TTM, Interest)",
                            type="csv")
if uploaded is not None:
    df = pd.read_csv(uploaded)
    source = "upload"


@st.cache_data
def synthetic_quotes(seed=7):
    """Synthetic S&P-500-like quotes generated from a Heston model plus noise in implied volatility."""
    S0, ttm, r = 3908.19, 0.3194, 0.0325
    K = np.arange(2800.0, 4801.0, 50.0)
    true = (0.065, 0.08, 2.0, 0.85, -0.75)
    iv = implied_vol(heston_fft_prices(S0, K, ttm, r, *true), S0, K, ttm, r)
    iv_noisy = iv + np.random.default_rng(seed).normal(0, 0.002, len(K))
    prices = bs_price(S0, K, ttm, r, iv_noisy)
    return pd.DataFrame({"Strike": K, "Midpoint": prices, "Underlying": S0, "TTM": ttm, "Interest": r}), true


true_params = None
if df is None:
    df, true_params = synthetic_quotes()
    source = "synthetic"
    st.info("The course file `data/options_SP500.csv` (S&P 500 calls of 06 Sep 2022, expiry 30 Dec 2022) "
            "was not found, so **synthetic S&P-500-like quotes** are generated from a Heston model with "
            "noise. Place the course file in `data/` or upload it above to calibrate to real quotes.", icon="ℹ️")

S0 = float(df["Underlying"].iloc[0])
ttm = float(df["TTM"].iloc[0])
r = float(df["Interest"].iloc[0])
strikes = df["Strike"].values.astype(float)
prices = df["Midpoint"].values.astype(float)
iv_mkt = implied_vol(prices, S0, strikes, ttm, r)
ok = ~np.isnan(iv_mkt)
strikes, prices, iv_mkt = strikes[ok], prices[ok], iv_mkt[ok]

m = st.columns(4)
m[0].metric("Underlying", f"{S0:,.2f}")
m[1].metric("Time to maturity", f"{ttm:.4f} yrs")
m[2].metric("Interest rate", f"{r:.4f}")
m[3].metric("Number of quotes", len(strikes))

fig = make_subplots(rows=1, cols=2, subplot_titles=("Call mid prices", "Implied volatility"))
fig.add_trace(go.Scatter(x=strikes, y=prices, mode="markers", marker=dict(color="black", size=5), name="Market"), row=1, col=1)
fig.add_trace(go.Scatter(x=strikes, y=iv_mkt, mode="markers+lines", marker=dict(color="black", size=5),
                         line=dict(color="grey", width=1), name="Market IV"), row=1, col=2)
fig.add_vline(x=S0, line_dash="dot", line_color="grey", row=1, col=2)
fig.update_xaxes(title_text="Strike")
fig.update_layout(height=380, showlegend=False)
st.plotly_chart(fig, use_container_width=True)
st.markdown(f"""
**Interpretation:** The implied volatility is far from flat: it falls from {iv_mkt[0]:.1%} for the
lowest strike to {iv_mkt.min():.1%} — the **volatility skew** typical for equity indices. Investors pay
a premium for downside protection (low-strike puts, and via put-call parity the corresponding calls),
which Black-Scholes with a single $\\sigma$ cannot reproduce. A model with stochastic volatility and
negative correlation (Heston) can.
""")
st.write("---")


def model_iv(params):
    p = heston_fft_prices(S0, strikes, ttm, r, *params)
    return implied_vol(p, S0, strikes, ttm, r), p


# ── Manual fit ───────────────────────────────────────────────────────
st.header("2. Fit It Yourself")
st.markdown("Move the sliders and try to bring the Heston smile (red) onto the market quotes (black).")
c = st.columns(5)
g0 = c[0].slider("γ(0)", 0.005, 0.30, 0.05, step=0.005)
ka = c[1].slider("κ", 0.005, 0.50, 0.05, step=0.005)
la = c[2].slider("λ", 0.1, 8.0, 0.5, step=0.1)
st_ = c[3].slider("σ̃", 0.05, 2.0, 0.3, step=0.05)
rh = c[4].slider("ρ", -0.99, 0.99, -0.5, step=0.01)
manual = (g0, ka, la, st_, rh)
iv_man, _ = model_iv(manual)
rmse_man = np.sqrt(np.nanmean((iv_man - iv_mkt) ** 2))
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=strikes, y=iv_mkt, mode="markers", marker=dict(color="black"), name="Market"))
fig2.add_trace(go.Scatter(x=strikes, y=iv_man, name="Heston (your parameters)", line=dict(color="crimson", width=2.5)))
fig2.update_layout(xaxis_title="Strike", yaxis_title="Implied volatility", height=380)
st.plotly_chart(fig2, use_container_width=True)
st.metric("IV root-mean-squared error", f"{rmse_man * 100:.3f} vol points")
st.write("---")

# ── Automatic calibration ────────────────────────────────────────────
st.header("3. Automatic Calibration (Nelder-Mead)")
c1, c2, c3 = st.columns(3)
objective = c1.radio("Objective", ["Implied volatilities", "Prices"], horizontal=True)
start = c2.radio("Initial guess", ["Course default (0.05, 0.05, 0.5, 0.3, −0.5)", "Your slider values"])
maxiter = c3.select_slider("Max. iterations", [100, 200, 400, 800], value=400)
x0 = np.array([0.05, 0.05, 0.5, 0.3, -0.5]) if start.startswith("Course") else np.array(manual)
bounds = [(1e-4, 2), (1e-4, 2), (1e-3, 10), (1e-3, 2), (-0.999, 0.999)]

if st.button("🚀 Run calibration", type="primary"):
    progress = st.empty()
    hist = []
    t0 = time.perf_counter()

    def loss(p):
        iv, pr = model_iv(p)
        if objective == "Prices":
            val = np.mean((pr - prices) ** 2)
        else:
            val = np.mean((np.nan_to_num(iv, nan=0.0) - iv_mkt) ** 2)
        hist.append(val)
        if len(hist) % 25 == 0:
            progress.text(f"Evaluation {len(hist)}: objective = {val:.3e}  "
                          f"(γ0={p[0]:.4f}, κ={p[1]:.4f}, λ={p[2]:.3f}, σ̃={p[3]:.3f}, ρ={p[4]:.3f})")
        return val

    res = minimize(loss, x0, method="Nelder-Mead", bounds=bounds, options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-12})
    st.session_state["calib"] = dict(x=res.x, hist=np.array(hist), x0=x0, objective=objective,
                                     secs=time.perf_counter() - t0, source=source, nit=res.nit)
    progress.empty()

if "calib" in st.session_state and st.session_state["calib"]["source"] == source:
    cal = st.session_state["calib"]
    x = cal["x"]
    iv_fit, pr_fit = model_iv(x)
    iv_init, pr_init = model_iv(cal["x0"])
    names = ["γ(0)", "κ", "λ", "σ̃", "ρ"]
    table = pd.DataFrame({"Initial guess": cal["x0"], "Calibrated": x}, index=names)
    if true_params is not None:
        table["True (synthetic data)"] = true_params
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(table.style.format("{:.4f}"), use_container_width=True)
        st.metric("IV RMSE after calibration", f"{np.sqrt(np.nanmean((iv_fit - iv_mkt) ** 2)) * 100:.3f} vol pts")
        st.caption(f"{len(cal['hist'])} model evaluations in {cal['secs']:.1f} s, objective: {cal['objective']}.")
    with col2:
        fig3 = go.Figure(go.Scatter(y=cal["hist"], mode="lines", line=dict(color="steelblue")))
        fig3.update_layout(yaxis_type="log", xaxis_title="Function evaluation", yaxis_title="Objective", height=300,
                           title="Convergence of Nelder-Mead")
        st.plotly_chart(fig3, use_container_width=True)

    fig4 = make_subplots(rows=1, cols=2, subplot_titles=("Implied volatility", "Prices"))
    fig4.add_trace(go.Scatter(x=strikes, y=iv_mkt, mode="markers", marker=dict(color="black"), name="Market"), row=1, col=1)
    fig4.add_trace(go.Scatter(x=strikes, y=iv_init, name="Initial guess", line=dict(color="grey", dash="dash")), row=1, col=1)
    fig4.add_trace(go.Scatter(x=strikes, y=iv_fit, name="Calibrated Heston", line=dict(color="crimson", width=2.5)), row=1, col=1)
    fig4.add_trace(go.Scatter(x=strikes, y=prices, mode="markers", marker=dict(color="black"), showlegend=False), row=1, col=2)
    fig4.add_trace(go.Scatter(x=strikes, y=pr_init, line=dict(color="grey", dash="dash"), showlegend=False), row=1, col=2)
    fig4.add_trace(go.Scatter(x=strikes, y=pr_fit, line=dict(color="crimson", width=2.5), showlegend=False), row=1, col=2)
    fig4.update_xaxes(title_text="Strike")
    fig4.update_layout(height=400, legend=dict(x=0.3, y=1))
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown(f"""
    **Interpretation:** The calibrated model reproduces the skew with $\\hat\\rho = {x[4]:.2f}$ (strongly
    negative — the leverage effect) and a vol of vol $\\hat{{\\widetilde\\sigma}} = {x[3]:.2f}$ that
    generates the curvature. In price space the fit looks perfect even for the initial guess, which
    shows why prices are a poor yardstick: the differences that matter are invisible next to the large
    ITM prices. {'Compare the calibrated values with the true parameters used to generate the synthetic data: the skew-related parameters ρ and σ̃ are usually recovered well, while κ and λ are only weakly identified by one maturity — several combinations with similar κ/λ fit almost equally well.' if true_params is not None else 'With a single maturity, κ and λ are only weakly identified: several combinations with similar κ/λ fit almost equally well, so check the result from different starting points.'}
    Switching the objective to *prices* typically fits the ATM region but misses the wings.
    """)
else:
    st.caption("Press the button to run the calibration (each evaluation prices all strikes with one FFT).")
