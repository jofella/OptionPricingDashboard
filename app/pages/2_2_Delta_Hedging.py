import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from util.black_scholes import bs_price, bs_greeks
from util.data_utils import load_dax_sample, get_log_returns
from util.ui import takeaways, test_yourself

st.title("🛡️ Delta Hedging in Practice")
st.markdown(r"""
In the Black-Scholes model a European option can be **replicated perfectly** by a self-financing
portfolio that holds $\varphi_1(t) = \partial_x v(t, S(t))$ shares and invests the rest in the bond:

$$V(t) = \varphi_0(t)B(t) + \varphi_1(t)S(t), \qquad \varphi_1(t) = \Phi(d_1) \text{ for a call.}$$

In reality the portfolio can only be **rebalanced at discrete times** $t_i$. At each $t_i$ the investor
buys $\varphi_1(t_i) - \varphi_1(t_{i-1})$ shares and finances them from the bond account $b$:

$$b(t_i) = b(t_{i-1})e^{r\Delta t} - \big(\varphi_1(t_i) - \varphi_1(t_{i-1})\big)S(t_i), \qquad b(0) = V(0) - \varphi_1(0)S(0).$$

We first hedge a call on the **DAX** (as in the tutorial), then move to a simulated Black-Scholes world
to separate the different sources of hedging error.
""")

with st.expander("📖 Why can't we hedge perfectly if the model says we can?"):
    st.markdown(r"""
    Three things break perfect replication in practice:

    1. **Discrete rebalancing.** Between two rebalancing dates the hedge is frozen while the option's
       delta keeps moving. The resulting error per period is roughly
       $\tfrac12\Gamma S^2\big((\Delta S/S)^2 - \sigma^2\Delta t\big)$ — zero on average but random.
       It shrinks like $1/\sqrt{n}$ in the number $n$ of rebalancing dates.
    2. **Wrong volatility.** The hedge uses an assumed $\sigma$; if the realised volatility differs, the
       hedger systematically gains or loses $\tfrac12\int_0^T \Gamma S^2(\sigma_{\text{real}}^2 - \sigma_{\text{hedge}}^2)\,dt$.
    3. **Model risk.** Real returns have jumps, fat tails and stochastic volatility — none of which the
       Black-Scholes delta accounts for. Transaction costs come on top.
    """)

with st.expander("📖 Intuition: what does 'self-financing' mean?"):
    st.markdown(r"""
    A strategy is **self-financing** if, after the initial investment, no money is added or withdrawn: every
    purchase of shares is paid for by selling bonds (or borrowing), and every sale of shares goes into the bond
    account. Formally, the value $V = \varphi_0B + \varphi_1S$ changes only because prices move,

    $$dV(t) = \varphi_0(t)\,dB(t) + \varphi_1(t)\,dS(t),$$

    not because of rebalancing. The bond update on this page,
    $b(t_i) = b(t_{i-1})e^{r\Delta t} - (\varphi_1(t_i) - \varphi_1(t_{i-1}))S(t_i)$, is exactly this rule in discrete
    time: the bond account earns interest and pays for the change in the share position.

    Replication means: a self-financing strategy whose terminal value equals the payoff. Its initial cost **is**
    the fair price — this is the economic content of the Black-Scholes formula.
    """)

st.write("---")

# ── 1. DAX ───────────────────────────────────────────────────────────
st.header("1. Hedging a DAX Call (Feb 2023 – Feb 2024)")
st.markdown(r"""
A European call on the DAX with strike $K = 16\,000$ is issued on 01 Feb 2023 and expires one year later.
We price and hedge it with Black-Scholes on the 255 daily closes of the tutorial, using
$t_i = i/255$, the EURIBOR rate of the issue date, and $\sigma = \sqrt{255}\cdot$ std of the daily log returns.
""")
dax = load_dax_sample()
S = dax.values.astype(float)
n = len(S)
lr = get_log_returns(S)

c1, c2, c3 = st.columns(3)
K = c1.number_input("Strike K", value=16000.0, step=250.0)
r = c2.number_input("Interest rate (EURIBOR 01 Feb 2023)", value=0.03116, step=0.005, format="%.5f")
reb = c3.select_slider("Rebalance every … trading days", [1, 2, 5, 10, 21, 63], value=1)
sigma = np.std(lr, ddof=1) * np.sqrt(n)
T = 1.0
t = np.arange(n) / n
dt = 1 / n

call = bs_price(S, K, T, r, sigma, True, t=t)
delta = bs_greeks(S, K, T, r, sigma, True, t=t)["Delta"]

phi = np.empty(n)
bond = np.empty(n)
phi[0] = delta[0]
bond[0] = call[0] - phi[0] * S[0]
for i in range(1, n):
    bond[i] = bond[i - 1] * np.exp(r * dt)
    phi[i] = delta[i] if i % reb == 0 else phi[i - 1]
    bond[i] -= (phi[i] - phi[i - 1]) * S[i]
port = phi * S + bond
err = port - call

m = st.columns(5)
m[0].metric("Trading days", n)
m[1].metric("σ (annualised, √n · std)", f"{sigma:.2%}")
m[2].metric("V(0)", f"{call[0]:,.1f}")
m[3].metric("Final hedging error", f"{err[-1]:,.1f}")
m[4].metric("Max |error|", f"{np.max(np.abs(err)):,.1f}")

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.35, 0.25], vertical_spacing=0.05,
                    subplot_titles=("DAX index", "Call price vs replicating portfolio", "Shares held φ₁ (delta)"))
fig.add_trace(go.Scatter(x=dax.index, y=S, name="DAX", line=dict(color="black", width=1.5)), row=1, col=1)
fig.add_hline(y=K, line_dash="dot", line_color="grey", row=1, col=1)
fig.add_trace(go.Scatter(x=dax.index, y=call, name="BS call price", line=dict(color="steelblue", width=2)), row=2, col=1)
fig.add_trace(go.Scatter(x=dax.index, y=port, name="Replicating portfolio", line=dict(color="crimson", dash="dash")), row=2, col=1)
fig.add_trace(go.Scatter(x=dax.index, y=delta, name="BS delta", line=dict(color="seagreen")), row=3, col=1)
fig.add_trace(go.Scatter(x=dax.index, y=phi, name="Held shares", line=dict(color="orange", shape="hv")), row=3, col=1)
fig.update_layout(height=750, legend=dict(orientation="h", y=-0.08))
st.plotly_chart(fig, use_container_width=True)

st.markdown(f"""
**Interpretation:** The call starts at {call[0]:,.1f} index points with delta {delta[0]:.2f}. As the DAX
rallies from {S[0]:,.0f} to {S[-1]:,.0f}, the delta climbs towards 1 — the option behaves more and more
like the index itself. The replicating portfolio tracks the call closely; with rebalancing every
{reb} day(s) the final hedging error is {err[-1]:+,.1f} points, i.e. {100 * err[-1] / call[0]:+.1f}% of the
initial premium.

The error is **not zero** although the Black-Scholes model claims perfect replication, because
(i) the hedge is rebalanced only {int(np.ceil(n / reb))} times, (ii) the volatility is estimated
from the same year (in reality it would have to be forecast), and (iii) the DAX does not follow a
geometric Brownian motion. Increase the rebalancing interval to see the error grow.
""")
with st.expander("🎓 Deep dive: where the hedging error comes from (the gamma P&L)"):
    st.markdown(r"""
    Between two rebalancing dates the hedger holds $\Delta$ shares while the option's value moves non-linearly.
    A second-order Taylor expansion of the option value combined with the PDE shows that the discounted P&L
    of the hedged position over one period of length $\delta t$ is approximately

    $$\delta\text{P\&L} \approx \tfrac12\,\Gamma\,S^2\Big(\Big(\tfrac{\delta S}{S}\Big)^2 - \sigma_{\text{hedge}}^2\,\delta t\Big).$$

    - If the realised squared return $(\delta S/S)^2$ equals its expectation $\sigma^2\delta t$, the errors average out.
      With $n$ rebalancing dates the individual errors are roughly independent with variance of order $1/n^2$,
      so their sum has standard deviation of order $1/\sqrt n$ — the slope in the right-hand plot.
    - If the realised volatility differs from the hedging volatility, the bracket has a **non-zero mean**, and the
      P&L accumulates to $\tfrac12\int_0^T \Gamma S^2(\sigma_{\text{real}}^2 - \sigma_{\text{hedge}}^2)\,dt$.
      Its sign depends only on whether realised volatility was above or below the hedging volatility —
      gamma is always positive for a long call or put.

    Since gamma is largest near the strike and close to maturity, both effects are concentrated on paths that
    hover around $K$ shortly before expiry.
    """)

with st.expander("📖 Intuition: what the DAX experiment does and does not show"):
    st.markdown(r"""
    The volatility used for the DAX hedge is estimated **from the same year** that is being hedged — in reality the
    hedger would have to forecast it, typically by using the implied volatility at inception. The DAX experiment
    therefore isolates mainly the discretisation effect and the non-normality of returns.

    One path is one experiment: a single year of data cannot tell whether the hedging error was "typical". That
    is why the second section repeats the experiment on thousands of simulated paths, where the distribution
    of the hedging error becomes visible.
    """)

st.write("---")

# ── 2. Simulated world ───────────────────────────────────────────────
st.header("2. Hedging Error in a Simulated Black-Scholes World")
st.markdown(r"""
To isolate the effect of discrete rebalancing, we now simulate paths from the Black-Scholes model
itself (so the model is *correct*) and hedge a call with $S(0) = K = 100$, $T = 1$. We also allow the
hedger to use a **different volatility** than the true one.
""")

c1, c2, c3, c4 = st.columns(4)
sig_true = c1.slider("True volatility", 0.05, 0.6, 0.20, step=0.01)
sig_hedge = c2.slider("Volatility used for hedging", 0.05, 0.6, 0.20, step=0.01)
n_reb = c3.select_slider("Rebalancing dates n", [4, 12, 52, 252, 1000], value=52)
n_paths = c4.select_slider("Paths", [500, 1000, 2000, 5000], value=2000)


@st.cache_data
def sim_hedge(sig_true, sig_hedge, n_steps, n_paths, r=0.02, S0=100.0, K=100.0, T=1.0, seed=1):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    logS = np.log(S0) + np.cumsum((r - 0.5 * sig_true**2) * dt + sig_true * np.sqrt(dt) * Z, axis=1)
    S = np.hstack([np.full((n_paths, 1), S0), np.exp(logS)])
    V0 = bs_price(S0, K, T, r, sig_hedge)
    phi = bs_greeks(S0, K, T, r, sig_hedge)["Delta"]
    bond = V0 - phi * S0
    for i in range(1, n_steps):
        bond = bond * np.exp(r * dt)
        new_phi = bs_greeks(S[:, i], K, T, r, sig_hedge, t=i * dt)["Delta"]
        bond -= (new_phi - phi) * S[:, i]
        phi = new_phi
    bond = bond * np.exp(r * dt)
    return phi * S[:, -1] + bond - np.maximum(S[:, -1] - K, 0), V0


@st.cache_data
def rmse_curve(sig, n_paths):
    ns = np.array([4, 8, 16, 32, 64, 128, 256, 512])
    return ns, np.array([np.sqrt(np.mean(sim_hedge(sig, sig, k, n_paths)[0] ** 2)) for k in ns])


pnl, V0_h = sim_hedge(sig_true, sig_hedge, n_reb, n_paths)
ns, rmse = rmse_curve(sig_true, n_paths)

m = st.columns(4)
m[0].metric("Premium charged", f"{V0_h:.3f}")
m[1].metric("Mean hedging P&L", f"{pnl.mean():+.3f}")
m[2].metric("Std of hedging P&L", f"{pnl.std():.3f}")
m[3].metric("True BS value", f"{bs_price(100, 100, 1, 0.02, sig_true):.3f}")

col1, col2 = st.columns(2)
with col1:
    fig2 = go.Figure(go.Histogram(x=pnl, nbinsx=60, marker_color="steelblue", opacity=0.8))
    fig2.add_vline(x=0, line_color="black")
    fig2.add_vline(x=pnl.mean(), line_dash="dash", line_color="crimson", annotation_text="mean")
    fig2.update_layout(xaxis_title="Terminal hedging P&L (portfolio − payoff)", yaxis_title="Count",
                       title=f"P&L distribution, n = {n_reb}")
    st.plotly_chart(fig2, use_container_width=True)
with col2:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=ns, y=rmse, mode="lines+markers", name="RMSE of hedge", line=dict(color="crimson")))
    fig3.add_trace(go.Scatter(x=ns, y=rmse[0] * np.sqrt(ns[0] / ns), name="∝ 1/√n", line=dict(color="grey", dash="dash")))
    fig3.update_layout(xaxis_type="log", yaxis_type="log", xaxis_title="Rebalancing dates n",
                       yaxis_title="RMSE", title="Convergence of discrete hedging (σ_hedge = σ_true)")
    st.plotly_chart(fig3, use_container_width=True)

st.markdown(f"""
**Interpretation:** With the correct volatility the P&L is centred at zero and its spread shrinks like
$1/\\sqrt{{n}}$ (right plot): halving the error needs **four times** as many trades. In the limit
$n \\to \\infty$ we recover the perfect continuous-time replication of the Black-Scholes theory.

With $\\sigma_{{\\text{{hedge}}}} = {sig_hedge:.2f}$ vs $\\sigma_{{\\text{{true}}}} = {sig_true:.2f}$ the mean P&L is
{pnl.mean():+.3f}. If you **sell** the option at a volatility above the realised one, you charge
more premium than the hedge consumes and earn a systematic profit — and lose if you under-price
volatility. This is why option traders talk about "being long or short volatility" and why
the volatility input, not the drift, is what matters for option pricing.
""")

with st.expander("🎓 Deep dive: transaction costs — why hedging continuously is impossible"):
    st.markdown(r"""
    The error shrinks like $1/\sqrt n$, so continuous rebalancing would remove it. But each rebalancing trade
    costs money (bid-ask spread, fees). With proportional costs $k$ per unit traded, the total trading volume of a
    delta hedge grows like $\sqrt n$, so total costs **grow** while the hedging error **shrinks**: there is an
    optimal rebalancing frequency.

    Leland (1985) showed that proportional costs can be approximately included by hedging with an enlarged
    volatility $\sigma_L^2 = \sigma^2\big(1 + \sqrt{2/\pi}\,\frac{k}{\sigma\sqrt{\delta t}}\big)$. More general
    frictions are the motivation for **deep hedging** (Chapter 6), where the cost term simply enters the loss
    function that the neural network minimises.
    """)

takeaways([
    "A self-financing replicating strategy costs exactly the option price; that is the meaning of the Black-Scholes price.",
    "Discrete rebalancing leaves a random error of order $1/\\sqrt{n}$; hedging with the wrong volatility leaves a systematic one.",
    "The hedging P&L is a gamma-weighted bet on realised vs. hedged volatility.",
    "Transaction costs make continuous hedging impossible: there is a trade-off between hedging error and costs.",
])

test_yourself("hedging", [
    (r"Write down the self-financing condition in discrete time for a portfolio of stock and bond.",
     r"$\varphi_0(t_{i-1})B(t_{i-1}) + \varphi_1(t_{i-1})S(t_{i-1}) = \varphi_0(t_i)B(t_{i-1}) + \varphi_1(t_i)S(t_{i-1})$: the new position is bought at the old prices with the old wealth — rebalancing neither adds nor withdraws money."),
    (r"How does the standard deviation of the hedging error scale with the number $n$ of rebalancing dates, and why?",
     r"Like $1/\sqrt n$: each period contributes an error with variance $O(\delta t^2) = O(1/n^2)$, the $n$ errors are roughly independent, so the total variance is $O(1/n)$."),
    (r"You sold a call at an implied volatility of 25%, the realised volatility turns out to be 18%, and you delta-hedge with 25%. Do you expect a profit or a loss?",
     r"A profit: the hedging P&L of the seller is $\frac12\int\Gamma S^2(\sigma_{\text{hedge}}^2 - \sigma_{\text{real}}^2)dt > 0$ (sign reversed relative to the option holder). You charged for more volatility than was delivered."),
    (r"Why is a short-dated at-the-money option the hardest to hedge?",
     r"Its gamma is largest, so the delta changes rapidly with the stock price; between rebalancing dates the frozen hedge becomes inaccurate fastest."),
])
