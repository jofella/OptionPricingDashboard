import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import quad, solve_ivp

from util.black_scholes import bs_price, bs_greeks, price_by_integration, perpetual_put
from util.binomial import crr_price

st.title("📐 The Black-Scholes Model")
st.markdown(r"""
Under the martingale measure $Q$ the stock follows a geometric Brownian motion,

$$dS(t) = rS(t)\,dt + \sigma S(t)\,dW^Q(t), \qquad S(T) = S(t)\exp\Big(\big(r - \tfrac{\sigma^2}{2}\big)(T-t) + \sigma\big(W^Q(T) - W^Q(t)\big)\Big),$$

so $\log S(T)$ is normally distributed. The fair price of a payoff $f(S(T))$ is
$V(t) = e^{-r(T-t)}E_Q[f(S(T)) \mid \mathcal F_t] = v(t, S(t))$, and $v$ solves the **Black-Scholes PDE**

$$\partial_t v + rx\,\partial_x v + \tfrac12\sigma^2x^2\,\partial_{xx} v = rv, \qquad v(T, x) = f(x).$$

For calls and puts both routes lead to the closed form (Section 2.4)

$$V_{\text{call}}(t) = S(t)\Phi(d_1) - Ke^{-r(T-t)}\Phi(d_2), \qquad V_{\text{put}}(t) = Ke^{-r(T-t)}\Phi(-d_2) - S(t)\Phi(-d_1),$$

$$d_1 = \frac{\log(S(t)/K) + (r + \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}, \qquad d_2 = d_1 - \sigma\sqrt{T-t}.$$

The replicating portfolio holds $\varphi_1(t) = \partial_x v(t, S(t))$ shares — the option's **delta**.
""")

with st.expander("📖 Intuition: what do Φ(d₁) and Φ(d₂) mean?"):
    st.markdown(r"""
    - $\Phi(d_2) = Q(S(T) > K)$ is the risk-neutral probability that the call ends **in the money**.
      $Ke^{-rT}\Phi(d_2)$ is the present value of the strike you expect to pay.
    - $\Phi(d_1)$ is the probability of exercise under the **stock measure** (using the stock as
      numéraire) and simultaneously the **delta**: the number of shares in the replicating portfolio.
    - The call is therefore "long $\Phi(d_1)$ shares, short $Ke^{-rT}\Phi(d_2)$ in bonds" — exactly the
      replicating portfolio $\varphi_1 S + \varphi_0 B$.

    Notice that the real-world drift $\mu$ appears **nowhere**: two investors who disagree about the
    expected return of the stock still agree on the option price, because the option can be replicated.
    """)

with st.expander("📖 The Greeks — sensitivities used for hedging"):
    st.markdown(r"""
    | Greek | Definition | Call (BS) | Role |
    |---|---|---|---|
    | Delta $\Delta$ | $\partial V/\partial S$ | $\Phi(d_1)$ | shares held in the hedge |
    | Gamma $\Gamma$ | $\partial^2 V/\partial S^2$ | $\varphi(d_1)/(S\sigma\sqrt{\tau})$ | how fast the hedge must be adjusted |
    | Vega | $\partial V/\partial\sigma$ | $S\varphi(d_1)\sqrt\tau$ | exposure to volatility |
    | Theta $\Theta$ | $\partial V/\partial t$ | $-\frac{S\varphi(d_1)\sigma}{2\sqrt\tau} - rKe^{-r\tau}\Phi(d_2)$ | time decay |
    | Rho | $\partial V/\partial r$ | $K\tau e^{-r\tau}\Phi(d_2)$ | interest-rate exposure |
    | Speed | $\partial^3 V/\partial S^3$ | $-\frac{\Gamma}{S}\big(\frac{d_1}{\sigma\sqrt\tau} + 1\big)$ | appears in the bias of finite-difference Greeks |

    The PDE links them: $\Theta + rS\Delta + \tfrac12\sigma^2S^2\Gamma = rV$. A delta-hedged position
    earns $\Theta$ but pays for it through $\Gamma$ — the trade-off behind every hedging P&L.
    """)

st.write("---")

c1, c2, c3, c4, c5 = st.columns(5)
S0 = c1.number_input("S(0)", value=120.0, min_value=1.0, step=5.0)
K = c2.number_input("Strike K", value=100.0, min_value=1.0, step=5.0)
T = c3.number_input("Maturity T", value=1.0, min_value=0.05, step=0.25)
r = c4.number_input("Interest rate r", value=0.02, step=0.01, format="%.3f")
sigma = c5.number_input("Volatility σ", value=0.30, min_value=0.01, step=0.05)

# ── 1. Price & Greeks ────────────────────────────────────────────────
st.header("1. Prices and Greeks")
call = bs_price(S0, K, T, r, sigma, True)
put = bs_price(S0, K, T, r, sigma, False)
gk = bs_greeks(S0, K, T, r, sigma, True)
m = st.columns(6)
m[0].metric("Call", f"{call:.4f}")
m[1].metric("Put", f"{put:.4f}")
m[2].metric("Δ (call)", f"{gk['Delta']:.4f}")
m[3].metric("Γ", f"{gk['Gamma']:.5f}")
m[4].metric("Vega", f"{gk['Vega']:.3f}")
m[5].metric("Θ (call, p.a.)", f"{gk['Theta']:.3f}")

c1, c2 = st.columns(2)
opt = c1.radio("Option", ["Call", "Put"], horizontal=True)
quantity = c2.selectbox("Plot", ["Price", "Delta", "Gamma", "Vega", "Theta", "Speed"])
S_grid = np.linspace(0.3 * K, 2.0 * K, 400)
fig = go.Figure()
for tau, col in zip([T, 0.5 * T, 0.1 * T, 0.01 * T], ["#08306b", "#2171b5", "#6baed6", "#c6dbef"]):
    if quantity == "Price":
        y = bs_price(S_grid, K, tau, r, sigma, opt == "Call")
    else:
        y = bs_greeks(S_grid, K, tau, r, sigma, opt == "Call")[quantity]
    fig.add_trace(go.Scatter(x=S_grid, y=y, name=f"τ = {tau:.2f}", line=dict(color=col, width=2)))
if quantity == "Price":
    fig.add_trace(go.Scatter(x=S_grid, y=np.maximum(S_grid - K, 0) if opt == "Call" else np.maximum(K - S_grid, 0),
                             name="Payoff", line=dict(color="crimson", dash="dash")))
fig.add_vline(x=K, line_dash="dot", line_color="grey")
fig.update_layout(xaxis_title="S", yaxis_title=quantity, legend=dict(x=0, y=1), height=420)
st.plotly_chart(fig, use_container_width=True)
st.markdown(f"""
**Interpretation:** As time to maturity $\\tau = T - t$ shrinks, the price curve collapses onto the
payoff and the Greeks concentrate around the strike. With $\\tau \\to 0$ the delta becomes a step
function and gamma a spike at $K$: an at-the-money option close to expiry requires **violent hedge
adjustments**, which is why hedging errors in practice are dominated by short-dated ATM options.
Today the call delta is {gk['Delta']:.3f}, i.e. the hedge holds {gk['Delta']:.3f} shares per option.
""")
st.write("---")

# ── 2. Pricing by integration ────────────────────────────────────────
st.header("2. Pricing by Integration")
st.markdown(r"""
Because the density of $S(T)$ is known explicitly, **any** European payoff $f$ can be priced by a
one-dimensional integral:

$$V(t) = \frac{e^{-r(T-t)}}{\sqrt{2\pi}}\int_{-\infty}^{\infty} f\Big(S(t)e^{(r-\sigma^2/2)(T-t) + \sigma\sqrt{T-t}\,x}\Big)\,e^{-x^2/2}\,dx.$$

Differentiating under the integral sign gives the delta for any a.e. differentiable payoff:

$$\frac{\partial V(t)}{\partial S(t)} = \frac{e^{-r(T-t)}}{\sqrt{2\pi}}\int f'\big(\cdots\big)\,e^{(r-\sigma^2/2)(T-t)+\sigma\sqrt{T-t}x}\,e^{-x^2/2}\,dx.$$

""")

c1, c2, c3 = st.columns(3)
pay_type = c1.selectbox("Payoff", ["Power call (S^α − K)⁺", "Call", "Put", "Straddle |S − K|", "Digital 1{S ≥ K}"])
alpha = c2.slider("Power α", 0.5, 1.5, 1.2, step=0.05, disabled=not pay_type.startswith("Power"))
M_crr = c3.select_slider("Binomial steps M for comparison", [100, 250, 500, 1000, 2000], value=1000)

if pay_type.startswith("Power"):
    f = lambda s: np.maximum(s**alpha - K, 0)
    fprime = lambda s: alpha * s ** (alpha - 1) * (s**alpha > K)
elif pay_type == "Call":
    f = lambda s: np.maximum(s - K, 0); fprime = lambda s: (s > K) * 1.0
elif pay_type == "Put":
    f = lambda s: np.maximum(K - s, 0); fprime = lambda s: -1.0 * (s < K)
elif pay_type.startswith("Straddle"):
    f = lambda s: np.abs(s - K); fprime = lambda s: np.sign(s - K)
else:
    f = lambda s: (s >= K) * 1.0; fprime = None

t0 = time.perf_counter()
for _ in range(20):
    v_int = price_by_integration(f, S0, T, r, sigma)
t_int = (time.perf_counter() - t0) / 20
t0 = time.perf_counter()
v_crr = crr_price(S0, T, r, sigma, M_crr, f)
t_crr = time.perf_counter() - t0

if fprime is not None:
    tau = T
    drift = (r - 0.5 * sigma**2) * tau
    delta_int = np.exp(-r * tau) / np.sqrt(2 * np.pi) * quad(
        lambda x: fprime(S0 * np.exp(drift + sigma * np.sqrt(tau) * x)) * np.exp(drift + sigma * np.sqrt(tau) * x)
        * np.exp(-0.5 * x**2), -np.inf, np.inf, limit=200)[0]
else:
    h = 1e-3 * S0
    delta_int = (price_by_integration(f, S0 + h, T, r, sigma) - price_by_integration(f, S0 - h, T, r, sigma)) / (2 * h)

m = st.columns(5)
m[0].metric("Integration price", f"{v_int:.5f}")
m[1].metric(f"Binomial price (M={M_crr})", f"{v_crr:.5f}", f"{v_crr - v_int:+.5f}", delta_color="off")
m[2].metric("Integration time", f"{t_int * 1e3:.2f} ms")
m[3].metric("Binomial time", f"{t_crr * 1e3:.1f} ms", f"{t_crr / t_int:.0f}× slower", delta_color="off")
m[4].metric("Delta by integration", f"{delta_int:.5f}")

x_plot = np.linspace(-4, 4, 400)
ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * x_plot)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=x_plot, y=f(ST) * np.exp(-0.5 * x_plot**2) / np.sqrt(2 * np.pi), fill="tozeroy",
                          name="Integrand f(S_T(x)) φ(x)", line=dict(color="steelblue")))
fig2.update_layout(xaxis_title="x (standard normal variable)", yaxis_title="integrand", height=320)
st.plotly_chart(fig2, use_container_width=True)
st.markdown(f"""
**Interpretation:** The shaded area (times $e^{{-rT}}$) **is** the option price. Numerical quadrature
evaluates this smooth one-dimensional integrand to near machine precision in about
{t_int * 1e3:.1f} ms, whereas the binomial tree with $M = {M_crr}$ needs {t_crr * 1e3:.1f} ms and still
carries a discretisation error of {v_crr - v_int:+.1e}. The three approaches compared:

| Method | Payoffs | American? | Speed / accuracy |
|---|---|---|---|
| Closed form | calls, puts, few others | no | exact, instant |
| Binomial tree | any non-path-dependent (+ barriers) | **yes** | slow, $O(1/M)$ error |
| Integration | any non-path-dependent | no | fast and accurate |
""")
st.write("---")

# ── 3. Perpetual American put ────────────────────────────────────────
st.header("3. The Perpetual American Put")
st.markdown(r"""
A perpetual American put pays $(K - S(t))^+$ at **any** time the holder chooses — there is no
maturity. Its value $v(x)$ does not depend on time and solves the free-boundary problem

$$\tfrac{\sigma^2}{2}x^2 v''(x) + rx\,v'(x) - rv(x) = 0 \quad \text{for } x \ge x^*, \qquad v(x) = K - x \quad \text{for } x \le x^*,$$

with the optimal exercise boundary $x^* = \dfrac{2Kr}{2r + \sigma^2}$. We solve the ODE numerically by
rewriting it as a first-order system for $(v, v')$ and starting at $x^*$ with $v(x^*) = K - x^*$,
$v'(x^*) = -1$ (**smooth pasting**).
""")


def ode_fun(x, v):
    return [v[1], 2 * r / (sigma**2 * x**2) * (v[0] - x * v[1])]


S_max = 3 * K
S_grid = np.linspace(1e-6, S_max, 400)
v_closed, x_star = perpetual_put(S_grid, K, r, sigma)
sol = solve_ivp(ode_fun, (x_star, S_max), [K - x_star, -1.0], t_eval=S_grid[S_grid >= x_star], rtol=1e-8, atol=1e-10)
v_num = np.maximum(K - S_grid, 0.0)
v_num[S_grid >= x_star] = sol.y[0]
v_eu = bs_price(np.maximum(S_grid, 1e-6), K, 1.0, r, sigma, call=False)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=S_grid, y=v_num, name="Perpetual American put (ODE)", line=dict(color="crimson", width=3)))
fig3.add_trace(go.Scatter(x=S_grid, y=v_closed, name="Closed form", line=dict(color="black", dash="dot")))
fig3.add_trace(go.Scatter(x=S_grid, y=v_eu, name="European put, T = 1", line=dict(color="steelblue")))
fig3.add_trace(go.Scatter(x=S_grid, y=np.maximum(K - S_grid, 0), name="Payoff", line=dict(color="grey", dash="dash")))
fig3.add_vline(x=x_star, line_dash="dash", line_color="crimson", annotation_text=f"x* = {x_star:.2f}")
fig3.update_layout(xaxis_title="S", yaxis_title="Value", height=420, legend=dict(x=0.55, y=1))
st.plotly_chart(fig3, use_container_width=True)
st.markdown(f"""
**Interpretation:** Below $x^* = {x_star:.2f}$ it is optimal to exercise immediately, so the value equals
the payoff. Above $x^*$ the option is held, and the value curve leaves the payoff **tangentially**
(smooth pasting). The ODE solution coincides with the closed form
$v(x) = (K - x^*)(x/x^*)^{{-2r/\\sigma^2}}$ (max deviation
{np.max(np.abs(v_num - v_closed)):.1e}). The perpetual put is worth more than the European put for every
$S$ — it contains every finite-maturity American put as a special case. Note how small $x^*$ is for
$r = {r}$: with low interest rates, receiving $K$ early earns little, so early exercise only pays off
deep in the money. Increase $r$ to watch the boundary move towards $K$.
""")
