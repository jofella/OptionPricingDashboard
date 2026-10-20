import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import quad, solve_ivp

from util.black_scholes import bs_price, bs_greeks, price_by_integration, perpetual_put
from util.binomial import crr_price
from util.ui import takeaways, test_yourself

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

with st.expander("🎓 Deep dive: Brownian motion, Itô's formula and the solution of the GBM"):
    st.markdown(r"""
    **Brownian motion** $W$ has continuous paths, independent increments and $W(t) - W(s) \sim \mathcal N(0, t - s)$.
    Its paths are so rough that $(dW)^2 = dt$ — the quadratic variation over $[0, t]$ equals $t$. This is why
    ordinary calculus fails and **Itô's formula** has an extra second-order term: for $X$ with
    $dX = \mu\,dt + \sigma\,dW$ and smooth $f$,

    $$df(X) = f'(X)\,dX + \tfrac12 f''(X)\,\sigma^2\,dt.$$

    Apply it to $f(S) = \log S$ for the stock $dS = rS\,dt + \sigma S\,dW^Q$ (with $f' = 1/S$, $f'' = -1/S^2$):

    $$d\log S = \frac{dS}{S} - \frac{1}{2}\frac{\sigma^2 S^2}{S^2}dt = \big(r - \tfrac{\sigma^2}{2}\big)dt + \sigma\,dW^Q.$$

    The right-hand side has constant coefficients, so it integrates directly:
    $S(T) = S(0)\exp\big((r - \sigma^2/2)T + \sigma W^Q(T)\big)$. The famous $-\sigma^2/2$ is the **Itô correction**:
    it makes $E_Q[S(T)] = S(0)e^{rT}$, because $E[e^{\sigma W(T)}] = e^{\sigma^2 T/2}$.

    **Why $r$ and not $\mu$?** Under the real-world measure $P$ the drift is $\mu$. Girsanov's theorem allows changing
    to an equivalent measure $Q$ under which the discounted stock is a martingale; this only changes the drift to $r$,
    the volatility stays $\sigma$. Pricing happens under $Q$.
    """)

with st.expander("🎓 Deep dive: deriving the Black-Scholes PDE by hedging"):
    st.markdown(r"""
    Let $V(t) = v(t, S(t))$. By Itô's formula (now in two variables),

    $$dV = \Big(\partial_t v + rS\,\partial_x v + \tfrac12\sigma^2 S^2\,\partial_{xx} v\Big)dt + \sigma S\,\partial_x v\,dW^Q.$$

    Two equivalent arguments lead to the PDE:

    1. **Martingale argument (lecture).** The discounted price $e^{-rt}V(t)$ must be a $Q$-martingale, so its
       $dt$-term must vanish: $\partial_t v + rx\,\partial_x v + \tfrac12\sigma^2x^2\partial_{xx}v - rv = 0$.
    2. **Hedging argument.** Hold the option and short $\Delta = \partial_x v$ shares. The $dW$-terms cancel, so the
       portfolio is **riskless** over $dt$ and must earn the risk-free rate — which gives exactly the same equation.

    Together with the terminal condition $v(T, x) = f(x)$ this is the **Black-Scholes PDE**. The
    **Feynman-Kac formula** links both views: the solution of the PDE is
    $v(t, x) = E_Q\big[e^{-r(T-t)}f(S(T)) \mid S(t) = x\big]$. Chapter 5 solves the PDE numerically; Chapters 1, 3 and 4
    compute the expectation.
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
with st.expander("📖 Intuition: theta, gamma and the hedger's daily P&L"):
    st.markdown(r"""
    Plugging the Greeks into the PDE gives the identity

    $$\Theta + rS\Delta + \tfrac12\sigma^2S^2\Gamma = rV.$$

    For a delta-hedged position (long the option, short $\Delta$ shares, financed at rate $r$) this means: over a
    short period $\delta t$ with stock move $\delta S$ the P&L is approximately

    $$\tfrac12\Gamma\big((\delta S)^2 - \sigma^2S^2\delta t\big).$$

    The option holder **gains when the stock moves more than** the implied volatility predicts and loses (through
    time decay $\Theta$) when it moves less. Long options = long gamma = short theta. This trade-off is why a call
    close to expiry and at the money (huge gamma) is both the most profitable and the most dangerous position to hedge.
    """)

with st.expander("📖 Intuition: how the Greeks behave"):
    st.markdown(r"""
    - **Delta** of a call rises from 0 (deep out of the money) to 1 (deep in the money); put delta = call delta − 1
      (from parity).
    - **Gamma** and **vega** are the same for calls and puts (parity: they differ by a linear position in $S$ and a bond).
      Both peak near the money; gamma explodes near expiry, vega is largest for long maturities.
    - **Theta** is usually negative for long options: with every day less time remains for the option to end up
      in the money.
    - **Rho** is positive for calls (higher $r$ lowers the present value of the strike you pay) and negative for puts.
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
with st.expander("🎓 Deep dive: where the integration formula comes from"):
    st.markdown(r"""
    Write $S(T) = S(t)\exp\big((r - \sigma^2/2)\tau + \sigma\sqrt\tau\,X\big)$ with $\tau = T - t$ and $X \sim \mathcal N(0,1)$
    independent of the past. Then

    $$V(t) = e^{-r\tau}E_Q[f(S(T))] = e^{-r\tau}\int_{-\infty}^\infty f\Big(S(t)e^{(r - \sigma^2/2)\tau + \sigma\sqrt\tau x}\Big)\frac{e^{-x^2/2}}{\sqrt{2\pi}}\,dx.$$

    This is just the expectation written against the standard normal density. Plugging in $f(x) = (x - K)^+$ and
    splitting the integral at the point where the payoff becomes positive reproduces the closed form with
    $\Phi(d_1)$ and $\Phi(d_2)$ — a good exercise.

    **Why it beats the tree:** the integrand is smooth except at one kink, and adaptive quadrature (Gauss-Kronrod in
    `scipy.integrate.quad`) places its evaluation points where they are needed, reaching machine precision with a
    few hundred function evaluations. A tree with $M$ steps needs $O(M^2)$ operations for an $O(1/M)$ error.

    **Limitations:** the integral representation needs the density of $S(T)$ (available in Black-Scholes, not in
    Heston — see Chapter 3) and handles neither path dependence nor early exercise.
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

with st.expander("🎓 Deep dive: solving the perpetual put by hand"):
    st.markdown(r"""
    Without maturity, the value $v$ does not depend on $t$, so the PDE becomes the ODE
    $\tfrac12\sigma^2x^2v'' + rxv' - rv = 0$. It is an **Euler equation**: try $v(x) = x^p$, which gives

    $$\tfrac12\sigma^2p(p - 1) + rp - r = 0 \quad\Longleftrightarrow\quad (p - 1)\big(\tfrac12\sigma^2 p + r\big) = 0,$$

    so $p = 1$ or $p = -2r/\sigma^2$. The solution $x^1$ grows without bound, which is impossible for a put
    (bounded by $K$), so $v(x) = A\,x^{-2r/\sigma^2}$ in the continuation region $x > x^*$. Two conditions fix $A$ and $x^*$:

    - **value matching** $v(x^*) = K - x^*$,
    - **smooth pasting** $v'(x^*) = -1$ (the optimal boundary is where the value curve touches the payoff tangentially).

    Solving gives $x^* = \dfrac{2Kr}{2r + \sigma^2}$ and $A = (K - x^*)(x^*)^{2r/\sigma^2}$, i.e. the closed form shown above.
    Smooth pasting is not an extra assumption: any other boundary would give a lower value, so it is the optimality condition.
    """)

takeaways([
    "Under $Q$ the stock is a geometric Brownian motion with drift $r$; the $-\\sigma^2/2$ in the exponent is the Itô correction.",
    "The Black-Scholes PDE follows from the martingale property (or a riskless hedge); Feynman-Kac connects PDE and expectation.",
    "Greeks are linked through $\\Theta + rS\\Delta + \\frac12\\sigma^2S^2\\Gamma = rV$ — the hedger's P&L is a gamma-vs-theta trade-off.",
    "Pricing by integration handles any European payoff fast and accurately, but needs the density and cannot do early exercise.",
    "Perpetual American put: an ODE with a free boundary, solved by value matching and smooth pasting.",
])

test_yourself("bs", [
    (r"Use Itô's formula to show that $E_Q[S(T)] = S(0)e^{rT}$.",
     r"$S(T) = S(0)e^{(r - \sigma^2/2)T + \sigma W(T)}$ and $E[e^{\sigma W(T)}] = e^{\sigma^2T/2}$ (moment generating function of $\mathcal N(0, T)$), so $E_Q[S(T)] = S(0)e^{rT}e^{-\sigma^2T/2}e^{\sigma^2T/2} = S(0)e^{rT}$."),
    (r"Why does the real-world drift $\mu$ not appear in the Black-Scholes formula?",
     r"The option can be replicated by a self-financing portfolio; its cost is determined by the dynamics under $Q$, where the drift is $r$. Changing $\mu$ changes $P$ but not $Q$ (Girsanov changes only the drift)."),
    (r"Interpret $\Phi(d_1)$ and $Ke^{-rT}\Phi(d_2)$ in the call formula.",
     r"$\Phi(d_1)$ is the delta (shares in the hedge); $\Phi(d_2) = Q(S(T) > K)$ is the risk-neutral exercise probability, so $Ke^{-rT}\Phi(d_2)$ is the present value of the expected strike payment."),
    (r"Why are gamma and vega equal for a call and a put with the same strike and maturity?",
     r"By put-call parity the difference call − put $= S - Ke^{-r(T-t)}$ is linear in $S$ and does not depend on $\sigma$, so its second $S$-derivative and its $\sigma$-derivative vanish."),
    (r"What are the two conditions that determine the free boundary $x^*$ of the perpetual American put?",
     r"Value matching $v(x^*) = K - x^*$ and smooth pasting $v'(x^*) = -1$."),
])
