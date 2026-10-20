import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from util.heston import heston_char, bs_char, f_tilde_call, laplace_price, laplace_delta, fft_call_prices
from util.black_scholes import bs_price, bs_greeks
from util.ui import takeaways, test_yourself

st.title("〰️ Option Pricing with Integral Transforms")
st.markdown(r"""
In many models (Heston, Lévy models, …) the density of $X(T) = \log S(T)$ is unknown, but its
**characteristic function** $\chi(u) = E_Q[e^{iuX(T)}]$ is available in closed form. Transform methods
exploit this (Chapter 3).

### The idea: write the payoff as a sum of exponentials
For an exponential payoff $e^{zX(T)}$ with $z\in\mathbb C$ the price is trivial:
$e^{-rT}E_Q[e^{zX(T)}] = e^{-rT}\chi(-iz)$. A general payoff $f(x)$ is decomposed into such exponentials
with its **bilateral Laplace transform** $\tilde f(z) = \int f(x)e^{-zx}dx$:

$$f(x) = \frac{1}{2\pi i}\int_{R-i\infty}^{R+i\infty}\tilde f(z)e^{zx}\,dz \quad\Longrightarrow\quad V(0) = \frac{e^{-rT}}{\pi}\int_0^\infty \operatorname{Re}\Big(\tilde f(R+iu)\,\chi(u-iR)\Big)\,du.$$

For the call $f(x) = (e^x - K)^+$ the transform exists only for $\operatorname{Re}(z) = R > 1$ and equals

$$\tilde f(z) = \frac{K^{1-z}}{z(z-1)}.$$

""")

with st.expander("📖 Intuition: why do we need the damping constant R?"):
    st.markdown(r"""
    The call payoff grows like $e^x$ as $x\to\infty$, so $\int f(x)e^{-zx}dx$ only converges if the
    factor $e^{-Rx}$ beats that growth, i.e. $R > 1$. The constant $R$ effectively **damps** the payoff
    before transforming it. Inside the admissible strip the result does not depend on $R$ (up to
    numerical error), but $R$ influences how fast the integrand decays and oscillates.

    What happens with other values of $R$? The very same formula $K^{1-z}/(z(z-1))$ is the transform of
    a **different function** on each strip:
    - $R > 1$: $(e^x - K)^+$ — the call,
    - $0 < R < 1$: $(e^x - K)^+ - e^x$ — so the integral returns call $-\,S(0)$,
    - $R < 0$: $(K - e^x)^+$ — the put.

    Try it with the slider below.
    """)

with st.expander("📖 The FFT trick (Section 3.2)"):
    st.markdown(r"""
    Write the log-strike as $\kappa = \log K$. The Laplace price is
    $V_\kappa(0) = \frac{e^{-rT + (1-R)\kappa}}{\pi}\int_0^\infty \operatorname{Re}\big(e^{-iu\kappa}g(u)\big)du$
    with $g(u) = \frac{\chi(u - iR)}{(R+iu)(R+iu-1)}$ — the strike only enters through $e^{-iu\kappa}$.
    Discretising the integral with the midpoint rule on $u_n = (n-\tfrac12)\delta$, $\delta = M/N$, and
    choosing log-strikes $\kappa_m = \kappa_1 + (m-1)\frac{2\pi}{M}$ turns the sum into a
    **discrete Fourier transform**

    $$\hat x_m = \sum_{n=1}^N x_n e^{-2\pi i(n-1)(m-1)/N}, \qquad x_n = g(u_n)\,\delta\,e^{-i(n-1)\delta\kappa_1},$$

    which the FFT evaluates in $O(N\log N)$ — giving prices for **all $N$ log-strikes at once**.
    Prices at arbitrary strikes are then obtained by interpolation. Trade-off: $\delta$ small means a
    fine $u$-grid (accurate integral) but a coarse strike grid $2\pi/M$; large $M$ needs large $N$.
    """)

with st.expander("🎓 Deep dive: computing the transform of the call payoff"):
    st.markdown(r"""
    For $f(x) = (e^x - K)^+$ the payoff is non-zero only for $x > \log K$, so

    $$\tilde f(z) = \int_{\log K}^\infty (e^x - K)e^{-zx}\,dx = \Big[\frac{e^{(1-z)x}}{1 - z} + \frac{Ke^{-zx}}{z}\Big]_{\log K}^{\infty}.$$

    The upper limit vanishes only if $\operatorname{Re}(z) > 1$ (both exponentials must decay). Evaluating at
    $x = \log K$ gives $-\frac{K^{1-z}}{1-z} - \frac{K^{1-z}}{z} = \frac{K^{1-z}}{z(z-1)}$. The pricing formula then
    follows from the **inversion formula** of the bilateral Laplace transform,

    $$f(x) = \frac{1}{2\pi i}\int_{R - i\infty}^{R + i\infty}\tilde f(z)e^{zx}\,dz,$$

    by taking expectations inside the integral (Fubini): $E_Q[e^{zX(T)}] = \chi(-iz)$. Writing $z = R + iu$ and using
    the symmetry of the integrand under $u \mapsto -u$ (complex conjugation) halves the integration range and
    leaves only the real part.
    """)

with st.expander("📖 Intuition: what is a characteristic function, and why is it so convenient?"):
    st.markdown(r"""
    $\chi(u) = E[e^{iuX}]$ is the expectation of an oscillating, bounded function — it always exists and determines the
    distribution uniquely. For the models used in finance it is often available in closed form even when the
    density is not:

    | Model | $\chi(u)$ of $\log S(T)$ | density |
    |---|---|---|
    | Black-Scholes | $\exp\big(iu m - \tfrac12\sigma^2Tu^2\big)$, $m = \log S(0) + (r - \sigma^2/2)T$ | normal — explicit |
    | Heston | exponential-affine in $\log S(0), \gamma(0)$ | not explicit |
    | Lévy models (NIG, Variance Gamma, …) | $\exp(T\,\psi(u))$ | often not explicit |

    The trick of the chapter: **decompose the payoff into exponentials** (via the Laplace transform), price each
    exponential with $\chi$, and add them up (the integral).
    """)

st.write("---")
model = st.radio("Model", ["Heston", "Black-Scholes"], horizontal=True)
c = st.columns(8)
S0 = c[0].number_input("S(0)", value=120.0, step=5.0)
r = c[1].number_input("r", value=0.02, step=0.01, format="%.3f")
T = c[2].number_input("T", value=1.0, min_value=0.1, step=0.25)
if model == "Heston":
    gam0 = c[3].number_input("γ(0)", value=0.09, min_value=0.001, step=0.01, format="%.3f")
    kappa = c[4].number_input("κ", value=0.09, min_value=0.001, step=0.01, format="%.3f")
    lamb = c[5].number_input("λ", value=2.5, min_value=0.01, step=0.25)
    sig_t = c[6].number_input("σ̃", value=0.3, min_value=0.01, step=0.05)
    rho = c[7].number_input("ρ", value=-0.5, min_value=-0.99, max_value=0.99, step=0.1)
    params = (gam0, kappa, lamb, sig_t, rho)
    char = lambda u: heston_char(u, S0, T, r, *params)
else:
    sigma = c[3].number_input("σ", value=0.3, min_value=0.01, step=0.05)
    params = (sigma,)
    char = lambda u: bs_char(u, S0, T, r, sigma)
st.write("---")

# ── 1. Laplace ───────────────────────────────────────────────────────
st.header("1. Laplace Transform Pricing")
c1, c2 = st.columns(2)
K = c1.slider("Strike K", float(0.4 * S0), float(2.0 * S0), 100.0, step=1.0)
R = c2.slider("Damping constant R", -3.0, 5.0, 1.5, step=0.1)
if abs(R) < 0.05 or abs(R - 1) < 0.05:
    st.warning("R = 0 and R = 1 are poles of the transform — pick a value away from them.")
else:
    u = np.linspace(0, 60, 1200)
    integrand = np.real(f_tilde_call(R + 1j * u, K) * char(u - 1j * R))
    val = laplace_price(K, char, r, T, R=R)
    call_ref = laplace_price(K, char, r, T, R=1.5)
    put_ref = call_ref - S0 + K * np.exp(-r * T)
    m = st.columns(4)
    m[0].metric(f"Laplace result (R = {R})", f"{val:.5f}")
    m[1].metric("Call price", f"{call_ref:.5f}")
    m[2].metric("Call − S(0)", f"{call_ref - S0:.5f}")
    m[3].metric("Put price (via parity)", f"{put_ref:.5f}")
    if R > 1:
        st.success("R > 1: the integral prices the **call**.")
    elif R > 0:
        st.info("0 < R < 1: the same formula now represents (eˣ − K)⁺ − eˣ, so the result is **call − S(0)**.")
    else:
        st.info("R < 0: the formula represents (K − eˣ)⁺, so the result is the **put** price.")
    fig = go.Figure(go.Scatter(x=u, y=integrand, line=dict(color="steelblue")))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(xaxis_title="u", yaxis_title="Re( f̃(R+iu) χ(u−iR) )", height=330,
                      title="Integrand of the Laplace pricing formula")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"""
    **Interpretation:** The integrand decays quickly in $u$ (it is essentially the Fourier transform of
    a smooth, damped density), so truncating the integral at $u_{{\\max}} = 100$ is harmless. Larger
    $|R|$ shifts mass to the start of the integrand and makes it more peaked, while values close to
    the poles $R = 0, 1$ make it decay slowly — both affect numerical accuracy, not the exact value.
    """)

with st.expander("📖 Intuition: what changes with R in practice"):
    st.markdown(r"""
    Inside the valid strip the price does **not** depend on $R$ mathematically — but numerically it matters:

    - $R$ close to 1: the factor $1/(z(z-1))$ is large near $u = 0$ (close to the pole), the integrand is peaked and
      hard to integrate accurately.
    - $R$ very large: $\chi(u - iR) = E[S(T)^R e^{iu\log S(T)}]$ involves high moments $E[S(T)^R]$, which may not exist
      (fat-tailed models) or become numerically huge.

    A value around $R = 1.5$ (tutorial choice) is a common compromise. For heavy-tailed models, $R$ must lie in the
    range where the moment $E[S(T)^R]$ is finite.
    """)

st.write("---")

# ── 2. FFT vs Laplace ────────────────────────────────────────────────
st.header("2. Fast Fourier Transform: All Strikes at Once")
c1, c2, c3 = st.columns(3)
n_strikes = c1.select_slider("Number of strikes", [10, 25, 50, 100, 200], value=100)
logN = c2.slider("log₂ N (FFT size)", 10, 16, 15)
M_fft = c3.select_slider("Upper integration bound M", [500, 1000, 2500, 5000, 10000], value=5000)
K_arr = np.linspace(0.4 * S0, 2.5 * S0, n_strikes)


@st.cache_data
def laplace_all(K_arr, S0, r, T, params, model):
    ch = (lambda u: heston_char(u, S0, T, r, *params)) if model == "Heston" else (lambda u: bs_char(u, S0, T, r, *params))
    t0 = time.perf_counter()
    prices = np.array([laplace_price(k, ch, r, T) for k in K_arr])
    return prices, time.perf_counter() - t0


lap_prices, t_lap = laplace_all(K_arr, S0, r, T, params, model)
t0 = time.perf_counter()
fft_prices, K_grid, V_grid = fft_call_prices(K_arr, char, r, T, R=1.5, M=float(M_fft), N=2**logN)
t_fft = time.perf_counter() - t0

m = st.columns(5)
m[0].metric("Laplace time (quad per strike)", f"{t_lap * 1e3:.0f} ms")
m[1].metric("FFT time (all strikes)", f"{t_fft * 1e3:.1f} ms", f"{t_lap / t_fft:.0f}× faster", delta_color="off")
m[2].metric("Max |FFT − Laplace|", f"{np.max(np.abs(fft_prices - lap_prices)):.1e}")
m[3].metric("δ = M/N", f"{M_fft / 2**logN:.4f}")
m[4].metric("Log-strike spacing 2π/M", f"{2 * np.pi / M_fft:.5f}")

fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Call prices across strikes", "FFT − Laplace"))
fig2.add_trace(go.Scatter(x=K_arr, y=lap_prices, mode="markers", name="Laplace (quad)", marker=dict(color="steelblue", size=6)), row=1, col=1)
fig2.add_trace(go.Scatter(x=K_arr, y=fft_prices, mode="lines", name="FFT", line=dict(color="crimson")), row=1, col=1)
if model == "Black-Scholes":
    fig2.add_trace(go.Scatter(x=K_arr, y=bs_price(S0, K_arr, T, r, params[0]), name="BS formula",
                              line=dict(color="black", dash="dot")), row=1, col=1)
fig2.add_trace(go.Scatter(x=K_arr, y=fft_prices - lap_prices, name="Difference", line=dict(color="orange")), row=1, col=2)
fig2.update_xaxes(title_text="K")
fig2.update_layout(height=400, legend=dict(x=0.25, y=1))
st.plotly_chart(fig2, use_container_width=True)
st.markdown(f"""
**Interpretation:** One FFT of size $N = 2^{{{logN}}}$ produces prices on {2**logN:,} log-strikes in
{t_fft * 1e3:.1f} ms; pricing {n_strikes} strikes one-by-one with adaptive quadrature took
{t_lap * 1e3:.0f} ms. The gain grows with the number of strikes, which is exactly what calibration needs
(hundreds of model evaluations, each over the whole strike range). Reduce $N$ or $M$ to see the
accuracy trade-off: with too small $N$ the integration step $\\delta = M/N$ becomes coarse and the
error grows; with too small $M$ the integral is truncated too early.
""")
with st.expander("🎓 Deep dive: why the FFT is fast, and how the grids are coupled"):
    st.markdown(r"""
    A discrete Fourier transform $\hat x_m = \sum_{n=1}^N x_n e^{-2\pi i(n-1)(m-1)/N}$ for all $m$ costs $N^2$ operations
    if done naively. The **Fast Fourier Transform** (Cooley-Tukey) splits the sum into even and odd indices
    recursively and needs only $O(N\log N)$ operations: for $N = 2^{15}$ that is about $5\cdot10^5$ instead of $10^9$.

    To use it, the integration grid in $u$ and the log-strike grid must fit together:

    $$u_n = (n - \tfrac12)\delta, \qquad \kappa_m = \kappa_1 + (m - 1)\Delta\kappa, \qquad \delta\cdot\Delta\kappa = \frac{2\pi}{N}.$$

    With $\delta = M/N$ this is $\Delta\kappa = 2\pi/M$ — so the upper integration bound $M$ sets the **strike resolution**,
    and $N$ sets the **integration accuracy** $\delta$. Fine integration *and* fine strike spacing require a large $N$.
    Strikes between grid points are obtained by interpolation, which adds a small error.
    """)

st.write("---")

# ── 3. Hedging ───────────────────────────────────────────────────────
st.header("3. Hedging Strategies via the Laplace Transform")
st.markdown(r"""
Since $S(0)$ enters the characteristic function only through $e^{iu\log S(0)}$, differentiating under
the integral is easy:

$$\frac{\partial V(0)}{\partial S(0)} = \frac{e^{-rT}}{\pi}\int_0^\infty\operatorname{Re}\Big(\tilde f(R+iu)\,\frac{R+iu}{S(0)}\,\chi(u-iR)\Big)\,du.$$

The Heston model is **incomplete**: volatility risk cannot be hedged with the stock alone. For
$\rho = 0$ (the lecture setting) the delta above is the variance-minimising stock position; for
$\rho \neq 0$ the minimum-variance hedge adds a correction $\frac{\rho\widetilde\sigma}{S}\,\partial_\gamma v$,
because stock moves then also carry information about variance moves.
""")


@st.cache_data
def deltas(K_arr, S0, r, T, params, model):
    ch = (lambda u: heston_char(u, S0, T, r, *params)) if model == "Heston" else (lambda u: bs_char(u, S0, T, r, *params))
    return np.array([laplace_delta(k, ch, S0, r, T) for k in K_arr])


K_d = np.linspace(0.4 * S0, 2.5 * S0, 60)
d_lap = deltas(K_d, S0, r, T, params, model)
sig_ref = np.sqrt(params[1] / params[2] + (params[0] - params[1] / params[2]) * (1 - np.exp(-params[2] * T)) / (params[2] * T)) \
    if model == "Heston" else params[0]
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=K_d, y=d_lap, name=f"{model} delta (Laplace)", line=dict(color="crimson", width=2.5)))
fig3.add_trace(go.Scatter(x=K_d, y=bs_greeks(S0, K_d, T, r, sig_ref)["Delta"], name=f"BS delta, σ = {sig_ref:.2%}",
                          line=dict(color="steelblue", dash="dash")))
fig3.update_layout(xaxis_title="Strike K", yaxis_title="φ₁(0)", height=380)
st.plotly_chart(fig3, use_container_width=True)
st.markdown(f"""
**Interpretation:** Deep in-the-money calls (small $K$) need one share per option, deep out-of-the-money
calls almost none; in between the delta falls smoothly with the strike. {'In the Heston model the delta differs from the Black-Scholes delta with comparable average volatility, because the Heston distribution of S(T) is skewed and fat-tailed — the transform method captures this automatically at no extra cost.' if model == 'Heston' else 'In the Black-Scholes model the transform delta coincides with Φ(d₁), a useful check of the implementation.'}
""")

with st.expander("🎓 Deep dive: Greeks from transforms"):
    st.markdown(r"""
    Any parameter that enters only $\chi$ can be differentiated under the integral:

    - **Delta:** $\partial_{S_0}\chi(u - iR) = \frac{R + iu}{S_0}\chi(u - iR)$, since $\chi$ depends on $S_0$ only through $S_0^{\,i(u - iR)}$.
    - **Gamma:** differentiate again — the factor becomes $\frac{(R+iu)(R+iu-1)}{S_0^2}$, which conveniently cancels
      the denominator of $\tilde f$.
    - **Vega-type sensitivities** (e.g. with respect to $\gamma(0)$): $\partial_{\gamma_0}\chi = \psi_2\,\chi$ in the Heston model.

    So hedging ratios cost one additional integral of the same type as the price — no simulation or finite differencing needed.
    """)

takeaways([
    "Transform methods only need the characteristic function, which is known for many models without a closed-form density.",
    "The Laplace transform of the call exists for $\\operatorname{Re}(z) > 1$; the damping constant $R$ must lie in that strip.",
    "The FFT computes prices for thousands of log-strikes in one $O(N\\log N)$ step — ideal for calibration.",
    "Greeks follow by differentiating the characteristic function under the integral.",
])

test_yourself("transforms", [
    (r"Derive the Laplace transform of the call payoff $(e^x - K)^+$ and state where it exists.",
     r"$\tilde f(z) = \int_{\log K}^\infty (e^x - K)e^{-zx}dx = \frac{K^{1-z}}{z(z-1)}$, valid for $\operatorname{Re}(z) > 1$."),
    (r"Why does the pricing formula contain $\chi(u - iR)$ rather than $\chi(u)$?",
     r"We need $E_Q[e^{zX(T)}]$ with $z = R + iu$; since $\chi(v) = E[e^{ivX}]$, this equals $\chi(-iz) = \chi(u - iR)$."),
    (r"What do the parameters $N$ and $M$ (upper bound) control in the FFT method?",
     r"$\delta = M/N$ is the integration step, $\Delta\kappa = 2\pi/M$ the log-strike spacing. $M$ trades integration range against strike resolution; $N$ (with fixed $M$) refines the integration."),
    (r"Why is the FFT method so useful for calibration?",
     r"Calibration evaluates the model at many parameter sets, each time for many strikes. The FFT prices all strikes of one maturity at once in $O(N\log N)$."),
])
