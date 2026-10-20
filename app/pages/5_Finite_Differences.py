import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import inspect
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from util.finite_differences import solve_heat_theta, thomas
from util.black_scholes import bs_price
from util.binomial import crr_price, payoff_fn

st.title("🔲 Finite Difference Methods")
st.markdown(r"""
Finite differences solve the Black-Scholes PDE directly on a grid and deliver the **whole price surface**
$v(t, S)$ at once (Chapter 5). The PDE is first transformed into the **heat equation**:

$$x = \log(S/K), \qquad \tilde t = \tfrac{\sigma^2}{2}(T - t), \qquad q = \tfrac{2r}{\sigma^2}, \qquad v(t, S) = K\,e^{-\frac{q-1}{2}x - \frac{(q+1)^2}{4}\tilde t}\,w(\tilde t, x),$$

$$\partial_{\tilde t}\,w = \partial_{xx}\,w, \qquad w(0, x) = \big(e^{\frac{x}{2}(q+1)} - e^{\frac{x}{2}(q-1)}\big)^+ \text{ (call)}.$$

The $x$-axis is truncated to $[a, b]$ with $m$ steps $\Delta x = (b-a)/m$, time to $\nu_{\max}$ steps
$\Delta\tilde t = \sigma^2T/(2\nu_{\max})$, and $\lambda = \Delta\tilde t/\Delta x^2$. Replacing derivatives by
difference quotients gives the **$\theta$-scheme**

$$w_i^{(\nu+1)} - w_i^{(\nu)} = \lambda\Big[\theta\,\delta^2 w_i^{(\nu+1)} + (1-\theta)\,\delta^2 w_i^{(\nu)}\Big], \qquad \delta^2 w_i = w_{i+1} - 2w_i + w_{i-1},$$

with $\theta = 0$ (**explicit**), $\theta = 1$ (**implicit**) and $\theta = \tfrac12$ (**Crank-Nicolson**).
""")

with st.expander("📖 Boundary conditions: the part everyone gets wrong once"):
    st.markdown(r"""
    The grid needs values at $x = a$ and $x = b$ for all times. For a call:
    - $S \to 0$: the call is worthless, $v = 0$, i.e. $w(\tilde t, a) = 0$.
    - $S \to \infty$: $v(t, S) \approx S - Ke^{-r(T-t)}$, which becomes
      $w(\tilde t, b) = e^{\frac{q+1}{2}b + \frac{(q+1)^2}{4}\tilde t} - e^{\frac{q-1}{2}b + \frac{(q-1)^2}{4}\tilde t}$.

    The tutorial first (deliberately) uses $w = 0$ at **both** ends: the computed "call" then decreases in
    $S$ near the upper edge, which is clearly wrong — try it below. The boundary values are also only
    *approximations* at a finite $b$, so the interval must be wide enough: the course grid
    $[a, b] = [-0.7, 0.4]$ ends at $S = Ke^{0.4} \approx 1.49K$, where the call still has substantial time value.
    """)

with st.expander("📖 Stability: why the explicit scheme can explode"):
    st.markdown(r"""
    The explicit scheme $w^{(\nu+1)} = A\,w^{(\nu)}$ with tridiagonal $A = \operatorname{tridiag}(\lambda, 1-2\lambda, \lambda)$
    is stable only if all eigenvalues of $A$ lie in $[-1, 1]$, which is the case iff

    $$\lambda = \frac{\Delta\tilde t}{\Delta x^2} \le \frac12 \quad\Longleftrightarrow\quad m \le (b-a)\sqrt{\frac{\nu_{\max}}{\sigma^2 T}}.$$

    If violated, rounding errors in the highest-frequency mode are amplified by $|1 - 4\lambda| > 1$ in every
    step and grow exponentially. Implicit and Crank-Nicolson schemes are **unconditionally stable**, but
    need a tridiagonal linear solve per step (Thomas algorithm, $O(m)$ operations).
    """)

st.write("---")
c = st.columns(4)
r = c[0].number_input("r", value=0.02, step=0.01, format="%.3f")
sigma = c[1].number_input("σ", value=0.30, min_value=0.05, step=0.05)
T = c[2].number_input("T", value=1.0, min_value=0.1, step=0.25)
K = c[3].number_input("Strike K", value=100.0, step=5.0)
c = st.columns(4)
grid = c[0].radio("x-interval [a, b]", ["Wide: [−1.5, 1.0]", "Course: [−0.7, 0.4]"])
a, b = (-1.5, 1.0) if grid.startswith("Wide") else (-0.7, 0.4)
nu_max = c[1].select_slider("Time steps ν_max", [500, 1000, 2000, 4000], value=2000)
m_stable = int((b - a) * np.sqrt(nu_max / (sigma**2 * T)))
c[2].metric("Largest stable m (explicit)", m_stable)


def surface_fig(t, S, V, title, zmax=None, S_lim=(0.4, 2.2)):
    keep = (S >= S_lim[0] * K) & (S <= S_lim[1] * K)
    ti = np.unique(np.linspace(0, len(t) - 1, 60).astype(int))
    si = np.where(keep)[0]
    si = si[np.unique(np.linspace(0, len(si) - 1, min(80, len(si))).astype(int))]
    Z = V[np.ix_(ti, si)]
    if zmax is not None:
        Z = np.clip(Z, -zmax, zmax)
    fig = go.Figure(go.Surface(x=S[si], y=t[ti], z=Z, colorscale="RdBu", reversescale=True, showscale=False))
    fig.update_layout(title=title, height=480, margin=dict(l=0, r=0, t=40, b=0),
                      scene=dict(xaxis_title="S", yaxis_title="t", zaxis_title="V", camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8))))
    return fig


# ── 1. Explicit scheme ───────────────────────────────────────────────
st.header("1. The Explicit Scheme and Its Stability Limit")
c1, c2 = st.columns(2)
m_exp = c1.slider("Space steps m", 20, int(2.5 * m_stable), m_stable, step=1)
bc = c2.radio("Boundary at x = b", ["Correct: v ≈ S − K·exp(−r(T−t))", "Naive: v = 0"], horizontal=True)
t_e, S_e, V_e, lam_e = solve_heat_theta(r, sigma, T, K, a, b, m_exp, nu_max, theta=0.0, kind="call",
                                        zero_boundary=bc.startswith("Naive"))
bs_e = bs_price(S_e, K, T, r, sigma)
win = (S_e > 0.6 * K) & (S_e < 1.4 * K)
err_e = np.max(np.abs(V_e[0][win] - bs_e[win]))
m = st.columns(3)
m[0].metric("λ = Δt̃ / Δx²", f"{lam_e:.3f}", "stable" if lam_e <= 0.5 else "UNSTABLE", delta_color="normal" if lam_e <= 0.5 else "inverse")
m[1].metric("Max error at t = 0 for S ∈ [0.6K, 1.4K]", f"{err_e:.2e}" if np.isfinite(err_e) else "overflow")
m[2].metric("S range of the grid", f"[{K * np.exp(a):.1f}, {K * np.exp(b):.1f}]")

col1, col2 = st.columns([3, 2])
with col1:
    st.plotly_chart(surface_fig(t_e, S_e, V_e, "Explicit scheme: v(t, S)", zmax=3 * K), use_container_width=True)
with col2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_e, y=np.clip(V_e[0], -3 * K, 3 * K), name="FD, t = 0", line=dict(color="crimson", width=2)))
    fig.add_trace(go.Scatter(x=S_e, y=bs_e, name="Black-Scholes", line=dict(color="black", dash="dash")))
    fig.update_layout(xaxis_title="S", yaxis_title="V(0, S)", height=480, legend=dict(x=0, y=1))
    st.plotly_chart(fig, use_container_width=True)

if lam_e > 0.5:
    st.markdown(f"""
    **Interpretation:** With $m = {m_exp} > {m_stable}$ we have $\\lambda = {lam_e:.3f} > 1/2$. The error mode with
    alternating signs is multiplied by $|1 - 4\\lambda| = {abs(1 - 4 * lam_e):.2f}$ in each of the {nu_max} steps,
    so tiny rounding errors explode into the saw-tooth garbage you see (values are clipped for display).
    A *finer* space grid made the result *worse* — the hallmark of instability, not of inaccuracy.
    """)
elif bc.startswith("Naive"):
    st.markdown("""
    **Interpretation:** With $v = 0$ imposed at the upper edge, the computed "call" is forced back to zero
    for large $S$: the surface bends down and the price is no longer increasing in $S$. The error then
    diffuses inwards from the boundary over time. The PDE scheme is fine — the boundary condition is wrong.
    """)
else:
    st.markdown(f"""
    **Interpretation:** At the stability limit the explicit scheme reproduces the Black-Scholes price with a
    maximum error of {err_e:.1e} in the central region. {'With the course interval the upper boundary sits at S ≈ ' + f'{K * np.exp(b):.0f}' + ', too close to the region of interest: the boundary approximation S − Ke^{−r(T−t)} ignores the remaining time value and the error is visible. Switch to the wide grid.' if not grid.startswith('Wide') else 'Push m beyond the stability limit to see the scheme explode, or switch to the naive boundary condition.'}
    """)
st.write("---")

# ── 2. θ-schemes ─────────────────────────────────────────────────────
st.header("2. Implicit and Crank-Nicolson Schemes")
c1, c2 = st.columns(2)
m_cn = c1.select_slider("Space steps m", [50, 100, 200, 400, 800, 1600], value=400)
theta_choice = c2.radio("Scheme", ["Crank-Nicolson (θ = ½)", "Implicit (θ = 1)", "Explicit (θ = 0)"], horizontal=True)
theta = {"C": 0.5, "I": 1.0, "E": 0.0}[theta_choice[0]]
t0_ = time.perf_counter()
t_c, S_c, V_c, lam_c = solve_heat_theta(r, sigma, T, K, a, b, m_cn, nu_max, theta=theta, kind="call")
secs = time.perf_counter() - t0_
bs_c = bs_price(S_c, K, T, r, sigma)
win = (S_c > 0.6 * K) & (S_c < 1.4 * K)
err_c = np.max(np.abs(V_c[0][win] - bs_c[win]))
m = st.columns(3)
m[0].metric("λ", f"{lam_c:.2f}")
m[1].metric("Max error at t = 0", f"{err_c:.2e}" if np.isfinite(err_c) else "overflow")
m[2].metric("Run time", f"{secs * 1e3:.0f} ms")
st.plotly_chart(surface_fig(t_c, S_c, V_c, f"{theta_choice}: v(t, S)", zmax=3 * K), use_container_width=True)


@st.cache_data
def convergence(r, sigma, T, K, a, b):
    ms = np.array([25, 50, 100, 200, 400, 800])
    out = {}
    for name, th in (("Explicit", 0.0), ("Implicit", 1.0), ("Crank-Nicolson", 0.5)):
        errs = []
        for m_ in ms:
            nu = max(int(np.ceil(2 * (sigma**2 * T) * m_**2 / (b - a) ** 2)), m_) if th == 0 else m_
            t_, S_, V_, _ = solve_heat_theta(r, sigma, T, K, a, b, m_, nu, theta=th, kind="call")
            w_ = (S_ > 0.6 * K) & (S_ < 1.4 * K)
            errs.append(np.max(np.abs(V_[0][w_] - bs_price(S_[w_], K, T, r, sigma))))
        out[name] = np.array(errs)
    return ms, out


ms, conv = convergence(r, sigma, T, K, a, b)
fig2 = go.Figure()
for (name, errs), col in zip(conv.items(), ["steelblue", "orange", "crimson"]):
    fig2.add_trace(go.Scatter(x=ms, y=errs, mode="lines+markers", name=name, line=dict(color=col)))
fig2.add_trace(go.Scatter(x=ms, y=conv["Crank-Nicolson"][0] * (ms[0] / ms) ** 2, name="∝ m⁻²", line=dict(color="grey", dash="dot")))
fig2.add_trace(go.Scatter(x=ms, y=conv["Implicit"][0] * (ms[0] / ms), name="∝ m⁻¹", line=dict(color="grey", dash="dash")))
fig2.update_layout(xaxis_type="log", yaxis_type="log", xaxis_title="m", yaxis_title="Max error at t = 0", height=400,
                   title="Convergence (implicit & CN: ν_max = m; explicit: ν_max at the stability limit)")
st.plotly_chart(fig2, use_container_width=True)
st.markdown(f"""
**Interpretation:** The implicit and Crank-Nicolson schemes remain stable for any $\\lambda$ — here
$\\lambda = {lam_c:.2f}$ {'(beyond the explicit limit 1/2: select θ = 0 to watch the same grid explode)' if lam_c > 0.5 else '(raise m to go beyond the explicit limit 1/2)'}. With $\\nu_{{\\max}} = m$, the implicit scheme is only
first-order accurate in time (error $\\propto m^{{-1}}$), whereas Crank-Nicolson, which averages the explicit and
implicit steps, is **second order in both** $\\Delta x$ and $\\Delta\\tilde t$ (error $\\propto m^{{-2}}$). The explicit
scheme is also second order in $\\Delta x$ and even slightly more accurate per $m$, but only because it is forced
to take $\\nu_{{\\max}} \\propto m^2$ time steps to stay stable — at $m = 800$ that is
{max(int(np.ceil(2 * (sigma**2 * T) * 800**2 / (b - a) ** 2)), 800):,} steps versus 800 for Crank-Nicolson.
""")

with st.expander("🧮 The tridiagonal (Thomas) algorithm"):
    st.markdown(r"""
    Each implicit/CN step solves $(1 + 2\theta\lambda)w_i - \theta\lambda(w_{i-1} + w_{i+1}) = \text{rhs}_i$.
    Gaussian elimination specialised to tridiagonal matrices needs only a forward sweep (eliminate the
    sub-diagonal) and a backward substitution — $O(m)$ operations instead of $O(m^3)$:
    """)
    st.code(inspect.getsource(thomas), language="python")
    alpha_t, beta_t, gamma_t = np.array([2.0, 3.0, 4.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0])
    A_t = np.diag(alpha_t) + np.diag(beta_t, 1) + np.diag(gamma_t, -1)
    st.write("Test: solving A x = A·(1, 2, 3) returns", thomas(alpha_t, beta_t, gamma_t, A_t @ np.array([1.0, 2.0, 3.0])))
    st.caption("The app itself uses scipy.linalg.solve_banded, which implements the same algorithm in compiled code.")
st.write("---")

# ── 3. American put ──────────────────────────────────────────────────
st.header("3. American Puts: Projection onto the Obstacle")
st.markdown(r"""
For an American put the value must never fall below the payoff. In transformed variables the payoff is the
**obstacle** $g(\tilde t, x) = e^{\frac{(q+1)^2}{4}\tilde t}\big(e^{\frac{x}{2}(q-1)} - e^{\frac{x}{2}(q+1)}\big)^+$, and the
problem becomes a linear complementarity problem. The simplest approach — used here — solves each step
as for a European option and then **projects**: $w^{(\nu+1)} \leftarrow \max(w^{(\nu+1)}, g^{(\nu+1)})$
(Brennan-Schwartz). The lower boundary condition becomes $v = K - S$.
""")
m_am = st.select_slider("Space steps m (Crank-Nicolson)", [100, 200, 400, 800], value=400)
t_a, S_a, V_a, _ = solve_heat_theta(r, sigma, T, K, a, b, m_am, nu_max, theta=0.5, kind="american put")
_, _, V_eu, _ = solve_heat_theta(r, sigma, T, K, a, b, m_am, nu_max, theta=0.5, kind="put")
payoff = np.maximum(K - S_a, 0)
free_t, free_s = [], []
for k in range(0, len(t_a) - 1, max(1, len(t_a) // 200)):
    held = np.where(V_a[k] > np.maximum(K - S_a, 0) + 1e-6 * K)[0]
    if held.size:
        free_t.append(t_a[k]); free_s.append(S_a[held[0]])
S_test = K
crr_ref = crr_price(S_test, T, r, sigma, 5000, payoff_fn("put", K), american=True)
fd_at = np.interp(S_test, S_a, V_a[0])
m = st.columns(4)
m[0].metric(f"FD American put at S = {S_test:.0f}", f"{fd_at:.4f}")
m[1].metric("CRR (M = 5000)", f"{crr_ref:.4f}")
m[2].metric("European put (BS)", f"{bs_price(S_test, K, T, r, sigma, False):.4f}")
m[3].metric("Exercise boundary S*(0)", f"{free_s[0]:.2f}" if free_s else "—")

col1, col2 = st.columns(2)
with col1:
    fig3 = go.Figure()
    sel = (S_a > 0.3 * K) & (S_a < 1.8 * K)
    fig3.add_trace(go.Scatter(x=S_a[sel], y=V_a[0][sel], name="American put (FD)", line=dict(color="crimson", width=2.5)))
    fig3.add_trace(go.Scatter(x=S_a[sel], y=V_eu[0][sel], name="European put (FD)", line=dict(color="steelblue")))
    fig3.add_trace(go.Scatter(x=S_a[sel], y=payoff[sel], name="Payoff", line=dict(color="grey", dash="dash")))
    fig3.update_layout(xaxis_title="S", yaxis_title="V(0, S)", height=420, legend=dict(x=0.5, y=1))
    st.plotly_chart(fig3, use_container_width=True)
with col2:
    fig4 = go.Figure(go.Scatter(x=free_t, y=free_s, line=dict(color="black", width=2.5), name="S*(t)"))
    fig4.add_hline(y=K, line_dash="dot", line_color="grey", annotation_text="K")
    fig4.update_layout(xaxis_title="t", yaxis_title="Critical stock price S*(t)", height=420,
                       title="Free boundary: exercise below the curve")
    st.plotly_chart(fig4, use_container_width=True)
st.markdown(f"""
**Interpretation:** The American put touches the payoff line below the free boundary
$S^*(0) \\approx {free_s[0] if free_s else float('nan'):.1f}$ and lies strictly above it otherwise; the European put
dips **below** its payoff deep in the money (receiving $K$ only at $T$ is worth less than receiving it now).
The free boundary rises towards $K$ as $t \\to T$. One FD run delivers this boundary and the prices for all $S$
at once, while the binomial tree prices one $S(0)$ at a time — at $S = {S_test:.0f}$ both agree within
{abs(fd_at - crr_ref):.3f}.
""")
