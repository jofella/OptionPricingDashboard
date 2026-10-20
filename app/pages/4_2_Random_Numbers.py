import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.stats import rayleigh, expon, norm


st.title("🔢 Random Number Generation")
st.markdown(r"""
Every Monte Carlo method starts from **uniform random numbers** on $[0,1]$ and transforms them into
draws from the distribution we actually need (Section 4.2). This page walks through the toolbox:

1. **Pseudo-random uniforms** — linear congruential generators.
2. **Inversion** — $X = F^{-1}(U)$.
3. **Acceptance/rejection** — sample from a simpler proposal and accept with probability $f/(Cg)$.
4. **Transformations of random vectors** — Box-Muller and correlated normals via Cholesky.
""")
st.write("---")

# ── 1. LCG ───────────────────────────────────────────────────────────
st.header("1. Linear Congruential Generators")
st.markdown(r"""
Computers are deterministic, so they produce **pseudo**-random numbers that merely pass statistical
tests. The classic recipe is the linear congruential generator

$$s_n = (A s_{n-1} + C) \bmod M, \qquad u_n = s_n / M.$$

There are at most $M$ different values, so the sequence is periodic; good choices of $A, C, M$ give
the full period $M$ (Hull-Dobell: $C$ odd and $A - 1$ divisible by 4 for $M = 2^k$). A full period is
not enough, though: consecutive pairs $(u_n, u_{n+1})$ always lie on a **lattice of parallel lines**, and
a badly chosen multiplier puts them on very few lines.
""")
gen = st.radio("Generator", ["Bad toy: M = 2048, A = 9, C = 1", "Better toy: M = 2048, A = 1229, C = 1",
                             "Park-Miller: M = 2³¹−1, A = 7⁵, C = 0", "NumPy PCG64 (modern)"], horizontal=True)


def lcg(A, C, M, n, seed=1):
    s = np.empty(n, dtype=np.int64)
    s_prev = seed
    for i in range(n):
        s_prev = (A * s_prev + C) % M
        s[i] = s_prev
    return s / M


n_u = 2000
if gen.startswith("Bad"):
    u = lcg(9, 1, 2048, n_u)
elif gen.startswith("Better"):
    u = lcg(1229, 1, 2048, n_u)
elif gen.startswith("Park"):
    u = lcg(7**5, 0, 2**31 - 1, n_u)
else:
    u = np.random.default_rng(1).random(n_u)
fig = make_subplots(rows=1, cols=2, subplot_titles=("Pairs (uₙ, uₙ₊₁)", "Histogram"))
fig.add_trace(go.Scatter(x=u[:-1], y=u[1:], mode="markers", marker=dict(size=3, color="steelblue")), row=1, col=1)
fig.add_trace(go.Histogram(x=u, xbins=dict(start=0, end=1, size=0.05), marker_color="steelblue"), row=1, col=2)
fig.update_layout(height=380, showlegend=False)
st.plotly_chart(fig, use_container_width=True)
st.markdown(r"""
**Interpretation:** All four generators produce a flat histogram — one-dimensional uniformity is easy,
and both toy generators even have the full period 2048. The pair plot is the real test: with $A = 9$
every pair satisfies $u_{n+1} \approx 9u_n \bmod 1$, so all points lie on just **9 lines** and
$(u_n, u_{n+1})$ is far from independent. With $A = 1229$ the points still form a lattice (only 2048
distinct values exist), but a much finer and more even one. Park-Miller's huge modulus hides the lattice
at the plotted resolution, and modern generators such as PCG64 combine several mechanisms to get
astronomically long periods.
""")
st.write("---")

# ── 2. Inversion ─────────────────────────────────────────────────────
st.header("2. The Inversion Method")
st.markdown(r"""
If $U \sim \mathcal U[0,1]$ and $F^{-1}(u) = \inf\{x : F(x) \ge u\}$ is the (pseudo-)inverse of a cdf,
then $X = F^{-1}(U)$ has cdf $F$ because $P(F^{-1}(U) \le x) = P(U \le F(x)) = F(x)$.

**Example — Rayleigh distribution** $\mathrm{Ray}(\sigma)$: $f(x) = \frac{x}{\sigma^2}e^{-x^2/(2\sigma^2)}$,
$F(x) = 1 - e^{-x^2/(2\sigma^2)}$, so $F^{-1}(u) = \sqrt{-2\sigma^2\log(1-u)}$.

If $F^{-1}$ is not available in closed form but $F$ can be evaluated on a grid $x_1 < \dots < x_n$,
we can **interpolate linearly** between the grid points: find $F(x_{i-1}) \le u \le F(x_i)$ and set
$F^{-1}(u) \approx x_{i-1} + \frac{u - F(x_{i-1})}{F(x_i) - F(x_{i-1})}(x_i - x_{i-1})$.
""")
c1, c2, c3 = st.columns(3)
sig_r = c1.slider("Rayleigh scale σ", 0.5, 6.0, 3.0, step=0.5)
N_inv = c2.select_slider("Samples", [100, 1000, 10000, 100000], value=1000)
n_grid = c3.select_slider("Grid points for interpolated inverse", [5, 10, 20, 50, 1000], value=20)
rng = np.random.default_rng(20)
U = rng.random(N_inv)
x_exact = np.sqrt(-2 * sig_r**2 * np.log(1 - U))
x_grid = np.linspace(0, 4 * sig_r, n_grid)
F_grid = 1 - np.exp(-x_grid**2 / (2 * sig_r**2))
x_interp = np.interp(U, F_grid, x_grid)
xx = np.linspace(0, 4 * sig_r, 400)

fig2 = make_subplots(rows=1, cols=3, subplot_titles=("cdf and its inverse", "Exact inversion", "Interpolated inversion"))
fig2.add_trace(go.Scatter(x=xx, y=rayleigh(scale=sig_r).cdf(xx), name="F(x)", line=dict(color="black")), row=1, col=1)
fig2.add_trace(go.Scatter(x=x_grid, y=F_grid, mode="markers+lines", name="grid", line=dict(color="orange", dash="dot")), row=1, col=1)
for k in range(6):
    fig2.add_trace(go.Scatter(x=[0, x_exact[k], x_exact[k]], y=[U[k], U[k], 0], mode="lines",
                              line=dict(color="crimson", width=1), showlegend=False), row=1, col=1)
for col, smp in ((2, x_exact), (3, x_interp)):
    fig2.add_trace(go.Histogram(x=smp, nbinsx=40, histnorm="probability density", marker_color="steelblue",
                                opacity=0.7, showlegend=False), row=1, col=col)
    fig2.add_trace(go.Scatter(x=xx, y=rayleigh(scale=sig_r).pdf(xx), line=dict(color="crimson"), showlegend=False), row=1, col=col)
fig2.update_layout(height=380, showlegend=False)
st.plotly_chart(fig2, use_container_width=True)
st.markdown(f"""
**Interpretation:** The red lines in the left panel show the mechanism: a uniform height $u$ is mapped
horizontally onto the cdf and down to $x = F^{{-1}}(u)$. Steep parts of $F$ (where the density is large)
catch many $u$'s and therefore produce many samples. With only {n_grid} grid points the interpolated
inverse replaces $F$ by a polygon, which turns the density into a **step function** — visible in the
right histogram for coarse grids. With enough grid points both methods are indistinguishable.
""")
st.write("---")

# ── 3. Acceptance / rejection ────────────────────────────────────────
st.header("3. Acceptance/Rejection")
st.markdown(r"""
Suppose we can sample $Y$ from a density $g$ with $f \le Cg$. Draw $Y \sim g$ and $U \sim \mathcal U[0,1]$
and **accept** $Y$ as a draw from $f$ if $U \le \frac{f(Y)}{Cg(Y)}$; otherwise try again.
The acceptance probability is $1/C$, so $C = \sup_x f(x)/g(x)$ should be as **small as possible**.

**Example:** target $f = \mathrm{Ray}(3)$, proposal $g_s = \mathrm{Exp}$ with scale $s$. The tutorial
starts with $s = 1$ and then optimises $s$.
""")
c1, c2 = st.columns(2)
s = c1.slider("Proposal scale s", 1.0, 10.0, 1.0, step=0.1)
N_ar = c2.select_slider("Target number of samples", [500, 1000, 5000], value=1000)
f = rayleigh(scale=3)
x_ar = np.linspace(1e-6, 40, 40000)


def C_of(s):
    return np.max(f.pdf(x_ar) / expon(scale=s).pdf(x_ar))


C = C_of(s)
s_grid = np.linspace(1.0, 10, 120)
C_grid = np.array([C_of(v) for v in s_grid])
s_opt = s_grid[np.argmin(C_grid)]

rng = np.random.default_rng(20)
accepted, trials = [], 0
while len(accepted) < N_ar:
    Y = rng.exponential(s, size=4 * N_ar)
    Uu = rng.random(4 * N_ar)
    ok = Uu <= f.pdf(Y) / (C * expon(scale=s).pdf(Y))
    need = N_ar - len(accepted)
    acc_idx = np.where(ok)[0][:need]
    trials += (acc_idx[-1] + 1) if len(acc_idx) == need else 4 * N_ar
    accepted.extend(Y[acc_idx])
samples = np.array(accepted)

m = st.columns(4)
m[0].metric("C = sup f/g", f"{C:.2f}")
m[1].metric("Acceptance rate 1/C", f"{1 / C:.1%}")
m[2].metric("Proposals needed", f"{trials:,}")
m[3].metric("Optimal scale s*", f"{s_opt:.2f}", f"C* = {C_grid.min():.2f}", delta_color="off")

xx = np.linspace(0, 15, 400)
fig3 = make_subplots(rows=1, cols=3, subplot_titles=("Target f and envelope C·g", "Accepted samples", "C as a function of s"))
fig3.add_trace(go.Scatter(x=xx, y=f.pdf(xx), name="f", line=dict(color="crimson")), row=1, col=1)
fig3.add_trace(go.Scatter(x=xx, y=C * expon(scale=s).pdf(xx), name="C g", line=dict(color="black", dash="dash")), row=1, col=1)
fig3.add_trace(go.Histogram(x=samples, nbinsx=40, histnorm="probability density", marker_color="steelblue", opacity=0.7), row=1, col=2)
fig3.add_trace(go.Scatter(x=xx, y=f.pdf(xx), line=dict(color="crimson")), row=1, col=2)
fig3.add_trace(go.Scatter(x=s_grid, y=C_grid, line=dict(color="steelblue")), row=1, col=3)
fig3.add_vline(x=s, line_dash="dot", line_color="crimson", row=1, col=3)
fig3.update_yaxes(range=[0, 2.5 * f.pdf(xx).max()], row=1, col=1)
fig3.update_yaxes(type="log", row=1, col=3)
fig3.update_xaxes(title_text="s", row=1, col=3)
fig3.update_layout(height=380, showlegend=False)
st.plotly_chart(fig3, use_container_width=True)
st.markdown(f"""
**Interpretation:** The envelope $C g_s$ must lie above $f$ everywhere. With $s = {s}$ it has to be
stretched by $C = {C:.1f}$, so only {1 / C:.1%} of the proposals are accepted and {trials:,} proposals were
needed for {N_ar:,} samples. The exponential with scale 1 puts too little mass where the Rayleigh(3)
density lives, which forces a huge $C$. The optimal scale $s^* \\approx {s_opt:.2f}$ gives $C^* \\approx {C_grid.min():.2f}$,
roughly {C / C_grid.min():.0f}× fewer proposals than the current choice. The samples are exact either way —
only efficiency changes.
""")
st.write("---")

# ── 4. Transformations ───────────────────────────────────────────────
st.header("4. Transformations: Box-Muller and Correlated Normals")
st.markdown(r"""
**Box-Muller.** If $U_1, U_2$ are independent uniforms, then

$$Z_1 = \sqrt{-2\log U_1}\cos(2\pi U_2), \qquad Z_2 = \sqrt{-2\log U_1}\sin(2\pi U_2)$$

are independent standard normals (change-of-variables theorem: the radius $\sqrt{-2\log U_1}$ has a
Rayleigh(1) law — inversion again! — and the angle is uniform).

**Correlated normals.** To sample $X \sim \mathcal N(\mu, \Sigma)$, take $A$ with $AA^\top = \Sigma$
(e.g. the **Cholesky factor**) and set $X = \mu + AZ$ with $Z$ standard normal. Then
$\operatorname{Cov}(X) = A\operatorname{Cov}(Z)A^\top = \Sigma$.
""")
rng = np.random.default_rng(3)
U1, U2 = rng.random(3000), rng.random(3000)
R_ = np.sqrt(-2 * np.log(U1))
Z1, Z2 = R_ * np.cos(2 * np.pi * U2), R_ * np.sin(2 * np.pi * U2)
col1, col2 = st.columns(2)
with col1:
    fig4 = make_subplots(rows=1, cols=2, subplot_titles=("Uniform input (U₁, U₂)", "Box-Muller output (Z₁, Z₂)"))
    fig4.add_trace(go.Scatter(x=U1, y=U2, mode="markers", marker=dict(size=3, color=R_, colorscale="Viridis")), row=1, col=1)
    fig4.add_trace(go.Scatter(x=Z1, y=Z2, mode="markers", marker=dict(size=3, color=R_, colorscale="Viridis")), row=1, col=2)
    fig4.update_layout(height=360, showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("Colour = radius √(−2 log U₁): small U₁ (left edge) is sent far out into the tails.")

with col2:
    rho_c = st.slider("Target correlation ρ", -0.99, 0.99, 0.7, step=0.01)
    Sig = np.array([[1.0, rho_c], [rho_c, 1.0]])
    A = np.linalg.cholesky(Sig)
    Zs = np.column_stack([Z1, Z2])
    X_c = Zs @ A.T
    fig5 = make_subplots(rows=1, cols=2, subplot_titles=("Independent Z", "X = AZ (Cholesky)"))
    fig5.add_trace(go.Scatter(x=Zs[:, 0], y=Zs[:, 1], mode="markers", marker=dict(size=3, color="grey")), row=1, col=1)
    fig5.add_trace(go.Scatter(x=X_c[:, 0], y=X_c[:, 1], mode="markers", marker=dict(size=3, color="crimson")), row=1, col=2)
    fig5.update_xaxes(range=[-4, 4]); fig5.update_yaxes(range=[-4, 4])
    fig5.update_layout(height=360, showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)
    st.caption(f"A = [[1, 0], [ρ, √(1−ρ²)]] = [[1, 0], [{A[1, 0]:.2f}, {A[1, 1]:.2f}]].  "
               f"Sample correlation of X: {np.corrcoef(X_c.T)[0, 1]:.3f}")

st.markdown(r"""
**Interpretation:** Box-Muller maps the unit square onto the plane: $U_2$ sets the angle, $U_1$ the
radius, and the points near the left edge of the square become the rare tail draws. The Cholesky
factor keeps the first coordinate and mixes the second as $X_2 = \rho Z_1 + \sqrt{1-\rho^2}\,Z_2$, which turns
the round cloud into an ellipse with exactly the requested correlation. This is how correlated Brownian
motions for baskets, or for the stock and variance in the Heston model, are generated.
""")
