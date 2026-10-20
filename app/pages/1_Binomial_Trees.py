import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from util.binomial import crr_params, crr_tree, crr_price, crr_delta_tree, payoff_fn
from util.black_scholes import bs_price, down_and_out_call

st.title("🌳 Binomial Trees — The Cox-Ross-Rubinstein Model")
st.markdown(r"""
The CRR model is a discrete-time market in which the stock moves up by a factor $u$ or down by a
factor $d$ in each of $M$ periods of length $\Delta t = T/M$. Here it is not a model in its own right
but a **numerical method**: for $M \to \infty$ a properly chosen sequence of CRR models converges
to the Black-Scholes model, so tree prices approximate Black-Scholes prices.

### Choosing $u$, $d$ and $q$
Real-world probabilities are irrelevant for pricing — we need the **martingale probability** $q$
that makes $S/B$ a $Q$-martingale. Matching the first two conditional moments of the
Black-Scholes model and imposing $ud = 1$ gives (Section 1.1):

$$\beta = \tfrac12\big(e^{-r\Delta t} + e^{(r+\sigma^2)\Delta t}\big), \qquad u = \beta + \sqrt{\beta^2 - 1}, \qquad d = 1/u, \qquad q = \frac{e^{r\Delta t} - d}{u - d}.$$

### Backward induction
Starting from the payoff $V_{jM} = g(S_{jM})$ with $S_{ji} = S(0)u^j d^{\,i-j}$ we move back in time:

$$V_{ji} = e^{-r\Delta t}\big(q V_{j+1,i+1} + (1-q) V_{j,i+1}\big) \qquad \text{(European)}$$

$$V_{ji} = \max\Big\{g(S_{ji}),\; e^{-r\Delta t}\big(q V_{j+1,i+1} + (1-q) V_{j,i+1}\big)\Big\} \qquad \text{(American)}$$

The **hedge** follows from matching both successor states:
$\varphi_1(t_i) = \dfrac{V(S u, t_i) - V(S d, t_i)}{S(u-d)}$ — the discrete analogue of the delta.
""")

with st.expander("📖 Intuition: why moment matching and why $ud = 1$?"):
    st.markdown(r"""
    Two conditions (correct conditional mean and variance of $S(t_i)$ given $S(t_{i-1})$) leave one
    degree of freedom among $u, d, q$. The choice $ud = 1$ makes the tree **recombine symmetrically
    around $S(0)$**: the median terminal node equals the initial price. Any other choice $ud = c$ is
    possible too — the strike-anchored tree below uses $c = (K/S(0))^{2/M}$ so that the median node
    sits **exactly at the strike**, which removes most of the oscillating discretisation error.

    The recombining structure is what makes the method cheap: after $M$ steps there are only
    $(M+1)(M+2)/2$ nodes instead of $2^M$ paths. With $k$ underlyings this becomes $(M+1)^k$ terminal
    nodes — the **curse of dimension** (for $M=99$, $k=6$ already $10^{12}$ nodes).
    """)

with st.expander("📖 Why the grid is finer for small prices"):
    st.markdown(r"""
    The tree moves **multiplicatively**: each node is $S \cdot u$ or $S \cdot d$. Equal steps in
    $\log S$ correspond to small absolute steps when $S$ is small and large absolute steps when $S$
    is large. This mirrors the log-normal Black-Scholes dynamics (percentage changes, not absolute
    changes, are i.i.d.). The downside: the resolution near a **fixed strike or barrier** depends on
    where the nodes happen to fall, which causes the saw-tooth convergence seen below.
    """)

st.write("---")

c1, c2, c3, c4, c5 = st.columns(5)
S0 = c1.number_input("S(0)", value=120.0, min_value=1.0, step=5.0)
K = c2.number_input("Strike K", value=100.0, min_value=1.0, step=5.0)
T = c3.number_input("Maturity T", value=1.0, min_value=0.05, step=0.25)
r = c4.number_input("Interest rate r", value=0.02, step=0.01, format="%.3f")
sigma = c5.number_input("Volatility σ", value=0.30, min_value=0.01, step=0.05)

st.write("---")

# ── 1. Tree explorer ──────────────────────────────────────────────────
st.header("1. Explore the Tree")
c1, c2, c3 = st.columns(3)
M_tree = c1.slider("Number of steps M", 2, 50, 8)
kind = c2.radio("Payoff", ["call", "put", "digital"], horizontal=True)
style = c3.radio("Exercise", ["European", "American"], horizontal=True)

g = payoff_fn(kind, K)
S_mat, V_mat, (u, d, q) = crr_tree(S0, T, r, sigma, M_tree, g, american=(style == "American"))
D_mat = crr_delta_tree(S_mat, V_mat)
dt = T / M_tree

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("u", f"{u:.4f}")
m2.metric("d", f"{d:.4f}")
m3.metric("q (martingale prob.)", f"{q:.4f}")
m4.metric("Tree price V₀₀", f"{V_mat[0, 0]:.4f}")
if kind in ("call", "put") and style == "European":
    m5.metric("Black-Scholes price", f"{bs_price(S0, K, T, r, sigma, kind == 'call'):.4f}")
else:
    m5.metric("Hedge ratio φ₁(t₁)", f"{D_mat[0, 0]:.4f}")

# early exercise nodes
early = np.zeros_like(S_mat, dtype=bool)
if style == "American":
    for i in range(M_tree):
        cont = np.exp(-r * dt) * (q * V_mat[1:i + 2, i + 1] + (1 - q) * V_mat[:i + 1, i + 1])
        ex = g(S_mat[:i + 1, i])
        early[:i + 1, i] = (ex > cont + 1e-12) & (ex > 0)

fig = go.Figure()
if M_tree <= 15:
    for i in range(M_tree):
        for j in range(i + 1):
            for jj in (j, j + 1):
                fig.add_trace(go.Scatter(x=[i * dt, (i + 1) * dt], y=[S_mat[j, i], S_mat[jj, i + 1]], mode="lines",
                                         line=dict(color="lightgrey", width=1), hoverinfo="skip", showlegend=False))
xs, ys, vs, hover = [], [], [], []
for i in range(M_tree + 1):
    for j in range(i + 1):
        xs.append(i * dt); ys.append(S_mat[j, i]); vs.append(V_mat[j, i])
        dtxt = f"<br>φ₁ (next period) = {D_mat[j, i]:.4f}" if i < M_tree else ""
        hover.append(f"t = {i * dt:.3f}, j = {j} ups<br>S = {S_mat[j, i]:.2f}<br>V = {V_mat[j, i]:.4f}{dtxt}")
fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Nodes",
                         marker=dict(size=10 if M_tree <= 15 else 5, color=vs, colorscale="Viridis",
                                     colorbar=dict(title="V"), line=dict(width=0.5, color="black")),
                         text=hover, hoverinfo="text"))
if early.any():
    ii, jj = np.where(early.T)
    fig.add_trace(go.Scatter(x=ii * dt, y=S_mat[jj, ii], mode="markers", name="Early exercise optimal",
                             marker=dict(size=14 if M_tree <= 15 else 8, color="rgba(0,0,0,0)",
                                         line=dict(width=2, color="crimson"))))
fig.add_hline(y=K, line_dash="dot", line_color="grey", annotation_text="K")
fig.update_layout(xaxis_title="t", yaxis_title="S(t)", height=500, legend=dict(x=0, y=1))
st.plotly_chart(fig, use_container_width=True)
st.caption("Hover a node to see its stock price, option value and the hedge ratio for the next period. "
           "Colour = option value.")

with st.expander("🎲 Follow one random path through the tree"):
    seed = st.number_input("Random seed", value=24, step=1)
    rng = np.random.default_rng(int(seed))
    flips = rng.binomial(1, 0.5, size=M_tree)
    path = np.append(0, flips.cumsum())
    t_ax = np.arange(M_tree + 1) * dt
    figp = make_subplots(rows=1, cols=2, subplot_titles=("Stock price S(t)", "Option price V(t)"))
    for col, mat in ((1, S_mat), (2, V_mat)):
        figp.add_trace(go.Scatter(x=np.repeat(t_ax, np.arange(1, M_tree + 2)),
                                  y=np.concatenate([mat[:i + 1, i] for i in range(M_tree + 1)]),
                                  mode="markers", marker=dict(symbol="cross", size=5, color="black"),
                                  showlegend=False), row=1, col=col)
        figp.add_trace(go.Scatter(x=t_ax, y=mat[path, np.arange(M_tree + 1)], mode="lines+markers",
                                  line=dict(color="crimson"), showlegend=False), row=1, col=col)
    figp.update_layout(height=380)
    st.plotly_chart(figp, use_container_width=True)
    st.markdown("Coin flips with real-world probability $p = 0.5$ select one path; the option price along "
                "it moves with the stock (a call rises with $S$) and converges to the payoff at $T$.")

st.markdown(f"""
**Reading the tree:** With $M = {M_tree}$ steps, the up-factor $u = {u:.4f}$ corresponds to a
{100 * (u - 1):.2f}% move per period, and $q = {q:.4f}$ is slightly {'above' if q > 0.5 else 'below'} $1/2$ —
the martingale probability compensates for the drift $r$ of the bond so that the discounted stock has
no drift. Note that $q$ is **not** a real-world probability: it is the unique weight that prices
every claim consistently with the bond and the stock.
""")
if style == "American" and kind == "put":
    st.markdown(f"""
    **Early exercise:** The circled nodes are where exercising beats holding. They form a connected
    region **below a critical stock price** that rises towards $K$ as $t \\to T$ — the discrete
    version of the free boundary. The American premium here is
    {V_mat[0, 0] - crr_tree(S0, T, r, sigma, M_tree, g)[1][0, 0]:.4f}.
    """)
elif style == "American" and kind == "call":
    st.markdown("""
    **No early exercise for calls:** For $r \\geq 0$ the continuation value of a call always exceeds
    its intrinsic value, so the American and European call prices coincide — no node is circled.
    """)
st.write("---")

# ── 2. Convergence ───────────────────────────────────────────────────
st.header("2. Convergence to Black-Scholes")
st.markdown(r"""
The tree price converges to the Black-Scholes price as $M \to \infty$, but **not monotonically**:
the error oscillates depending on where the strike falls relative to the terminal nodes. The
**strike-anchored** tree replaces $ud = 1$ by $ud = c$ with $c = (K/S(0))^{2/M}$, i.e.

$$u = \beta + \sqrt{\beta^2 - c}, \qquad \beta = \tfrac12\big(c\,e^{-r\Delta t} + e^{(r+\sigma^2)\Delta t}\big),$$

which forces the middle terminal node $S(0)u^{M/2}d^{M/2}$ to equal $K$.
""")


@st.cache_data
def conv_in_M(S0, K, T, r, sigma, M_max):
    g = payoff_fn("call", K)
    Ms = np.arange(10, M_max + 1, 2 if M_max <= 300 else 5)
    crr = np.array([crr_price(S0, T, r, sigma, M, g) for M in Ms])
    anch = np.array([crr_price(S0, T, r, sigma, M, g, c=(K / S0) ** (2 / M)) for M in Ms])
    odd = np.array([crr_price(S0, T, r, sigma, M + 1, g) for M in Ms])
    return Ms, crr, anch, odd


@st.cache_data
def conv_in_S(K, T, r, sigma, M):
    g = payoff_fn("call", K)
    S_grid = np.linspace(20, 300, 400)
    crr = np.array([crr_price(s, T, r, sigma, M, g) for s in S_grid])
    anch = np.array([crr_price(s, T, r, sigma, M, g, c=(K / s) ** (2 / M)) for s in S_grid])
    return S_grid, crr, anch


view = st.radio("Show error as a function of", ["number of steps M", "initial price S(0)"], horizontal=True)
bs_ref = bs_price(S0, K, T, r, sigma)
if view == "number of steps M":
    M_max = st.slider("Largest M", 50, 600, 300, step=50)
    Ms, crr_v, anch_v, odd_v = conv_in_M(S0, K, T, r, sigma, M_max)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=Ms, y=crr_v - bs_ref, mode="lines", name="CRR (even M)", line=dict(color="steelblue")))
    fig2.add_trace(go.Scatter(x=Ms + 1, y=odd_v - bs_ref, mode="lines", name="CRR (odd M)", line=dict(color="lightskyblue")))
    fig2.add_trace(go.Scatter(x=Ms, y=anch_v - bs_ref, mode="lines", name="Strike-anchored", line=dict(color="crimson", width=2)))
    fig2.add_hline(y=0, line_color="black", line_width=1)
    fig2.update_layout(xaxis_title="M", yaxis_title="V_tree(0) − V_BS(0)", legend=dict(x=0.7, y=1))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(f"""
    **Interpretation:** At $M = {Ms[-1]}$ the CRR error is {crr_v[-1] - bs_ref:+.5f} while the anchored
    tree is off by only {anch_v[-1] - bs_ref:+.6f}. The plain CRR error decays roughly like $1/M$ but
    with a **saw-tooth pattern** — every time a terminal node crosses the strike, the payoff kink is
    resolved differently. Anchoring a node at $K$ removes this source of error, so the anchored curve
    is smooth and much closer to zero. The price of this: $u, d$ now depend on $K$, so one tree can no
    longer price several strikes at once.
    """)
else:
    M_s = st.slider("Steps M", 20, 300, 100, step=10)
    S_grid, crr_s, anch_s = conv_in_S(K, T, r, sigma, M_s)
    bs_s = bs_price(S_grid, K, T, r, sigma)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=S_grid, y=crr_s - bs_s, mode="lines", name="CRR", line=dict(color="steelblue")))
    fig2.add_trace(go.Scatter(x=S_grid, y=anch_s - bs_s, mode="lines", name="Strike-anchored", line=dict(color="crimson")))
    fig2.add_vline(x=K, line_dash="dot", line_color="grey", annotation_text="K")
    fig2.update_layout(xaxis_title="S(0)", yaxis_title="V_tree(0) − V_BS(0)")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(f"""
    **Interpretation:** For fixed $M = {M_s}$, the plain CRR error oscillates wildly as $S(0)$ moves,
    because the position of the strike between two terminal nodes changes with $S(0)$. The maximum
    absolute error is {np.max(np.abs(crr_s - bs_s)):.4f} for CRR vs {np.max(np.abs(anch_s - bs_s)):.4f}
    for the anchored tree. The error is largest near the money, where the payoff kink matters most.
    """)
st.write("---")

# ── 3. Bounds & parity ───────────────────────────────────────────────
st.header("3. No-Arbitrage Bounds and Put-Call Parity")
st.markdown(r"""
In **any** arbitrage-free model, European option prices satisfy

$$S(0) - Ke^{-rT} \le V_{\text{call}}(0) \le S(0), \qquad 0 \le V_{\text{put}}(0) \le Ke^{-rT},$$

and the **put-call parity** $V_{\text{put}}(t) = V_{\text{call}}(t) - S(t) + Ke^{-r(T-t)}$. Below the CRR
prices ($M=100$) are computed for volatilities from almost zero to 500%.
""")


@st.cache_data
def prices_vs_sigma(S0, K, T, r):
    sigmas = np.linspace(0.01, 5, 120)
    call = np.array([crr_price(S0, T, r, s, 100, payoff_fn("call", K)) for s in sigmas])
    put = np.array([crr_price(S0, T, r, s, 100, payoff_fn("put", K)) for s in sigmas])
    return sigmas, call, put


sigmas, call_s, put_s = prices_vs_sigma(S0, K, T, r)
fig3 = make_subplots(rows=1, cols=3, subplot_titles=("Call vs σ", "Put vs σ", "Parity residual"))
fig3.add_trace(go.Scatter(x=sigmas, y=call_s, name="Call", line=dict(color="steelblue")), row=1, col=1)
for yb in (S0, S0 - K * np.exp(-r * T)):
    fig3.add_hline(y=yb, line_dash="dash", line_color="crimson", row=1, col=1)
fig3.add_trace(go.Scatter(x=sigmas, y=put_s, name="Put", line=dict(color="seagreen")), row=1, col=2)
for yb in (0, K * np.exp(-r * T)):
    fig3.add_hline(y=yb, line_dash="dash", line_color="crimson", row=1, col=2)
fig3.add_trace(go.Scatter(x=sigmas, y=put_s - (call_s - S0 + K * np.exp(-r * T)), name="Residual",
                          line=dict(color="orange")), row=1, col=3)
fig3.update_xaxes(title_text="σ")
fig3.update_layout(height=380, showlegend=False)
st.plotly_chart(fig3, use_container_width=True)
st.markdown(f"""
**Interpretation:** For $\\sigma \\to 0$ the stock becomes riskless and the call tends to its lower bound
$\\max(S(0) - Ke^{{-rT}}, 0) = {max(S0 - K * np.exp(-r * T), 0):.2f}$; for $\\sigma \\to \\infty$ it approaches
the upper bound $S(0) = {S0:.0f}$ (the option becomes as valuable as the stock itself, since almost
every path ends near zero but a few explode). The put behaves the same way between $0$ and
$Ke^{{-rT}}$. The parity residual is at machine precision (max {np.max(np.abs(put_s - (call_s - S0 + K * np.exp(-r * T)))):.1e})
for **every** $\\sigma$: parity is a model-free identity that the tree satisfies exactly.
""")
st.write("---")

# ── 4. Barrier options ───────────────────────────────────────────────
st.header("4. Down-and-Out Barrier Call")
st.markdown(r"""
A down-and-out call pays $(S(T) - K)^+$ only if the price **never** fell below the barrier $B$.
The payoff is path-dependent, yet the recombining tree still works: conditional on not having been
knocked out, the value depends only on the current node. We simply set $V_{ji} = 0$ whenever
$S_{ji} < B$ during backward induction (Section 1.3).
""")


@st.cache_data
def barrier_curve(S0, K, T, r, sigma, M):
    Bs = np.linspace(0.5 * S0, 1.2 * S0, 120)
    g = payoff_fn("call", K)
    prices = np.array([crr_price(S0, T, r, sigma, M, g, barrier=B) for B in Bs])
    return Bs, prices


@st.cache_data
def barrier_conv(S0, K, T, r, sigma, B):
    Ms = np.arange(20, 801, 4)
    g = payoff_fn("call", K)
    return Ms, np.array([crr_price(S0, T, r, sigma, M, g, barrier=B) for M in Ms])


c1, c2 = st.columns(2)
M_bar = c1.slider("Steps M (barrier)", 20, 500, 200, step=20)
B_hi = max(float(min(K, S0)) - 1.0, 0.5 * S0 + 1.0)
B_fix = c2.slider("Barrier B for the convergence plot (B < K)", float(0.5 * S0), B_hi, float(min(0.8 * S0, B_hi)))
Bs, bar_prices = barrier_curve(S0, K, T, r, sigma, M_bar)
Ms_b, conv_b = barrier_conv(S0, K, T, r, sigma, B_fix)
closed = down_and_out_call(S0, K, B_fix, T, r, sigma)

col1, col2 = st.columns(2)
with col1:
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=Bs, y=bar_prices, name="CRR barrier price", line=dict(color="steelblue")))
    valid = Bs < min(K, S0)
    fig4.add_trace(go.Scatter(x=Bs[valid], y=down_and_out_call(S0, K, Bs[valid], T, r, sigma),
                              name="Closed form (continuous)", line=dict(color="crimson", dash="dash")))
    fig4.add_hline(y=bs_ref, line_dash="dot", line_color="grey", annotation_text="Vanilla call")
    fig4.update_layout(xaxis_title="Barrier B", yaxis_title="V(0)", title="Price vs barrier", legend=dict(x=0, y=0.1))
    st.plotly_chart(fig4, use_container_width=True)
with col2:
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=Ms_b, y=conv_b, name="CRR", line=dict(color="steelblue")))
    fig5.add_hline(y=closed, line_dash="dash", line_color="crimson", annotation_text="Continuous barrier")
    fig5.update_layout(xaxis_title="M", yaxis_title="V(0)", title=f"Convergence for B = {B_fix:.1f}")
    st.plotly_chart(fig5, use_container_width=True)

st.markdown(f"""
**Interpretation:** For $B \\to 0$ the barrier is irrelevant and the price tends to the vanilla call
({bs_ref:.2f}); for $B \\geq S(0)$ the option is knocked out immediately and is worthless. The **step
pattern** in the left plot and the slow, saw-tooth convergence on the right have the same cause: the
tree can only "see" the barrier at its node levels, so the effective barrier jumps between node levels
as $B$ or $M$ varies. Discrete monitoring also checks the barrier only at $t_i$, so the tree price lies
**above** the continuously monitored price ({closed:.4f}); at $M = {Ms_b[-1]}$ the gap is still
{conv_b[-1] - closed:+.4f}. Letting $M \\to \\infty$ smooths the left curve into the closed form.
""")
st.write("---")

# ── 5. American options & Richardson ─────────────────────────────────
st.header("5. American Puts and Richardson Extrapolation")
st.markdown(r"""
For American options, the error of the binomial price alternates in sign between **even and odd $M$**.
This regular behaviour can be exploited by **Richardson extrapolation**: if
$V_M \approx V + c/M$, then $2V_{2M} - V_M$ cancels the leading error term. With the strike-anchored
tree the lecture suggests the combination $2V_{4M}(0) - V_{2M}(0)$.
""")

S0_am = st.slider("S(0) for the American put", 50.0, 150.0, 80.0, step=5.0)


@st.cache_data
def american_study(S0, K, T, r, sigma):
    g = payoff_fn("put", K)
    anch = lambda M: crr_price(S0, T, r, sigma, M, g, american=True, c=(K / S0) ** (2 / M))
    ref = anch(20000)
    Ms_test = np.arange(50, 501, 5)
    test = np.array([anch(M) for M in Ms_test])
    Ms = np.arange(10, 501, 10)
    v2 = np.array([anch(2 * M) for M in Ms])
    v4 = np.array([anch(4 * M) for M in Ms])
    eu = bs_price(S0, K, T, r, sigma, call=False)
    return ref, Ms_test, test, Ms, v2, v4, eu


with st.spinner("Computing reference price with M = 20 000 (cached afterwards)..."):
    ref, Ms_test, test, Ms_r, v2, v4, eu_put = american_study(S0_am, K, T, r, sigma)

m1, m2, m3 = st.columns(3)
m1.metric("American put (M = 20 000)", f"{ref:.5f}")
m2.metric("European put (BS)", f"{eu_put:.5f}")
m3.metric("Early exercise premium", f"{ref - eu_put:.5f}")

col1, col2 = st.columns(2)
with col1:
    fig6 = go.Figure()
    even = Ms_test % 2 == 0
    fig6.add_trace(go.Scatter(x=Ms_test[even], y=test[even] - ref, mode="markers", name="even M", marker=dict(color="steelblue")))
    fig6.add_trace(go.Scatter(x=Ms_test[~even], y=test[~even] - ref, mode="markers", name="odd M", marker=dict(color="orange")))
    fig6.add_hline(y=0, line_color="black", line_width=1)
    fig6.update_layout(xaxis_title="M", yaxis_title="V_M(0) − V_ref(0)", title="Error of the anchored tree")
    st.plotly_chart(fig6, use_container_width=True)
with col2:
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=4 * Ms_r, y=v4, name="Crude V_4M", line=dict(color="steelblue")))
    fig7.add_trace(go.Scatter(x=4 * Ms_r, y=2 * v4 - v2, name="Richardson 2V_4M − V_2M", line=dict(color="crimson")))
    fig7.add_hline(y=ref, line_dash="dash", line_color="black", annotation_text="Reference")
    fig7.update_layout(xaxis_type="log", xaxis_title="Largest number of steps used (4M)", yaxis_title="V(0)",
                       title="Crude vs extrapolated")
    st.plotly_chart(fig7, use_container_width=True)

err_crude = np.abs(v4 - ref)
err_rich = np.abs(2 * v4 - v2 - ref)
st.markdown(f"""
**Interpretation:** The left plot shows the over-/undershooting pattern for even vs odd $M$.
On the right, the extrapolated price with $4M = {4 * Ms_r[-1]}$ has an error of {err_rich[-1]:.2e}
compared with {err_crude[-1]:.2e} for the crude price. To reach the accuracy that Richardson
achieves at $4M = 200$ ({err_rich[4]:.1e}), the crude tree needs far more steps — and since the cost of a
tree grows like $M^2$, extrapolation is almost free accuracy. The early exercise premium of
{ref - eu_put:.4f} is the value of the right to exercise before maturity; it is largest deep in the
money where receiving $K$ early earns interest.
""")
