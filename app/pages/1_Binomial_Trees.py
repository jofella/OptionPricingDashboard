import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from util.binomial import crr_params, crr_tree, crr_price, crr_delta_tree, payoff_fn
from util.black_scholes import bs_price, down_and_out_call
from util.ui import takeaways, test_yourself

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

with st.expander("🎓 Deep dive: one-period replication — why real-world probabilities do not matter"):
    st.markdown(r"""
    Take a single period: the stock moves from $S$ to $Su$ or $Sd$, and an option pays $V_u$ or $V_d$.
    Build a portfolio of $\varphi_1$ shares and $\varphi_0$ bonds that pays **exactly** the same in both states:

    $$\varphi_0 B e^{r\Delta t} + \varphi_1 S u = V_u, \qquad \varphi_0 B e^{r\Delta t} + \varphi_1 S d = V_d.$$

    Two equations, two unknowns. Subtracting them gives the hedge ratio and then the bond position:

    $$\varphi_1 = \frac{V_u - V_d}{S(u - d)}, \qquad \varphi_0 B = e^{-r\Delta t}\,\frac{u V_d - d V_u}{u - d}.$$

    The portfolio costs $\varphi_0 B + \varphi_1 S$ today. If the option traded at any other price, buying the cheaper
    of the two and selling the other would be a riskless profit (arbitrage). Rearranging the cost gives

    $$V = e^{-r\Delta t}\big(q V_u + (1 - q) V_d\big), \qquad q = \frac{e^{r\Delta t} - d}{u - d}.$$

    **Numerical example.** $S = 100$, $u = 1.1$, $d = 0.9$, $r = 0$, call with $K = 100$: $V_u = 10$, $V_d = 0$.
    Then $\varphi_1 = 10/20 = 0.5$ shares and $\varphi_0 B = -45$ (borrow 45). Cost: $0.5 \cdot 100 - 45 = 5$.
    With $q = (1 - 0.9)/0.2 = 0.5$ the formula gives $0.5 \cdot 10 = 5$ as well.

    Nowhere did the real-world probability $p$ of an up-move enter. Whether the stock goes up with 90% or 10%
    probability, the replicating portfolio costs the same — so the option price must be the same. The number $q$
    is merely the weight that makes this pricing rule look like an expectation; it is the **martingale (risk-neutral)
    probability**. The condition $d < e^{r\Delta t} < u$ guarantees $0 < q < 1$, i.e. absence of arbitrage.
    """)

with st.expander("🎓 Deep dive: why the CRR model converges to Black-Scholes"):
    st.markdown(r"""
    After $M$ steps, $\log S(T) = \log S(0) + \sum_{i=1}^M \xi_i$ with i.i.d. steps $\xi_i \in \{\log u, \log d\}$.
    By construction (moment matching), each step has under $Q$ approximately

    $$E_Q[\xi_i] \approx \big(r - \tfrac{\sigma^2}{2}\big)\Delta t, \qquad \operatorname{Var}_Q(\xi_i) \approx \sigma^2 \Delta t,$$

    since $\log u = -\log d \approx \sigma\sqrt{\Delta t}$. Summing $M = T/\Delta t$ such steps gives mean
    $(r - \sigma^2/2)T$ and variance $\sigma^2 T$, and by the **central limit theorem** the sum becomes normal:

    $$\log S(T) \;\xrightarrow{M\to\infty}\; \mathcal N\Big(\log S(0) + \big(r - \tfrac{\sigma^2}{2}\big)T,\; \sigma^2 T\Big),$$

    which is exactly the law of $\log S(T)$ under $Q$ in the Black-Scholes model. Since prices are expectations of
    (continuous, not too fast growing) payoffs, tree prices converge to Black-Scholes prices. For calls and puts the
    error is of order $1/M$, but the constant in front of $1/M$ oscillates — see Section 2.

    This is why the lecture views the tree as a **numerical method** rather than as a model: the output is a
    "correct answer in a slightly different market", which converges to the answer we want.
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
with st.expander("🎓 Deep dive: the hedge ratio is a discrete delta"):
    st.markdown(r"""
    The hedge ratio $\varphi_1(t_i) = \dfrac{V(Su, t_i) - V(Sd, t_i)}{Su - Sd}$ is a **difference quotient** of the
    option price with respect to the stock price. As $\Delta t \to 0$ the two nodes $Su$ and $Sd$ move together
    ($Su - Sd \approx 2S\sigma\sqrt{\Delta t}$), and the quotient converges to the derivative
    $\partial_x v(t, S)$ — the Black-Scholes **delta**. In the same way, the change of $\varphi_1$ between
    neighbouring nodes approximates **gamma**.

    The bond position $\varphi_0(t_i) = \big(V(S, t_{i-1}) - \varphi_1(t_i) S\big)/B(t_{i-1})$ finances the
    shares. For a call it is negative (the hedger borrows money to buy shares); for a put the hedger is short
    shares and holds a positive bond position. Hover over the nodes above: deep in the money the call hedge
    holds almost one share, deep out of the money almost none.
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
with st.expander("🎓 Deep dive: where does the oscillating error come from?"):
    st.markdown(r"""
    The call payoff $(S - K)^+$ has a **kink** at $K$. The tree only "sees" the payoff at the terminal nodes,
    which are spaced by a factor $u^2$. How well the kink is resolved depends on **where $K$ falls between two
    neighbouring terminal nodes** — e.g. exactly on a node, or halfway between two. As $M$ changes, this relative
    position jumps around, and so does the error. A careful expansion (Diener & Diener, 2004) shows

    $$V_M - V_{BS} = \frac{c_1(\theta_M)}{M} + O\big(M^{-3/2}\big),$$

    where $\theta_M \in [0, 1)$ is the relative position of $\log(K/S(0))$ on the grid of terminal log-prices. The
    coefficient $c_1$ depends on $\theta_M$, which is why the error oscillates rather than decaying smoothly.

    **The fix:** choose the tree so that $\theta_M$ is constant, e.g. a node exactly at $K$ (the anchored tree with
    $ud = c$). The error is then a smooth $c/M$, and a smooth error is exactly what extrapolation methods need.
    Leisen and Reimer (1996) go further and choose $u, d, q$ via a normal-binomial inversion formula, which
    gives errors of order $1/M^2$ for odd $M$.
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
with st.expander("🎓 Deep dive: proving the bounds and the parity"):
    st.markdown(r"""
    All three statements follow from **static replication** — comparing portfolios at maturity — and therefore
    hold in every arbitrage-free model, not only in the tree.

    - **Parity.** Long call, short put: payoff $(S_T - K)^+ - (K - S_T)^+ = S_T - K$. The same payoff is produced by
      one share and a short position in a zero-coupon bond with face value $K$. Equal payoffs ⇒ equal prices:
      $V_{\text{call}} - V_{\text{put}} = S(0) - Ke^{-rT}$.
    - **Upper bound for the call.** $(S_T - K)^+ \le S_T$, so the call is worth at most one share: $V_{\text{call}} \le S(0)$.
    - **Lower bound for the call.** $(S_T - K)^+ \ge S_T - K$, so $V_{\text{call}} \ge S(0) - Ke^{-rT}$; together with
      $V_{\text{call}} \ge 0$ this gives $V_{\text{call}} \ge (S(0) - Ke^{-rT})^+$. Equivalently, by Jensen's inequality,
      $E_Q[e^{-rT}(S_T - K)^+] \ge (E_Q[e^{-rT}S_T] - Ke^{-rT})^+$.
    - **Put bounds.** $0 \le (K - S_T)^+ \le K$ gives $0 \le V_{\text{put}} \le Ke^{-rT}$.

    **Why does the price increase with $\sigma$?** The payoff is convex, so a more spread-out distribution of $S_T$
    (with the same mean $S(0)e^{rT}$) increases its expectation — options are "long volatility".
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
with st.expander("📖 Intuition: why trade barrier options, and how to treat the barrier properly"):
    st.markdown(r"""
    A knock-out option is **cheaper** than the vanilla option because it is worthless in some scenarios.
    Someone who believes the stock will never fall below $B$ gets the call payoff at a discount. Barrier options are
    popular in structured products and FX markets for exactly that reason.

    Numerically they are delicate: the payoff jumps from "full call" to zero when the barrier is touched, so small
    changes in how the barrier is represented cause large price changes. Remedies:

    - **Put the barrier on a node layer**: choose $M$ (or the anchoring constant $c$) so that $B = S(0)d^k$ for some $k$.
    - **Continuity correction** (Broadie-Glasserman-Kou): a discretely monitored barrier $B$ behaves like a continuous
      barrier shifted away from $S(0)$ by the factor $e^{0.5826\,\sigma\sqrt{\Delta t}}$.
    - **Trinomial trees**, which have an extra degree of freedom to align nodes with the barrier.

    Also distinguish the *contract*: a barrier monitored daily is a different product from one monitored
    continuously, and the tree with $M$ steps prices a barrier monitored at the $M$ tree dates.
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

with st.expander("🎓 Deep dive: American options as an optimal stopping problem"):
    st.markdown(r"""
    The holder of an American option chooses a **stopping time** $\tau$ (a decision rule that only uses
    information available up to now) and receives $g(S(\tau))$. The fair price is

    $$V(0) = \sup_{\tau} E_Q\big[e^{-r\tau} g(S(\tau))\big].$$

    In discrete time this is solved backwards by **dynamic programming**: at each node compare the immediate
    payoff with the discounted expected value of continuing,

    $$V(S(t_{i-1}), t_{i-1}) = \max\Big\{g(S(t_{i-1})),\; e^{-r\Delta t}E_Q\big[V(S(t_i), t_i) \mid S(t_{i-1})\big]\Big\}.$$

    The resulting process is the **Snell envelope** of the payoff: the smallest supermartingale (after discounting)
    that dominates the payoff. Exercising at the first time $V = g$ is optimal.

    **Why is early exercise of a call never optimal (no dividends, $r \ge 0$)?** At any $t < T$,

    $$V_{\text{call}}(t) \ge S(t) - Ke^{-r(T-t)} \ge S(t) - K,$$

    so the option alive is always worth at least as much as exercising it. Intuitively: exercising early means paying
    $K$ earlier (losing interest) and giving up the insurance against the stock falling below $K$.
    For the put, receiving $K$ early *earns* interest, so deep in the money exercise becomes optimal.
    """)

with st.expander("🎓 Deep dive: the algebra behind Richardson extrapolation"):
    st.markdown(r"""
    Suppose the error has the form $V_M = V + \frac{c}{M} + O(M^{-2})$ with an **unknown but fixed** constant $c$.
    Then

    $$2V_{2M} - V_M = 2V + \frac{2c}{2M} - V - \frac{c}{M} + O(M^{-2}) = V + O(M^{-2}),$$

    so the leading error term cancels. The same idea underlies Romberg integration and is used with Monte Carlo
    discretisation bias and finite differences (Section 5.4).

    The catch: the assumption "fixed $c$" must hold. For the plain CRR tree $c$ oscillates with $M$ (see Section 2),
    and the extrapolation can make things *worse*. The anchored tree makes the error regular; using the pair
    $(2M, 4M)$ ensures both trees have the same parity, so they over- or undershoot in the same way.
    """)

takeaways([
    "The binomial tree prices by **replication**: the price is the cost of the hedging portfolio, which gives the risk-neutral weights $q$.",
    "With moment-matched $u, d, q$ the tree converges to Black-Scholes (central limit theorem), with an oscillating $O(1/M)$ error.",
    "Barriers and American exercise fit naturally into backward induction; barriers converge slowly because of the grid.",
    "Anchoring the tree at the strike makes the error smooth, and Richardson extrapolation then removes its leading term.",
])

test_yourself("binomial", [
    (r"Why does the real-world probability $p$ of an up-move not appear in the option price?",
     r"Because the option can be **replicated** by stock and bond in every one-period tree. The price must equal the cost of the replicating portfolio (otherwise arbitrage), and this cost does not depend on $p$. $q$ is just the weight that rewrites the cost as a discounted expectation."),
    (r"Derive $q$ from the martingale condition $E_Q[S(t_i)/B(t_i) \mid \mathcal F_{t_{i-1}}] = S(t_{i-1})/B(t_{i-1})$.",
     r"$q S u + (1-q) S d = S e^{r\Delta t}$, hence $q = \dfrac{e^{r\Delta t} - d}{u - d}$. It lies in $(0,1)$ iff $d < e^{r\Delta t} < u$ (no arbitrage)."),
    (r"How many nodes does a recombining tree with $M$ steps have, and how many paths? Why does that matter for path-dependent options?",
     r"$(M+1)(M+2)/2$ nodes but $2^M$ paths. Path-dependent payoffs generally need the whole path, so the cheap recombining structure is lost — barrier options are the lucky exception, because 'not yet knocked out' plus the current node is enough information."),
    (r"Show that an American call on a non-dividend-paying stock is worth the same as the European call when $r \ge 0$.",
     r"$V^{\text{Eur}}_{\text{call}}(t) \ge S(t) - Ke^{-r(T-t)} \ge S(t) - K$, so continuing is always worth at least the exercise value; the max in the American recursion always picks the continuation value."),
    (r"What must be true about the error of a method for Richardson extrapolation $2V_{2M} - V_M$ to help?",
     r"The error must have the regular form $c/M + O(M^{-2})$ with a constant $c$ independent of $M$. Oscillating errors (plain CRR near the strike) violate this."),
])
