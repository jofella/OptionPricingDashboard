import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from util.binomial import crr_price, crr_tree, crr_params, crr_bermudan, payoff_fn
from util.black_scholes import bs_price
from util.monte_carlo import gbm_paths, longstaff_schwartz_put, apply_exercise_policy
from util.ui import takeaways, test_yourself

st.title("⏱️ American Options by Regression: Longstaff-Schwartz")
st.markdown(r"""
The value of an American option is an **optimal stopping problem**,
$V(0) = \sup_\tau E_Q\big[e^{-r\tau}g(S(\tau))\big]$. On exercise dates $t_1 < \dots < t_m$ the holder
compares the immediate payoff $g(S(t_i))$ with the **continuation value**

$$C_i(s) = E_Q\big[e^{-r(\tau_{i+1} - t_i)}g(S(\tau_{i+1})) \,\big|\, S(t_i) = s\big].$$

Trees and PDEs compute $C_i$ on a grid; in Monte Carlo every path has a different future, so we cannot
average over "the" future of one node. **Longstaff and Schwartz** estimate $C_i$ by **least-squares
regression** across paths (Section 4.6):

1. Simulate $N$ paths and set $\tau_n = T$, cash flow $g_n = g(S_{n,m})$.
2. For $i = m-1, \dots, 1$: on the in-the-money paths regress the discounted future cash flows
   $e^{-r(\tau_n - t_i)}g_n$ on basis functions $\ell_0(S_{n,i}), \dots, \ell_M(S_{n,i})$ (here Laguerre polynomials).
   Where $g(S_{n,i}) \ge \widehat C_i(S_{n,i})$, exercise: set $\tau_n = t_i$, $g_n = g(S_{n,i})$.
3. $\widehat V_0 = \max\big\{g(S(0)),\; \frac1N\sum_n e^{-r\tau_n}g_n\big\}$.
""")

with st.expander("📖 Why regress only on in-the-money paths, and is the estimate biased?"):
    st.markdown(r"""
    Exercise is only an option where $g > 0$; out-of-the-money paths would never be exercised, so including
    them only distorts the fit in the region that matters. Note that the regression *decides* when to
    exercise, but the cash flow assigned to a path is always its **realised** discounted payoff, not the
    fitted value — this keeps the estimator from compounding regression errors.

    **Bias:** any exercise rule is suboptimal, so applying the estimated rule to **new, independent paths**
    gives a low-biased estimate (a lower bound in expectation). Evaluating on the same paths that were
    used for the regression has a small high bias (the rule "peeks" at the future). The truth is
    typically bracketed by the two.
    """)

with st.expander("📖 Intuition: why American options are hard for Monte Carlo"):
    st.markdown(r"""
    Monte Carlo simulates **forward** in time, but the exercise decision needs information about the **future**:
    exercising today is optimal only if it beats the expected value of waiting. On a tree every node has many
    successors over which we can average; a simulated path has exactly **one** future. A naive "simulation within
    the simulation" (new paths from every node) explodes in cost.

    Longstaff and Schwartz's idea: although one path's future is very noisy, the paths that are at similar stock prices
    today together reveal the *average* future — and averaging across neighbours is precisely what a **regression**
    does.
    """)

st.write("---")
c = st.columns(5)
S0 = c[0].number_input("S(0)", value=100.0, step=5.0)
K = c[1].number_input("Strike K", value=100.0, step=5.0)
T = c[2].number_input("T", value=1.0, min_value=0.1, step=0.25)
r = c[3].number_input("r", value=0.06, step=0.01, format="%.3f")
sigma = c[4].number_input("σ", value=0.30, min_value=0.01, step=0.05)
c = st.columns(4)
N = c[0].select_slider("Paths N", [5000, 10000, 20000, 50000], value=20000)
m = c[1].select_slider("Exercise dates m", [10, 25, 50, 100], value=50)
deg = c[2].slider("Basis degree M (Laguerre ℓ₀…ℓ_M)", 1, 8, 3)
seed = c[3].number_input("Seed", value=20, step=1)


@st.cache_data
def run_ls(S0, K, T, r, sigma, N, m, deg, seed):
    rng = np.random.default_rng(int(seed))
    dW = rng.standard_normal((N, m)) * np.sqrt(T / m)
    S = gbm_paths(S0, T, r, sigma, dW, "exact")
    V0, eps, tau, ex_pts, fits = longstaff_schwartz_put(S, K, T, r, deg, return_details=True)
    S_new = gbm_paths(S0, T, r, sigma, np.random.default_rng(int(seed) + 999).standard_normal((N, m)) * np.sqrt(T / m), "exact")
    V_out, eps_out = apply_exercise_policy(S_new, K, T, r, fits, deg)
    return V0, eps, V_out, eps_out, tau, ex_pts, fits, S[:200]


@st.cache_data
def crr_reference(S0, K, T, r, sigma):
    g = payoff_fn("put", K)
    ref = crr_price(S0, T, r, sigma, 10000, g, american=True)
    M = 400
    S_mat, V_mat, (u, d, q) = crr_tree(S0, T, r, sigma, M, g, american=True)
    dt = T / M
    bnd_t, bnd_s = [], []
    for i in range(1, M):
        cont = np.exp(-r * dt) * (q * V_mat[1:i + 2, i + 1] + (1 - q) * V_mat[:i + 1, i + 1])
        ex = g(S_mat[:i + 1, i])
        early = (ex > cont + 1e-10) & (ex > 0)
        if early.any():
            bnd_t.append(i * dt)
            bnd_s.append(S_mat[:i + 1, i][early].max())
    return ref, np.array(bnd_t), np.array(bnd_s)


with st.spinner("Simulating paths and running the backward regression..."):
    V0, eps, V_out, eps_out, tau, ex_pts, fits, S_show = run_ls(S0, K, T, r, sigma, N, m, deg, seed)
    ref, bnd_t, bnd_s = crr_reference(S0, K, T, r, sigma)
eu = bs_price(S0, K, T, r, sigma, call=False)

mm = st.columns(5)
mm[0].metric("LS price (in-sample)", f"{V0:.4f}", f"± {eps:.4f}", delta_color="off")
mm[1].metric("LS price (out-of-sample)", f"{V_out:.4f}", f"± {eps_out:.4f}", delta_color="off")
mm[2].metric("CRR reference (M = 10 000)", f"{ref:.4f}")
mm[3].metric("European put (BS)", f"{eu:.4f}")
mm[4].metric("Early exercise premium", f"{ref - eu:.4f}")

# ── Regression at one date ───────────────────────────────────────────
st.header("1. The Regression Step")
dates = sorted(fits.keys())
i_sel = st.select_slider("Exercise date t_i", options=dates, value=dates[len(dates) // 2],
                         format_func=lambda i: f"t = {i * T / m:.3f}")
x, y, cont, ex_now, a = fits[i_sel]
Sx = x * K
order = np.argsort(Sx)
sub = np.random.default_rng(0).choice(len(Sx), size=min(3000, len(Sx)), replace=False)
fig = go.Figure()
fig.add_trace(go.Scatter(x=Sx[sub], y=y[sub], mode="markers", marker=dict(size=3, color="lightsteelblue"),
                         name="Discounted future cash flow (ITM paths)"))
fig.add_trace(go.Scatter(x=Sx[order], y=cont[order], name="Fitted continuation value Ĉᵢ(S)", line=dict(color="steelblue", width=3)))
fig.add_trace(go.Scatter(x=Sx[order], y=K - Sx[order], name="Exercise value K − S", line=dict(color="crimson", width=3)))
if ex_now.any():
    fig.add_vline(x=Sx[ex_now].max(), line_dash="dash", line_color="black", annotation_text="exercise below here", annotation_position="bottom left")
fig.update_layout(xaxis_title=f"S(t_i), t_i = {i_sel * T / m:.3f}", yaxis_title="Value", height=430, legend=dict(x=0.45, y=1))
st.plotly_chart(fig, use_container_width=True)
st.markdown(f"""
**Interpretation:** Each light dot is one in-the-money path: its stock price today and the discounted
cash flow it will actually receive under the current exercise rule. The cloud is very noisy — one path's
future says little — but the regression across {len(x):,} paths recovers the **conditional expectation**
(blue). Where the red exercise value lies above the blue curve, exercising now beats waiting; the
crossing point is the estimated **exercise boundary** at this date.
""")
with st.expander("🎓 Deep dive: regression = conditional expectation (L² projection)"):
    st.markdown(r"""
    The conditional expectation $C_i(S) = E[Y \mid S(t_i) = S]$ of the discounted future cash flow $Y$ is the function of
    $S(t_i)$ that minimises the mean squared error $E\big[(Y - c(S(t_i)))^2\big]$ — it is the **orthogonal projection** of $Y$
    onto the space $L^2$ of functions of $S(t_i)$. Restricting $c$ to the span of basis functions,
    $c(S) = \sum_{k=0}^M a_k\ell_k(S)$, and replacing the expectation by the sample average gives ordinary least squares:

    $$\hat a = \arg\min_a \sum_{n\in\text{ITM}}\Big(Y_n - \sum_k a_k\ell_k(S_{n,i})\Big)^2.$$

    As the number of paths $N \to \infty$ the coefficients converge to the projection; as the number of basis
    functions grows, the projection converges to the true conditional expectation (Clément, Lamberton & Protter, 2002).
    Laguerre polynomials are the original choice of Longstaff and Schwartz, but simple polynomials $1, S, S^2, \dots$
    work similarly — scaling $S$ (here by $K$) keeps the regression matrix well conditioned.
    """)

st.write("---")

# ── Exercise boundary ────────────────────────────────────────────────
st.header("2. The Exercise Boundary")
fig2 = go.Figure()
tg = np.linspace(0, T, S_show.shape[1])
for k in range(40):
    fig2.add_trace(go.Scatter(x=tg, y=S_show[k], mode="lines", line=dict(color="lightgrey", width=0.8), showlegend=False))
stop = tau[:40]
fig2.add_trace(go.Scatter(x=stop[stop < T], y=S_show[np.arange(40), np.round(stop / T * m).astype(int)][stop < T],
                          mode="markers", marker=dict(color="black", size=7, symbol="x"), name="Exercised here"))
if len(ex_pts):
    fig2.add_trace(go.Scatter(x=ex_pts[:, 0], y=ex_pts[:, 1], mode="markers", marker=dict(color="crimson", size=5),
                              name="LS boundary (highest exercised S per date)"))
fig2.add_trace(go.Scatter(x=bnd_t, y=bnd_s, name="CRR boundary", line=dict(color="black", width=2)))
fig2.add_hline(y=K, line_dash="dot", line_color="grey", annotation_text="K")
fig2.update_layout(xaxis_title="t", yaxis_title="S(t)", height=450)
st.plotly_chart(fig2, use_container_width=True)
st.markdown(f"""
**Interpretation:** The put is exercised as soon as a path drops below the boundary $S^*(t)$, which rises
to $K$ at maturity (just before expiry there is no time value left to lose). The regression-based boundary
(red) scatters around the binomial one (black) because it is estimated from finitely many noisy paths,
yet the resulting price is accurate: in-sample {V0:.4f}, out-of-sample {V_out:.4f}, reference {ref:.4f}.
The pricing error is **second order** in the boundary error, because at the optimal boundary the
value is flat with respect to small changes of the rule.
""")
with st.expander("📖 Intuition: why the price is accurate even though the boundary is noisy"):
    st.markdown(r"""
    At the optimal boundary, the holder is **indifferent** between exercising and continuing. Moving the boundary a little
    therefore changes the value only to second order (an envelope-theorem argument). A boundary error of 1% causes a
    price error far below 1% — which is why rough regressions still give good prices, but also why the *boundary*
    itself is estimated much less precisely than the price.
    """)

st.write("---")

# ── Sensitivity ──────────────────────────────────────────────────────
st.header("3. Influence of the Basis and of the Number of Exercise Dates")


@st.cache_data
def sensitivity(S0, K, T, r, sigma, N, seed):
    rows = []
    g = payoff_fn("put", K)
    for d in range(1, 7):
        out = run_ls(S0, K, T, r, sigma, N, 50, d, seed)
        rows.append(("degree", d, out[0], out[2], out[3], crr_bermudan(S0, T, r, sigma, 50, g)))
    for mm_ in (5, 10, 25, 50, 100):
        out = run_ls(S0, K, T, r, sigma, N, mm_, 3, seed)
        rows.append(("dates", mm_, out[0], out[2], out[3], crr_bermudan(S0, T, r, sigma, mm_, g)))
    return rows


with st.spinner("Re-running Longstaff-Schwartz for several settings (cached)..."):
    rows = sensitivity(S0, K, T, r, sigma, min(N, 20000), seed)
deg_rows = [row[1:] for row in rows if row[0] == "degree"]
date_rows = [row[1:] for row in rows if row[0] == "dates"]


def sens_fig(rows_, xtitle, log_x):
    x = [row[0] for row in rows_]
    f = go.Figure()
    f.add_trace(go.Scatter(x=x, y=[row[4] for row in rows_], mode="lines", name="exact Bermudan value (binomial)",
                           line=dict(color="black", width=2)))
    f.add_trace(go.Scatter(x=x, y=[row[1] for row in rows_], mode="lines+markers",
                           name="LS in-sample (high bias)", line=dict(color="steelblue")))
    f.add_trace(go.Scatter(x=x, y=[row[2] for row in rows_], mode="lines+markers", name="LS out-of-sample (low bias)",
                           line=dict(color="crimson"), error_y=dict(type="data", array=[row[3] for row in rows_], visible=True)))
    f.add_hline(y=ref, line_dash="dash", line_color="grey", annotation_text="American (CRR)", annotation_position="top left")
    f.update_layout(xaxis_title=xtitle, yaxis_title="Price", height=380, legend=dict(x=0.3, y=0.02),
                    xaxis_type="log" if log_x else "linear")
    return f


col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(sens_fig(deg_rows, "Basis degree M (m = 50 dates)", False), use_container_width=True)
with col2:
    st.plotly_chart(sens_fig(date_rows, "Exercise dates m (degree 3)", True), use_container_width=True)
berm = [row[4] for row in date_rows]
half_w = np.mean([row[3] for row in date_rows])
st.markdown(f"""
**Interpretation:** A constant-plus-linear basis ($M = 1$) is too rigid to represent the curved continuation
value, so the exercise rule is poor; from $M \\approx 2$–$3$ on the estimates stabilise — more basis functions
mainly add estimation noise. The **out-of-sample** estimate applies the fitted rule to fresh paths; since no
estimated rule beats the optimal one, it is low-biased. The **in-sample** estimate reuses the paths the rule was
fitted on, "peeks" at their futures and is biased upwards — which is why it can lie above the true value.

With only $m$ exercise dates the product is a **Bermudan** option. Its exact value (black, from a binomial tree that
allows exercise only on those dates) rises from {berm[0]:.3f} for $m = 5$ to {berm[-1]:.3f} for $m = 100$, towards
the American value {ref:.3f}. This effect is small compared with the Monte Carlo error bars (±{half_w:.2f}), so
the individual LS estimates do not necessarily line up monotonically — to resolve such differences one needs
far more paths. Compared with the binomial tree, Longstaff-Schwartz is slower in one dimension but generalises
to many underlyings and complex dynamics, where trees and PDE grids suffer from the curse of dimension.
""")

with st.expander("🎓 Deep dive: lower and upper bounds"):
    st.markdown(r"""
    - **Lower bound.** Any exercise rule $\tau$ gives $E_Q[e^{-r\tau}g(S_\tau)] \le V(0)$. Evaluating the fitted rule on
      **independent** paths yields an unbiased estimate of this suboptimal value → low-biased price.
    - **Upper bound (dual method).** Rogers (2002) and Haugh-Kogan showed
      $V(0) = \inf_M E\big[\max_t (e^{-rt}g(S_t) - M_t)\big]$ over martingales $M$ with $M_0 = 0$. Any martingale gives an
      upper bound; Andersen and Broadie (2004) build a good one from the regression's continuation values.

    The gap between the two bounds is an honest error measure for the exercise rule — a standard way to report
    American Monte Carlo prices in practice.
    """)

takeaways([
    "American options need the continuation value, which a single simulated path cannot provide.",
    "Longstaff-Schwartz estimates it by least-squares regression across in-the-money paths (an $L^2$ projection).",
    "Cash flows use realised payoffs, not fitted values; the fitted rule only decides when to exercise.",
    "Out-of-sample evaluation gives a lower bound; duality gives an upper bound; few exercise dates price a Bermudan option.",
])

test_yourself("lsm", [
    (r"Describe the backward step of the Longstaff-Schwartz algorithm at date $t_i$.",
     r"On in-the-money paths, regress the discounted realised future cash flows on basis functions of $S(t_i)$; where the immediate payoff is at least the fitted continuation value, set the exercise time to $t_i$ and the cash flow to $g(S(t_i))$."),
    (r"Why are only in-the-money paths used in the regression?",
     r"Exercise is only relevant where $g > 0$; fitting the continuation value on the relevant region gives a more accurate fit with the same number of basis functions."),
    (r"Why does applying the estimated exercise rule to new paths give a low-biased estimate?",
     r"The true price is the supremum over all stopping rules; any fixed (estimated) rule achieves at most this value, and on independent paths its value is estimated without bias."),
    (r"How does the price change with the number $m$ of exercise dates, and why?",
     r"It increases towards the American price: more dates means more exercise opportunities (a Bermudan option with $m$ dates is worth less than or equal to one with more dates)."),
])
