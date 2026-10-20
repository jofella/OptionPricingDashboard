import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.integrate import quad
from scipy.stats import norminvgauss

from util.black_scholes import bs_price, down_and_out_call
from util.monte_carlo import gbm_terminal, gbm_paths, mc_summary

st.title("📉 Variance Reduction")
st.markdown(r"""
The Monte Carlo error is $1.96\,\sigma/\sqrt N$. Instead of increasing $N$ we can reduce $\sigma^2$
by simulating a *different* random variable with the **same mean** but smaller variance (Section 4.3).
Three classic techniques:

| Technique | Idea | Works well when |
|---|---|---|
| Antithetic variables | pair every draw $X$ with $-X$ | the payoff is monotone in $X$ |
| Control variates | subtract a correlated quantity with **known** mean | a similar, analytically priced product exists |
| Importance sampling | sample from a shifted density and reweight | the payoff lives in a rare region |
""")
st.write("---")

c = st.columns(5)
S0 = c[0].number_input("S(0)", value=120.0, step=5.0)
K = c[1].number_input("Strike K", value=100.0, step=5.0)
T = c[2].number_input("T", value=1.0, min_value=0.1, step=0.25)
r = c[3].number_input("r", value=0.02, step=0.01, format="%.3f")
sigma = c[4].number_input("σ", value=0.30, min_value=0.01, step=0.05)

# ── 1. Antithetic ────────────────────────────────────────────────────
st.header("1. Antithetic Variables")
st.markdown(r"""
Since $X \sim \mathcal N(0,1)$ implies $-X \sim \mathcal N(0,1)$, the estimator

$$\widehat V^{\text{anti}}_N = \frac1N\sum_{n=1}^N \frac{h(X_n) + h(-X_n)}{2}$$

is unbiased. Its variance is $\frac{1}{2N}\big(\operatorname{Var}h(X) + \operatorname{Cov}(h(X), h(-X))\big)$,
so it helps exactly when $h(X)$ and $h(-X)$ are **negatively correlated** — e.g. for a monotone payoff.
Each pair costs **two** payoff evaluations, so a fair comparison uses $2N$ plain draws.
""")
c1, c2 = st.columns(2)
payoff_kind = c1.radio("Payoff", ["Call", "Put", "Straddle |S−K|"], horizontal=True)
N_a = c2.select_slider("Pairs N", [1000, 5000, 10000, 50000], value=10000)
if payoff_kind == "Call":
    h = lambda s: np.maximum(s - K, 0)
    true_a = bs_price(S0, K, T, r, sigma)
elif payoff_kind == "Put":
    h = lambda s: np.maximum(K - s, 0)
    true_a = bs_price(S0, K, T, r, sigma, False)
else:
    h = lambda s: np.abs(s - K)
    true_a = bs_price(S0, K, T, r, sigma) + bs_price(S0, K, T, r, sigma, False)

rng = np.random.default_rng(20)
X = rng.standard_normal(2 * N_a)
disc = np.exp(-r * T)
plain = disc * h(gbm_terminal(S0, T, r, sigma, X))                    # 2N plain draws
anti = 0.5 * disc * (h(gbm_terminal(S0, T, r, sigma, X[:N_a])) + h(gbm_terminal(S0, T, r, sigma, -X[:N_a])))
e_p, v_p, eps_p = mc_summary(plain)
e_a, v_a, eps_a = mc_summary(anti)
corr = np.corrcoef(h(gbm_terminal(S0, T, r, sigma, X[:N_a])), h(gbm_terminal(S0, T, r, sigma, -X[:N_a])))[0, 1]

m = st.columns(5)
m[0].metric("True price", f"{true_a:.4f}")
m[1].metric(f"Plain MC (2N = {2 * N_a:,})", f"{e_p:.4f}", f"± {eps_p:.4f}", delta_color="off")
m[2].metric(f"Antithetic (N = {N_a:,} pairs)", f"{e_a:.4f}", f"± {eps_a:.4f}", delta_color="off")
m[3].metric("Corr(h(X), h(−X))", f"{corr:+.3f}")
m[4].metric("Efficiency gain", f"{(eps_p / eps_a) ** 2:.2f}×")

n_ax = np.arange(1, N_a + 1)
idx = np.unique(np.geomspace(50, N_a, 300).astype(int)) - 1
fig = go.Figure()
fig.add_trace(go.Scatter(x=2 * n_ax[idx], y=(np.cumsum(plain[: 2 * N_a]) / np.arange(1, 2 * N_a + 1))[2 * n_ax[idx] - 1],
                         name="Plain MC", line=dict(color="steelblue")))
fig.add_trace(go.Scatter(x=2 * n_ax[idx], y=(np.cumsum(anti) / n_ax)[idx], name="Antithetic", line=dict(color="crimson")))
fig.add_hline(y=true_a, line_dash="dash", line_color="black", annotation_text="exact")
fig.update_layout(xaxis_type="log", xaxis_title="Payoff evaluations", yaxis_title="Estimate", height=380)
st.plotly_chart(fig, use_container_width=True)
st.markdown(f"""
**Interpretation:** For the {payoff_kind.lower()} the correlation between $h(X)$ and $h(-X)$ is
{corr:+.3f}. {'Because the payoff is monotone in X, a high draw is paired with a low one and their errors largely cancel — at equal computational cost the confidence interval shrinks by a factor ' + f'{eps_p / eps_a:.2f}.' if corr < 0 else 'The straddle is **not monotone**: both very high and very low draws pay a lot, so h(X) and h(−X) are positively correlated and antithetic sampling is *worse* than plain Monte Carlo at equal cost. Antithetic variables are no free lunch.'}
""")
st.write("---")

# ── 2. Control variates ──────────────────────────────────────────────
st.header("2. Control Variates — Pricing a Parisian Option")
st.markdown(r"""
A **cumulative Parisian down-and-out call** pays $(S(T)-K)^+$ unless the price spends more than a
total time $\tau$ below the barrier $B$:

$$V(T) = (S(T) - K)^+\,1_{\{\int_0^T 1_{\{S(t) < B\}}dt < \tau\}}.$$

No closed form exists, but a similar product does: the ordinary down-and-out call (knocked out at the
*first* touch) and of course the vanilla call. With a control $Y$ whose mean $E[Y]$ is known,

$$\widehat V_{CV} = \frac1N\sum_n\big(h(X_n) - \beta(Y_n - E[Y])\big), \qquad \beta^* = \frac{\operatorname{Cov}(h(X), Y)}{\operatorname{Var}(Y)},$$

reduces the variance by the factor $1 - \rho^2_{h(X),Y}$.
""")
c1, c2, c3, c4 = st.columns(4)
B_hi = max(float(min(K, S0)) - 1.0, 0.4 * S0 + 1.0)
B = c1.slider("Barrier B", float(0.4 * S0), B_hi, float(min(80.0, B_hi)))
tau_m = c2.slider("Trigger period τ (months)", 0.25, 6.0, 1.0, step=0.25)
m_steps = c3.select_slider("Time steps m", [50, 100, 250, 500], value=250)
control = c4.radio("Control variate Y", ["Down-and-out call", "Vanilla call"])
tau = tau_m / 12
N_cv = 10000


@st.cache_data
def parisian_sim(S0, K, B, T, r, sigma, m, tau, N, seed=20):
    rng = np.random.default_rng(seed)
    dW = rng.standard_normal((N, m)) * np.sqrt(T / m)
    S = gbm_paths(S0, T, r, sigma, dW, scheme="milstein")
    disc_call = np.exp(-r * T) * np.maximum(S[:, -1] - K, 0)
    below = (S[:, 1:] < B).sum(axis=1)
    f = disc_call * (below < tau * m)
    Y_bar = disc_call * (S.min(axis=1) >= B)
    return f, Y_bar, disc_call, S[:30]


f, Y_bar, Y_van, S_show = parisian_sim(S0, K, B, T, r, sigma, m_steps, tau, N_cv)
if control == "Down-and-out call":
    Y = Y_bar
    B_shift = B * np.exp(-0.5826 * sigma * np.sqrt(T / m_steps))   # continuity correction for discrete monitoring
    EY = down_and_out_call(S0, K, B_shift, T, r, sigma)
else:
    Y = Y_van
    EY = bs_price(S0, K, T, r, sigma)

e_mc, v_mc, eps_mc = mc_summary(f)
beta = np.cov(f, Y)[0, 1] / np.var(Y, ddof=1)
cv = f - beta * (Y - EY)
e_cv, v_cv, eps_cv = mc_summary(cv)
rho_fy = np.corrcoef(f, Y)[0, 1]

m = st.columns(5)
m[0].metric("Plain MC", f"{e_mc:.4f}", f"± {eps_mc:.4f}", delta_color="off")
m[1].metric("Control variate", f"{e_cv:.4f}", f"± {eps_cv:.4f}", delta_color="off")
m[2].metric("ρ(h(X), Y)", f"{rho_fy:.4f}")
m[3].metric("β*", f"{beta:.4f}")
m[4].metric("Variance reduction", f"{v_mc / v_cv:.0f}×")

col1, col2 = st.columns(2)
with col1:
    fig2 = go.Figure()
    tt = np.linspace(0, T, S_show.shape[1])
    for k in range(S_show.shape[0]):
        fig2.add_trace(go.Scatter(x=tt, y=S_show[k], mode="lines", line=dict(width=0.8), showlegend=False))
    fig2.add_hline(y=B, line_dash="dash", line_color="crimson", annotation_text="B")
    fig2.add_hline(y=K, line_dash="dot", line_color="grey", annotation_text="K")
    fig2.update_layout(xaxis_title="t", yaxis_title="S(t)", title="Milstein paths", height=380)
    st.plotly_chart(fig2, use_container_width=True)
with col2:
    sub = slice(0, 3000)
    fig3 = go.Figure(go.Scatter(x=Y[sub], y=f[sub], mode="markers", marker=dict(size=3, color="steelblue", opacity=0.5)))
    fig3.update_layout(xaxis_title="Control Y (discounted)", yaxis_title="Parisian payoff (discounted)",
                       title="Payoff vs control", height=380)
    st.plotly_chart(fig3, use_container_width=True)

st.markdown(f"""
**Interpretation:** Most points lie on the diagonal — on these paths both products pay the same.
Deviations come only from paths that dip below $B$: the Parisian option survives short dips that knock
out the barrier option. With correlation {rho_fy:.4f} the variance falls by a factor of
$1/(1-\\rho^2) \\approx {v_mc / v_cv:.0f}$, which would otherwise require {v_mc / v_cv:.0f} times as many paths.

**A subtle point:** the control variate needs the **exact** mean of $Y$. For the discretely monitored
barrier option on the simulated grid, the continuous-barrier closed form is slightly off; we use the
Broadie-Glasserman-Kou correction, i.e. the closed form evaluated at $Be^{{-0.5826\\,\\sigma\\sqrt{{\\Delta t}}}}$.
(Using the sample mean of $Y$ instead of $E[Y]$ would cancel the correction term completely.)
The vanilla call has an exact mean but is less correlated when $B$ is close to $S(0)$ — try moving the barrier.
""")
st.write("---")

# ── 3. Importance sampling ───────────────────────────────────────────
st.header("3. Importance Sampling in the Exponential NIG Model")
st.markdown(r"""
In the **exponential NIG model** $S(t) = S(0)e^{(r - \omega)t + X(t)}$, where $X(t) \sim \mathrm{NIG}(\alpha, \beta, \delta t, 0)$ has
much heavier tails than the normal distribution and $\omega = \delta\big(\sqrt{\alpha^2-\beta^2} - \sqrt{\alpha^2-(\beta+1)^2}\big)$
makes the discounted stock a martingale. Importance sampling draws $Y$ from an auxiliary density $h$ and
reweights:

$$V_{IS} = \frac1N\sum_{n=1}^N f(Y_n)\frac{g(Y_n)}{h(Y_n)}.$$

**Exponential tilting** chooses $h(x) = g(x)e^{\theta x}/E[e^{\theta X}]$; for NIG this is again NIG with
$\beta \to \beta + \theta$, and the weight is $g/h = e^{-\theta x}E[e^{\theta X}]$. The variance is minimal
when $E\big[f(X)^2e^{-\theta X}\big]E[e^{\theta X}]$ is minimal. Below, $X = \delta T Z$ with
$Z \sim \mathrm{NIG}(\alpha\delta T, \beta\delta T, 1, 0)$ as in the tutorial, and the tilt acts on $Z$.
""")

with st.expander("⚠️ Why the tutorial's α = 1 is a trap: infinite variance"):
    st.markdown(r"""
    The NIG density decays like $e^{-(\alpha - \beta)x}$ in the right tail. A call payoff grows like $e^{x}$,
    so $E[f(X)^2] < \infty$ requires $E[e^{2X}] < \infty$, i.e. $\alpha - \beta > 2$. With the tutorial's
    $\alpha = 1, \beta = 0$ the **plain Monte Carlo variance is infinite**: the estimator still converges
    (the mean is finite since $\alpha - \beta \ge 1$), but the CLT fails and the printed "confidence
    intervals" are meaningless — they jump whenever an extreme draw occurs.

    For the tilted estimator the second moment is finite only if $\delta T(2 - \alpha + \beta) < \theta < \delta T(\alpha - \beta)$.
    For $\alpha = 1$ this interval is empty — no tilt can fix it, it can only make the divergence slower,
    which is why the sample errors in the tutorial look much better near $\theta \approx 0.28$ without
    being trustworthy. We therefore default to $\alpha = 3$; set $\alpha = 1$ to see the effect.
    """)

c1, c2, c3, c4 = st.columns(4)
alpha = c1.slider("α (tail heaviness)", 1.0, 6.0, 3.0, step=0.5)
delta = c2.slider("δ (scale)", 0.05, 1.0, 0.27, step=0.01)
K_is = c3.slider("Strike for IS", float(0.5 * S0), float(3.0 * S0), float(K), step=5.0)
beta_n = 0.0
a_, b_ = alpha * delta * T, beta_n * delta * T
th_lo, th_hi = -a_ - b_, a_ - b_
theta = c4.slider("Tilt θ", float(th_lo + 1e-3), float(th_hi - 1e-3), float(min(0.1, th_hi - 1e-3)), step=0.005)
omega = delta * (np.sqrt(alpha**2 - beta_n**2) - np.sqrt(max(alpha**2 - (beta_n + 1) ** 2, 0.0)))
mgf = lambda t: np.exp(np.sqrt(a_**2 - b_**2) - np.sqrt(a_**2 - (b_ + t) ** 2))
payoff = lambda z: np.exp(-r * T) * np.maximum(S0 * np.exp((r - omega) * T + delta * T * z) - K_is, 0)
Ng = 10000


@st.cache_data
def nig_reference(a_, b_, S0, K_is, r, T, omega, delta):
    g = norminvgauss(a=a_, b=b_)
    z0 = (np.log(K_is / S0) - (r - omega) * T) / (delta * T)

    def integrand(z):  # payoff x density, evaluated in log space to avoid inf * 0 far in the tail
        lp = g.logpdf(z)
        return np.exp(-r * T) * (np.exp(np.log(S0) + (r - omega) * T + delta * T * z + lp) - K_is * np.exp(lp))

    return quad(integrand, z0, np.inf, limit=500)[0]


ref = nig_reference(a_, b_, S0, K_is, r, T, omega, delta)
Z = norminvgauss(a=a_, b=b_).rvs(Ng, random_state=20)
Yt = norminvgauss(a=a_, b=b_ + theta).rvs(Ng, random_state=20)
plain_is = payoff(Z)
is_w = payoff(Yt) * np.exp(-theta * Yt) * mgf(theta)
e_pl, v_pl, eps_pl = mc_summary(plain_is)
e_is, v_is, eps_is = mc_summary(is_w)

thetas = np.linspace(th_lo + 1e-3, th_hi - 1e-3, 150)
second_moment = np.array([np.mean(plain_is**2 * np.exp(-t * Z)) * mgf(t) for t in thetas])
theta_opt = thetas[np.argmin(second_moment)]
finite_lo = delta * T * (2 - alpha + beta_n)

m = st.columns(5)
m[0].metric("Reference price (integration)", f"{ref:.4f}")
m[1].metric("Plain MC", f"{e_pl:.4f}", f"± {eps_pl:.4f}", delta_color="off")
m[2].metric(f"IS with θ = {theta:.3f}", f"{e_is:.4f}", f"± {eps_is:.4f}", delta_color="off")
m[3].metric("Estimated optimal θ", f"{theta_opt:.3f}")
m[4].metric("Variance ratio plain / IS", f"{v_pl / v_is:.2f}×")

col1, col2 = st.columns(2)
with col1:
    zz = np.linspace(np.percentile(Z, 0.1), np.percentile(Yt, 99.9), 600)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=zz, y=norminvgauss(a=a_, b=b_).pdf(zz), name="g: original NIG", line=dict(color="steelblue")))
    fig4.add_trace(go.Scatter(x=zz, y=norminvgauss(a=a_, b=b_ + theta).pdf(zz), name="h: tilted NIG", line=dict(color="crimson")))
    pay_scaled = payoff(zz) / max(payoff(zz).max(), 1e-12) * norminvgauss(a=a_, b=b_).pdf(zz).max()
    fig4.add_trace(go.Scatter(x=zz, y=pay_scaled, name="payoff (scaled)", line=dict(color="grey", dash="dot")))
    fig4.update_layout(xaxis_title="z", yaxis_title="density", height=380, legend=dict(x=0, y=1))
    st.plotly_chart(fig4, use_container_width=True)
with col2:
    fig5 = go.Figure(go.Scatter(x=thetas, y=second_moment - ref**2, line=dict(color="steelblue")))
    fig5.add_vline(x=theta, line_dash="dot", line_color="crimson", annotation_text="current θ")
    if finite_lo > th_lo:
        fig5.add_vrect(x0=th_lo, x1=min(finite_lo, th_hi), fillcolor="crimson", opacity=0.08, line_width=0,
                       annotation_text="infinite variance", annotation_position="top left")
    fig5.update_layout(yaxis_type="log", xaxis_title="θ", yaxis_title="Estimated Var of IS estimator", height=380,
                       title="E[f(Z)²·exp(−θZ)]·E[exp(θZ)] − V²")
    st.plotly_chart(fig5, use_container_width=True)

st.markdown(f"""
**Interpretation:** Tilting with $\\theta > 0$ shifts probability mass to the right, where the call pays off,
and the weight $e^{{-\\theta z}}E[e^{{\\theta Z}}]$ corrects for over-sampling that region. At $\\theta = {theta:.3f}$
the variance is {v_pl / v_is:.2f}× smaller than plain Monte Carlo; the estimated optimum is
$\\theta^* \\approx {theta_opt:.3f}$. The effect grows for **out-of-the-money** strikes, where plain
Monte Carlo wastes most draws on zero payoffs — raise the strike to see it. {'The shaded area marks tilts for which the variance is infinite; the estimated curve there is unreliable.' if finite_lo > th_lo else ''}
{'**Warning:** with α ≤ 2 the plain estimator has infinite variance — compare the ± values with the reference price.' if alpha - beta_n <= 2 else ''}
""")
