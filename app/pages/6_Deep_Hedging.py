import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from util.black_scholes import bs_price, bs_greeks
from util.monte_carlo import gbm_paths
from util.heston import simulate_heston_paths, heston_fft_prices
from util.deep_hedging import train_deep_hedge
from util.ui import takeaways, test_yourself

st.title("🧠 Deep Hedging")
st.markdown(r"""
### Neural networks in one paragraph (Section 6.1)
A feed-forward neural network maps an input $x$ through layers of the form $x \mapsto \varrho(Wx + b)$ with
weight matrices $W$, bias vectors $b$ and a non-linear **activation** $\varrho$ (here $\tanh$). Stacking a few
such layers gives a flexible function $g_\theta$ whose parameters $\theta$ are fitted by **gradient descent**
on a loss function, with gradients computed by backpropagation. By the universal approximation theorem,
networks can approximate any continuous function on compact sets arbitrarily well.

### Hedging as a learning problem (Section 6.2)
We want a self-financing strategy $\varphi$ that replicates a claim $X$ as well as possible. The
**variance-optimal** hedge minimises the expected squared hedging error

$$E\Big[\big(\pi + (\varphi\cdot\hat S)_N - \hat X\big)^2\Big], \qquad (\varphi\cdot\hat S)_N = \sum_{t=1}^N\varphi_t\big(\hat S_t - \hat S_{t-1}\big),$$

with discounted prices $\hat S, \hat X$ and initial capital $\pi$. **Deep hedging** restricts the strategy to a
neural network, $\varphi_t = g_\theta(t, S_{t-1})$, simulates $n$ paths $s^j$ and minimises the empirical loss

$$\sum_{j=1}^n\Big(\pi + \sum_{t=1}^N g_\theta(t, s^j_{t-1})\big(\hat s^j_t - \hat s^j_{t-1}\big) - \hat x^j\Big)^2$$

over $\theta$ and $\pi$ — an ordinary parametric optimisation. Crucially, nothing in this recipe needs a
pricing formula, a PDE or completeness: it works for **any** model we can simulate from, including
incomplete markets and (with a different loss) transaction costs.
""")

with st.expander("📖 Implementation details of this page"):
    st.markdown(r"""
    - One network is **shared across time**; it receives the time to maturity $(T - t)/T$ and the scaled
      log-moneyness $\log(S_{t-1}/K)/\hat\sigma$ as inputs (the lecture allows a separate network $g_t$ per date;
      sharing is the common simplification and generalises better).
    - Architecture: 2 → 16 → 16 → 1 with $\tanh$ activations, trained with mini-batch **Adam**
      (gradients by hand-written backpropagation in NumPy — no deep learning library needed).
    - The initial capital $\pi$ is a trainable parameter. For the quadratic loss its optimum is the
      mean of the discounted payoff minus the hedge gains, i.e. a price under a (variance-optimal) martingale measure.
    - Evaluation uses **fresh test paths** not seen during training.
    """)

with st.expander("🎓 Deep dive: anatomy of a neural network"):
    st.markdown(r"""
    A single **neuron** computes a weighted sum of its inputs plus a bias and applies a non-linear activation:
    $y = \varrho(w^\top x + b)$ — loosely inspired by a biological neuron that "fires" once its input exceeds a threshold.
    A **layer** applies many neurons in parallel, $x \mapsto \varrho(Wx + b)$, and a **feed-forward network** chains layers:

    $$g_\theta(x) = W_L\,\varrho\big(W_{L-1}\cdots\varrho(W_1x + b_1)\cdots + b_{L-1}\big) + b_L, \qquad \theta = (W_1, b_1, \dots, W_L, b_L).$$

    - **Activation functions:** $\tanh$ (smooth, used here), ReLU $\max(x, 0)$, sigmoid $1/(1 + e^{-x})$. Without the
      non-linearity the whole network would collapse to one linear map.
    - **Depth and width:** the network on this page has 2 inputs, two hidden layers with 16 neurons and 1 output —
      $2\cdot16 + 16 + 16\cdot16 + 16 + 16 + 1 = 337$ parameters.
    - **Universal approximation theorem:** a network with one hidden layer and enough neurons can approximate any
      continuous function on a compact set arbitrarily well. Depth makes this more efficient for many functions.
    """)

with st.expander("🎓 Deep dive: training — gradient descent, backpropagation and Adam"):
    st.markdown(r"""
    Training minimises the empirical loss $L(\theta) = \frac1n\sum_j \ell_j(\theta)$ by **gradient descent**,
    $\theta \leftarrow \theta - \eta\nabla L(\theta)$ with learning rate $\eta$.

    - **Backpropagation** computes $\nabla L$ efficiently with the chain rule, layer by layer from the output back to the
      input — the cost is about twice a forward evaluation, independent of the number of parameters.
    - **Stochastic (mini-batch) gradient descent** uses a random subset of paths per step: cheaper steps, noisy but
      unbiased gradients, and the noise helps to escape poor regions.
    - **Adam** keeps running averages of the gradient (momentum) and of its square, and scales each parameter's step by
      $1/\sqrt{\text{average squared gradient}}$ — parameters with small gradients get relatively larger steps. It is the
      default optimiser in deep learning and the one used on this page.
    - **Overfitting:** with too few paths the network can learn the noise of the training sample. That is why the results
      are evaluated on fresh **test paths**.
    """)

with st.expander("📖 Intuition: why should the network find the delta at all?"):
    st.markdown(r"""
    In the Black-Scholes model the claim can be replicated, and the replicating strategy (the delta hedge, rebalanced
    continuously) makes the hedging error zero. With $N$ discrete dates no strategy is perfect, but the delta hedge is
    close to the variance-optimal one. Minimising the squared hedging error over all strategies of the form
    $\varphi_t = g_\theta(t, S_{t-1})$ must therefore lead to something very close to $\Phi(d_1)$ — the network
    *rediscovers* the delta purely from simulated data.

    In incomplete markets (Heston, transaction costs, discrete trading) there is no formula, but the same optimisation still
    finds the best strategy within the network class. That is the selling point of deep hedging: **the method does not
    care whether the model has a pricing formula — only whether it can be simulated.**
    """)

st.write("---")
c = st.columns(5)
model = c[0].radio("Market model", ["Black-Scholes", "Heston"])
S0 = c[1].number_input("S(0) = K", value=100.0, step=5.0)
T = c[2].select_slider("Maturity T", [1 / 12, 0.25, 0.5, 1.0], value=0.25, format_func=lambda v: f"{v:.3g} yrs")
N = c[3].select_slider("Rebalancing dates N", [5, 10, 20, 50], value=20)
r = c[4].number_input("r", value=0.02, step=0.01, format="%.3f")
K = S0
c = st.columns(4)
if model == "Black-Scholes":
    sigma = c[0].slider("σ", 0.05, 0.6, 0.2, step=0.01)
    hparams = None
else:
    gam0 = c[0].number_input("γ(0)", value=0.04, step=0.01, format="%.3f")
    kappa = c[1].number_input("κ", value=0.08, step=0.01, format="%.3f")
    lamb = c[2].number_input("λ", value=2.0, step=0.5)
    sig_t = c[3].number_input("σ̃", value=0.6, step=0.1)
    rho = st.slider("ρ", -0.95, 0.95, -0.7, step=0.05)
    hparams = (gam0, kappa, lamb, sig_t, rho)
    sigma = np.sqrt(gam0)
c = st.columns(3)
n_paths = c[0].select_slider("Training paths n", [5000, 10000, 20000], value=10000)
steps = c[1].select_slider("Training steps", [500, 1000, 1500, 2500], value=1500)
seed = c[2].number_input("Seed", value=0, step=1)


def simulate(n, seed):
    rng = np.random.default_rng(seed)
    if model == "Black-Scholes":
        return gbm_paths(S0, T, r, sigma, rng.standard_normal((n, N)) * np.sqrt(T / N), "exact")
    fine = 10
    S, _ = simulate_heston_paths(S0, T, r, *hparams, N * fine, n, rng)
    return S[:, ::fine]


config = (model, S0, T, N, r, sigma, hparams, n_paths, steps, seed)
if st.button("🧠 Train the hedging network", type="primary"):
    S_train = simulate(n_paths, int(seed))
    pay_train = np.maximum(S_train[:, -1] - K, 0)
    status = st.empty()
    bar = st.progress(0.0)
    t0 = time.perf_counter()

    def cb(step, loss):
        bar.progress(step / steps)
        status.text(f"Step {step}/{steps} — mean squared hedging error {loss:.4f}")

    net, hist = train_deep_hedge(S_train, pay_train, r, T, K, steps=steps, seed=int(seed), callback=cb)
    st.session_state["deep_hedge"] = dict(config=config, net=net, hist=hist, secs=time.perf_counter() - t0)
    bar.empty(); status.empty()

res = st.session_state.get("deep_hedge")
trained = res is not None and res["config"] == config
if not trained:
    st.info("Choose a market and press **Train** (takes roughly 10–30 seconds). Changing any setting requires retraining.")
else:

    net, hist = res["net"], res["hist"]
    S_test = simulate(20000, int(seed) + 12345)
    pay_test = np.maximum(S_test[:, -1] - K, 0)
    pnl_deep = net.pnl(S_test, pay_test, r)

    t_grid = np.linspace(0, T, N + 1)
    if model == "Black-Scholes":
        sig_bs = sigma
        ref_price = bs_price(S0, K, T, r, sigma)
    else:
        sig_bs = np.sqrt(kappa / lamb + (gam0 - kappa / lamb) * (1 - np.exp(-lamb * T)) / (lamb * T))
        ref_price = heston_fft_prices(S0, np.array([K]), T, r, *hparams)[0]
    delta_bs = np.column_stack([bs_greeks(S_test[:, i], K, T, r, sig_bs, t=t_grid[i])["Delta"] for i in range(N)])
    dS = np.diff(S_test * np.exp(-r * t_grid), axis=1)
    pnl_bs = ref_price + np.sum(delta_bs * dS, axis=1) - pay_test * np.exp(-r * T)

    m = st.columns(5)
    m[0].metric("Learned initial capital π", f"{net.pi:.4f}")
    m[1].metric("Model price" if model == "Heston" else "Black-Scholes price", f"{ref_price:.4f}")
    m[2].metric("Test RMSE — deep hedge", f"{np.sqrt(np.mean(pnl_deep**2)):.4f}")
    m[3].metric("Test RMSE — BS delta hedge", f"{np.sqrt(np.mean(pnl_bs**2)):.4f}")
    m[4].metric("Training time", f"{res['secs']:.0f} s")

    col1, col2 = st.columns(2)
    with col1:
        k = 25
        smooth = np.convolve(hist, np.ones(k) / k, mode="valid")
        fig = go.Figure(go.Scatter(y=smooth, line=dict(color="steelblue")))
        fig.add_hline(y=np.mean(pnl_bs**2), line_dash="dash", line_color="crimson", annotation_text="BS delta hedge (test)")
        fig.update_layout(yaxis_type="log", xaxis_title="Training step", yaxis_title="Mean squared hedging error",
                          title="Training loss (moving average)", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=pnl_bs, nbinsx=80, name="BS delta hedge", marker_color="crimson", opacity=0.5))
        fig2.add_trace(go.Histogram(x=pnl_deep, nbinsx=80, name="Deep hedge", marker_color="steelblue", opacity=0.5))
        fig2.update_layout(barmode="overlay", xaxis_title="Terminal hedging error (test paths)", yaxis_title="Count",
                           title="Hedging error distribution", height=380, legend=dict(x=0, y=1))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("What did the network learn?")
    S_axis = np.linspace(0.75 * K, 1.25 * K, 200)
    fig3 = make_subplots(rows=1, cols=3, subplot_titles=[f"t = {f * T:.3g}" for f in (0.0, 0.5, 0.9)], shared_yaxes=True)
    for j, frac in enumerate((0.0, 0.5, 0.9), start=1):
        t_ = frac * T
        fig3.add_trace(go.Scatter(x=S_axis, y=net.ratio(t_, S_axis), name="Network φ(t, S)", line=dict(color="steelblue", width=3),
                                  showlegend=(j == 1)), row=1, col=j)
        fig3.add_trace(go.Scatter(x=S_axis, y=bs_greeks(S_axis, K, T, r, sig_bs, t=t_)["Delta"], name="BS delta",
                                  line=dict(color="crimson", dash="dash"), showlegend=(j == 1)), row=1, col=j)
    fig3.update_xaxes(title_text="S")
    fig3.update_yaxes(title_text="Hedge ratio", row=1, col=1)
    fig3.update_layout(height=380, legend=dict(x=0, y=1))
    st.plotly_chart(fig3, use_container_width=True)

    rmse_d, rmse_b = np.sqrt(np.mean(pnl_deep**2)), np.sqrt(np.mean(pnl_bs**2))
    if model == "Black-Scholes":
        st.markdown(f"""
        **Interpretation:** Without ever seeing the Black-Scholes formula, the network rediscovers the delta:
        its hedge ratio follows $\\Phi(d_1)$ and steepens into a step function as maturity approaches. The learned
        initial capital $\\pi = {net.pi:.4f}$ is close to the Black-Scholes price {ref_price:.4f}. In this complete
        market the BS delta is (up to discrete rebalancing) optimal, so the network can at best match it — its
        test RMSE of {rmse_d:.4f} vs {rmse_b:.4f} reflects the remaining approximation and sampling error. This
        is the sanity check: in a world where theory gives the answer, the learning approach finds it.
        """)
    else:
        probe = np.array([0.9 * K, K, 1.1 * K])
        nn_h = net.ratio(0.5 * T, probe)
        bs_h = bs_greeks(probe, K, T, r, sig_bs, t=0.5 * T)["Delta"]
        st.markdown(f"""
        **Interpretation:** In the Heston model the market is incomplete and no formula gives the optimal
        stock-only hedge. The network learns it from simulated paths. At $t = T/2$ it holds
        {nn_h[0]:.3f} / {nn_h[1]:.3f} / {nn_h[2]:.3f} shares at $S = 0.9K / K / 1.1K$, versus
        {bs_h[0]:.3f} / {bs_h[1]:.3f} / {bs_h[2]:.3f} for the Black-Scholes delta with average volatility.
        Two effects drive the difference: the Heston distribution is skewed, so its own delta $\\partial_S v$
        differs from $\\Phi(d_1)$; and the variance-optimal stock position adds
        $\\frac{{\\rho\\widetilde\\sigma}}{{S}}\\partial_\\gamma v$ — with $\\rho = {rho} < 0$ a falling stock tends to come with
        rising volatility, which supports the call's value, so fewer shares are needed.

        The deep hedge's test RMSE is {rmse_d:.4f} compared with {rmse_b:.4f} for the BS delta. The remaining
        error is genuine **unhedgeable volatility risk** — to remove it one would also need to trade a volatility
        instrument (e.g. another option), which deep hedging handles by simply adding it as a further hedging asset.
        """)

with st.expander("🎓 Deep dive: beyond the quadratic loss — risk measures and transaction costs"):
    st.markdown(r"""
    The quadratic loss penalises gains and losses symmetrically, which is the classical **variance-optimal hedging**
    criterion (Föllmer-Schweizer, known from the discrete-time Mathematical Finance course). Buehler, Gonon, Teichmann
    and Wood (2019), on which Section 6.2 is based, allow general **convex risk measures** $\rho$ instead:

    $$\min_{\theta}\;\rho\Big(-\big(\pi + (g_\theta\cdot\hat S)_N - \hat X - \text{costs}\big)\Big),$$

    e.g. the **expected shortfall** of the hedging loss (the risk measure from the risk management course). The minimal
    $\pi$ is then an *indifference price*: the capital that makes holding the hedged short position acceptable.

    **Transaction costs** simply enter the loss as $\sum_t c\,|\varphi_t - \varphi_{t-1}|S_t$; the network then learns
    to trade less (a "no-trade band" around the delta). **Further inputs** — e.g. the current variance in Heston or the
    previous position — let the strategy depend on more information. None of these extensions changes the algorithm:
    simulate, evaluate the loss, follow the gradient.
    """)

takeaways([
    "A neural network is a flexible parametric function; training adjusts its parameters by gradient descent with backpropagation.",
    "Deep hedging parametrises the hedge $\\varphi_t = g_\\theta(t, S_{t-1})$ and minimises the simulated hedging loss over $\\theta$ (and the initial capital).",
    "In Black-Scholes the network rediscovers the delta; in incomplete markets it finds the best hedge without a formula.",
    "Risk measures, transaction costs and extra information enter only through the loss and inputs — the recipe stays the same.",
])

test_yourself("deephedging", [
    (r"Write down the optimisation problem solved in deep hedging with the quadratic criterion.",
     r"$\min_{\theta, \pi}\frac1n\sum_{j=1}^n\Big(\pi + \sum_{t=1}^N g_\theta(t, s^j_{t-1})(\hat s^j_t - \hat s^j_{t-1}) - \hat x^j\Big)^2$ over the network parameters $\theta$ and the initial capital $\pi$, using simulated paths $s^j$."),
    (r"Why is the learned $\pi$ close to the Black-Scholes price in the Black-Scholes model?",
     r"For the optimal hedge the hedging error has mean (almost) zero, so $\pi \approx E[\hat X - (\varphi\cdot\hat S)_N] = E_Q[\hat X]$ — the hedge gains have zero mean under $Q$ where the discounted stock is a martingale (the simulated paths are $Q$-paths)."),
    (r"What is backpropagation?",
     r"An efficient way to compute the gradient of the loss with respect to all network parameters by applying the chain rule backwards through the layers."),
    (r"Name two advantages of deep hedging over formula-based hedging.",
     r"It works in any model that can be simulated (incomplete markets, no closed forms), and it can include transaction costs, trading constraints and general risk measures directly in the objective."),
])
