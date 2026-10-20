"""Deep hedging (Section 6.2) with a small numpy neural network trained by Adam.

The hedge phi_t = g_theta(t, S_{t-1}) is one network shared over time (time-to-maturity is an input
feature), and pi + sum_t phi_t (S^_t - S^_{t-1}) - X^ is driven to zero in mean square.
"""
import numpy as np


class MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, np.sqrt(2 / (m + n)), (m, n)) for m, n in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(n) for n in sizes[1:]]

    def forward(self, x):
        self.acts = [x]
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if k < len(self.W) - 1:
                x = np.tanh(x)
            self.acts.append(x)
        return x

    def backward(self, grad_out):
        gW, gb = [], []
        g = grad_out
        for k in range(len(self.W) - 1, -1, -1):
            gW.insert(0, self.acts[k].T @ g)
            gb.insert(0, g.sum(axis=0))
            if k > 0:
                g = (g @ self.W[k].T) * (1 - self.acts[k] ** 2)
        return gW, gb


class DeepHedge:
    """Network plus the feature scaling it was trained with."""

    def __init__(self, net, pi, K, T, scale):
        self.net, self.pi, self.K, self.T, self.scale = net, pi, K, T, scale

    def features(self, t, S):
        return np.column_stack([(self.T - t) / self.T, np.log(S / self.K) / self.scale])

    def ratio(self, t, S):
        S = np.asarray(S, dtype=float)
        return self.net.forward(self.features(np.broadcast_to(t, S.shape).ravel(), S.ravel())).reshape(S.shape)

    def strategy(self, S):
        n, N1 = S.shape
        t = np.tile(np.linspace(0, self.T, N1)[:-1], n)
        return self.ratio(t, S[:, :-1].ravel()).reshape(n, N1 - 1)

    def pnl(self, S, payoff, r):
        t_grid = np.linspace(0, self.T, S.shape[1])
        dS = np.diff(S * np.exp(-r * t_grid), axis=1)
        return self.pi + np.sum(self.strategy(S) * dS, axis=1) - payoff * np.exp(-r * self.T)


def train_deep_hedge(S, payoff, r, T, K, hidden=(16, 16), steps=1500, batch=512, lr=5e-3, seed=0, callback=None):
    """Mini-batch Adam on the empirical quadratic hedging loss (6.3). Returns the DeepHedge and loss history."""
    rng = np.random.default_rng(seed)
    n, N1 = S.shape
    N = N1 - 1
    t_grid = np.linspace(0, T, N1)
    dS_all = np.diff(S * np.exp(-r * t_grid), axis=1)
    X_all = payoff * np.exp(-r * T)
    scale = max(np.std(np.log(S[:, -1] / S[:, 0])), 1e-3)

    model = DeepHedge(MLP([2, *hidden, 1], rng), float(np.mean(X_all)), K, T, scale)
    feats_all = model.features(np.tile(t_grid[:-1], n), S[:, :-1].ravel()).reshape(n, N, 2)

    params = model.net.W + model.net.b
    m_ = [np.zeros_like(p) for p in params] + [0.0]
    v_ = [np.zeros_like(p) for p in params] + [0.0]
    b1, b2, eps = 0.9, 0.999, 1e-8
    history = []
    for step in range(1, steps + 1):
        idx = rng.choice(n, size=min(batch, n), replace=False)
        nb = len(idx)
        phi = model.net.forward(feats_all[idx].reshape(-1, 2)).reshape(nb, N)
        err = model.pi + np.sum(phi * dS_all[idx], axis=1) - X_all[idx]
        history.append(np.mean(err**2))
        gW, gb = model.net.backward(((2 / nb) * err[:, None] * dS_all[idx]).reshape(-1, 1))
        grads = gW + gb + [2 * np.mean(err)]
        nW, nB = len(model.net.W), len(model.net.b)
        for k, g in enumerate(grads):
            m_[k] = b1 * m_[k] + (1 - b1) * g
            v_[k] = b2 * v_[k] + (1 - b2) * g * g
            upd = lr * (m_[k] / (1 - b1**step)) / (np.sqrt(v_[k] / (1 - b2**step)) + eps)
            if k < nW:
                model.net.W[k] -= upd
            elif k < nW + nB:
                model.net.b[k - nW] -= upd
            else:
                model.pi -= upd
        if callback is not None and step % 50 == 0:
            callback(step, np.mean(history[-50:]))
    return model, np.array(history)
