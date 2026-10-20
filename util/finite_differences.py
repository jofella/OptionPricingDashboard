"""Finite differences for the Black-Scholes PDE after the heat-equation transformation (Chapter 5).

x = log(S/K),  t~ = sigma^2 (T - t) / 2,  q = 2r / sigma^2,
v(t, S) = K exp(-(q-1) x / 2 - (q+1)^2 t~ / 4) w(t~, x),   w_t~ = w_xx.
"""
import numpy as np
from scipy.linalg import solve_banded


def _boundaries(kind, q):
    """Initial condition w(0, x) and boundary functions r1(t~, a), r2(t~, b) in transformed variables."""
    e = lambda c, x, t: np.exp(c * x / 2 + c**2 * t / 4)
    if kind == "call":
        w0 = lambda x: np.maximum(np.exp(x / 2 * (q + 1)) - np.exp(x / 2 * (q - 1)), 0)
        r1 = lambda t, x: 0.0 * x
        r2 = lambda t, x: e(q + 1, x, t) - e(q - 1, x, t)
    elif kind == "put":
        w0 = lambda x: np.maximum(np.exp(x / 2 * (q - 1)) - np.exp(x / 2 * (q + 1)), 0)
        r1 = lambda t, x: e(q - 1, x, t) - e(q + 1, x, t)
        r2 = lambda t, x: 0.0 * x
    elif kind == "american put":
        w0 = lambda x: np.maximum(np.exp(x / 2 * (q - 1)) - np.exp(x / 2 * (q + 1)), 0)
        r1 = lambda t, x: np.exp((q + 1) ** 2 * t / 4) * np.maximum(np.exp(x / 2 * (q - 1)) - np.exp(x / 2 * (q + 1)), 0)
        r2 = lambda t, x: 0.0 * x
    else:
        raise ValueError(kind)
    return w0, r1, r2


def solve_heat_theta(r, sigma, T, K, a, b, m, nu_max, theta=0.5, kind="call", zero_boundary=False):
    """theta-scheme: theta = 0 explicit, 1 fully implicit, 1/2 Crank-Nicolson.

    zero_boundary=True imposes w = 0 at both ends (the naive choice that breaks the call surface).

    Returns t (ascending), S (grid incl. boundaries), V[t, S], and lambda = dt~ / dx^2."""
    q = 2 * r / sigma**2
    dx = (b - a) / m
    dt = sigma**2 * T / (2 * nu_max)
    lam = dt / dx**2
    x = np.linspace(a, b, m + 1)
    tt = np.arange(nu_max + 1) * dt
    w0, r1, r2 = _boundaries(kind, q)
    if zero_boundary:
        r1 = r2 = lambda t, x: 0.0 * x

    w = np.zeros((nu_max + 1, m + 1))
    w[0] = w0(x)
    w[:, 0] = r1(tt, a)
    w[:, -1] = r2(tt, b)
    if not zero_boundary:
        w[0, 0], w[0, -1] = w0(np.array([a, b]))

    n_in = m - 1
    ab = np.zeros((3, n_in))
    ab[0, 1:] = -theta * lam
    ab[1, :] = 1 + 2 * theta * lam
    ab[2, :-1] = -theta * lam
    american = kind == "american put"

    with np.errstate(over="ignore", invalid="ignore"):
        for nu in range(nu_max):
            wn = w[nu]
            rhs = wn[1:-1] + (1 - theta) * lam * (wn[:-2] - 2 * wn[1:-1] + wn[2:])
            if theta > 0:
                rhs[0] += theta * lam * w[nu + 1, 0]
                rhs[-1] += theta * lam * w[nu + 1, -1]
                w[nu + 1, 1:-1] = solve_banded((1, 1), ab, rhs)
            else:
                w[nu + 1, 1:-1] = rhs
            if american:
                # projection onto the obstacle (Brennan-Schwartz style early exercise)
                w[nu + 1, 1:-1] = np.maximum(w[nu + 1, 1:-1], r1(tt[nu + 1], x[1:-1]))

    S = K * np.exp(x)
    t = T - 2 / sigma**2 * tt
    V = K * w * np.exp(-x / 2 * (q - 1) - (q + 1) ** 2 / 4 * tt[:, None])
    # flip so that time runs forward (row 0 = t = 0)
    return t[::-1], S, V[::-1], lam


def thomas(alpha, beta, gamma, rhs):
    """Tridiagonal solver from the lecture (diag alpha, upper beta, lower gamma)."""
    n = len(alpha)
    a_hat = np.array(alpha, dtype=float)
    b_hat = np.array(rhs, dtype=float)
    for i in range(n - 1):
        factor = gamma[i] / a_hat[i]
        a_hat[i + 1] -= factor * beta[i]
        b_hat[i + 1] -= factor * b_hat[i]
    x = np.zeros(n)
    x[-1] = b_hat[-1] / a_hat[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (b_hat[i] - beta[i] * x[i + 1]) / a_hat[i]
    return x
