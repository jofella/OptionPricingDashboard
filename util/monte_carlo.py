import numpy as np
from scipy.special import eval_laguerre
from scipy.stats import norm


def gbm_terminal(S0, T, r, sigma, X):
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * X)


def mc_summary(samples):
    """Estimate, sample variance and half-width of the approximate 95% confidence interval."""
    n = len(samples)
    est = np.mean(samples)
    var = np.var(samples, ddof=1)
    return est, var, 1.96 * np.sqrt(var / n)


def running_mean(x, axis=-1):
    n = np.arange(1, x.shape[axis] + 1)
    return np.cumsum(x, axis=axis) / n


# ── SDE schemes (Section 4.5) ──────────────────────────────────────────
def gbm_paths(S0, T, r, sigma, dW, scheme="exact"):
    """Paths of dS = rS dt + sigma S dW on an equidistant grid, driven by the given increments dW (N x m)."""
    N, m = dW.shape
    dt = T / m
    S = np.empty((N, m + 1))
    S[:, 0] = S0
    if scheme == "exact":
        W = np.cumsum(dW, axis=1)
        t = np.arange(1, m + 1) * dt
        S[:, 1:] = S0 * np.exp((r - 0.5 * sigma**2) * t + sigma * W)
        return S
    for i in range(m):
        s = S[:, i]
        S[:, i + 1] = s + r * s * dt + sigma * s * dW[:, i]
        if scheme == "milstein":
            S[:, i + 1] += 0.5 * sigma**2 * s * (dW[:, i] ** 2 - dt)
    return S


def coarsen(dW, factor):
    """Sum consecutive Brownian increments so a coarse path uses the same Brownian motion."""
    N, m = dW.shape
    return dW.reshape(N, m // factor, factor).sum(axis=2)


# ── Sensitivities (Section 4.4) ────────────────────────────────────────
def delta_fd(S, K, T, r, sigma, X, h):
    """Central finite differences with common random numbers."""
    up = np.exp(-r * T) * np.maximum(gbm_terminal(S + h / 2, T, r, sigma, X) - K, 0)
    dn = np.exp(-r * T) * np.maximum(gbm_terminal(S - h / 2, T, r, sigma, X) - K, 0)
    return (up - dn) / h


def delta_pathwise(S, K, T, r, sigma, X, digital=False):
    """Infinitesimal perturbation: d/dS of the discounted payoff along each path."""
    ST = gbm_terminal(S, T, r, sigma, X)
    if digital:
        return np.zeros_like(X)  # derivative of an indicator is 0 almost surely
    return np.exp(-r * T) * (ST > K) * ST / S


def delta_lr(S, K, T, r, sigma, X, digital=False):
    """Likelihood ratio: payoff times the score d/dS log density of S(T) = X / (S sigma sqrt T)."""
    ST = gbm_terminal(S, T, r, sigma, X)
    pay = (ST >= K).astype(float) if digital else np.maximum(ST - K, 0)
    return np.exp(-r * T) * pay * X / (S * sigma * np.sqrt(T))


def digital_delta_exact(S, K, T, r, sigma):
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-r * T) * norm.pdf(d2) / (S * sigma * np.sqrt(T))


# ── Longstaff-Schwartz (Section 4.6) ───────────────────────────────────
def longstaff_schwartz_put(S, K, T, r, n_basis, return_details=False):
    """Regression-based Monte Carlo for an American (Bermudan) put on simulated paths S (N x (m+1)).

    Continuation values are regressed on Laguerre polynomials of the moneyness S/K
    (scaling keeps the regression well conditioned)."""
    N, m1 = S.shape
    m = m1 - 1
    dt = T / m
    tau = np.full(N, m)                       # exercise index
    cash = np.maximum(K - S[:, m], 0.0)       # cash flow at exercise
    exercise_pts = []
    fits = {}
    for i in range(m - 1, 0, -1):
        itm = np.where(K - S[:, i] > 0)[0]
        if itm.size == 0:
            continue
        y = np.exp(-r * (tau[itm] - i) * dt) * cash[itm]
        x = S[itm, i] / K
        X = np.column_stack([eval_laguerre(k, x) for k in range(n_basis + 1)])
        a = np.linalg.lstsq(X, y, rcond=None)[0]
        cont = X @ a
        ex_now = K - S[itm, i] >= cont
        idx = itm[ex_now]
        tau[idx] = i
        cash[idx] = K - S[idx, i]
        if return_details:
            fits[i] = (x, y, cont, ex_now, a)
            if idx.size:
                exercise_pts.append((i * dt, S[idx, i].max()))
    disc_cash = np.exp(-r * tau * dt) * cash
    V0 = max(max(K - S[0, 0], 0.0), disc_cash.mean())
    eps = 1.96 * np.sqrt(np.var(disc_cash, ddof=1) / N)
    if return_details:
        return V0, eps, tau * dt, np.array(exercise_pts), fits
    return V0, eps


def apply_exercise_policy(S, K, T, r, fits, n_basis):
    """Value of the put when the exercise rule estimated on other paths is applied to fresh paths S.

    Any fixed exercise rule is suboptimal, so this gives a low-biased (lower bound) estimate."""
    N, m1 = S.shape
    m = m1 - 1
    dt = T / m
    tau = np.full(N, m)
    alive = np.ones(N, dtype=bool)
    for i in range(1, m):
        if i not in fits:
            continue
        a = fits[i][4]
        itm = alive & (S[:, i] < K)
        if not itm.any():
            continue
        x = S[itm, i] / K
        X = np.column_stack([eval_laguerre(k, x) for k in range(n_basis + 1)])
        ex = (K - S[itm, i]) >= X @ a
        idx = np.where(itm)[0][ex]
        tau[idx] = i
        alive[idx] = False
    cash = np.maximum(K - S[np.arange(N), tau], 0.0)
    disc_cash = np.exp(-r * tau * dt) * cash
    return disc_cash.mean(), 1.96 * np.sqrt(np.var(disc_cash, ddof=1) / N)
