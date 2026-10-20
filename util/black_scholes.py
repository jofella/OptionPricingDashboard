import numpy as np
from scipy.stats import norm
from scipy.integrate import quad


def d1_d2(S, K, tau, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return d1, d1 - sigma * np.sqrt(tau)


def bs_price(S, K, T, r, sigma, call=True, t=0.0):
    """European call/put price in the Black-Scholes model (lecture formula, Section 2.4)."""
    tau = np.maximum(T - t, 1e-12)
    d1, d2 = d1_d2(S, K, tau, r, sigma)
    if call:
        return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    return K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, call=True, t=0.0):
    tau = np.maximum(T - t, 1e-12)
    d1, d2 = d1_d2(S, K, tau, r, sigma)
    pdf = norm.pdf(d1)
    disc = np.exp(-r * tau)
    gamma = pdf / (S * sigma * np.sqrt(tau))
    vega = S * pdf * np.sqrt(tau)
    if call:
        delta = norm.cdf(d1)
        theta = -S * pdf * sigma / (2 * np.sqrt(tau)) - r * K * disc * norm.cdf(d2)
        rho = K * tau * disc * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = -S * pdf * sigma / (2 * np.sqrt(tau)) + r * K * disc * norm.cdf(-d2)
        rho = -K * tau * disc * norm.cdf(-d2)
    speed = -gamma / S * (d1 / (sigma * np.sqrt(tau)) + 1)
    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho, "Speed": speed}


def price_by_integration(f, S, T, r, sigma, t=0.0):
    """V(t) = e^{-r(T-t)} / sqrt(2 pi) * int f(S e^{(r - s^2/2)(T-t) + s sqrt(T-t) x}) e^{-x^2/2} dx."""
    tau = T - t

    def integrand(x):
        return f(S * np.exp((r - 0.5 * sigma**2) * tau + sigma * np.sqrt(tau) * x)) * np.exp(-0.5 * x**2)

    return np.exp(-r * tau) / np.sqrt(2 * np.pi) * quad(integrand, -np.inf, np.inf, limit=200)[0]


def implied_vol(price, S, K, T, r, call=True, lo=1e-6, hi=5.0, n_iter=100):
    """Vectorised bisection for the Black-Scholes implied volatility.

    The BS price is strictly increasing in sigma, so bisection on [lo, hi] always converges
    (it plays the role of the bracketing root finder used in the tutorial)."""
    price = np.asarray(price, dtype=float)
    K = np.broadcast_to(np.asarray(K, dtype=float), price.shape)
    lo = np.full(price.shape, lo)
    hi = np.full(price.shape, hi)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        too_high = bs_price(S, K, T, r, mid, call) > price
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    iv = 0.5 * (lo + hi)
    # prices outside the no-arbitrage bounds have no implied volatility
    intrinsic = np.maximum(S - K * np.exp(-r * T), 0) if call else np.maximum(K * np.exp(-r * T) - S, 0)
    return np.where(price <= intrinsic + 1e-10, np.nan, iv)


def down_and_out_call(S0, K, B, T, r, sigma):
    """Closed-form continuously monitored down-and-out call (B < K)."""
    d1, d2 = d1_d2(S0, K, T, r, sigma)
    m1 = (np.log(B**2 / (K * S0)) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    m2 = m1 - sigma * np.sqrt(T)
    return (S0 * (norm.cdf(d1) - (B / S0) ** (1 + 2 * r / sigma**2) * norm.cdf(m1))
            - K * np.exp(-r * T) * (norm.cdf(d2) - (B / S0) ** (2 * r / sigma**2 - 1) * norm.cdf(m2)))


def chooser_price(S0, K, T, t_star, r, sigma):
    """Simple chooser = call(K, T) + put(K e^{-r(T - t*)}, t*)."""
    return bs_price(S0, K, T, r, sigma, call=True) + bs_price(S0, K * np.exp(-r * (T - t_star)), t_star, r, sigma, call=False)


def perpetual_put(S, K, r, sigma):
    """Perpetual American put: exercise boundary x* and value function."""
    x_star = 2 * K * r / (2 * r + sigma**2)
    S = np.asarray(S, dtype=float)
    cont = (K - x_star) * (S / x_star) ** (-2 * r / sigma**2)
    return np.where(S <= x_star, K - S, cont), x_star
