"""Heston model: d gamma = (kappa - lambda gamma) dt + sigma_tilde sqrt(gamma) dW  (lecture notation, Section 2.5)."""
import numpy as np
from scipy.integrate import quad, simpson

from util.black_scholes import implied_vol


def heston_char(u, S0, T, r, gam0, kappa, lamb, sig_tilde, rho):
    """Characteristic function chi(u) = E_Q[exp(iu log S(T))] (numerically stable form from the tutorial)."""
    B = rho * sig_tilde * u * 1j - lamb
    D = np.sqrt(B**2 + sig_tilde**2 * u * (u + 1j))
    eDT = np.exp(-D * T)
    lnG = np.log(B * (eDT - 1) / (2 * D) + (eDT + 1) / 2)
    psi0 = u * 1j * r * T - kappa / sig_tilde**2 * ((B + D) * T + 2 * lnG)
    psi1 = u * 1j
    psi2 = u * (u + 1j) * (eDT - 1) / (B * (eDT - 1) + D * (eDT + 1))
    return np.exp(psi0 + psi1 * np.log(S0) + psi2 * gam0)


def bs_char(u, S0, T, r, sigma):
    """Characteristic function of log S(T) in the Black-Scholes model."""
    m = np.log(S0) + (r - 0.5 * sigma**2) * T
    return np.exp(1j * u * m - 0.5 * sigma**2 * T * u**2)


def gil_pelaez_density(x_grid, char, u_max=50.0):
    """f(x) = 1/pi * int_0^inf Re(e^{-iux} chi(u)) du."""
    return np.array([quad(lambda u: np.real(np.exp(-1j * u * x) * char(u)), 0, u_max, limit=200)[0] / np.pi
                     for x in x_grid])


def price_from_density(x_grid, dens, g, r, T):
    """V(0) = e^{-rT} int g(e^x) f(x) dx via Simpson's rule on the sampled density."""
    return np.exp(-r * T) * simpson(g(np.exp(x_grid)) * dens, x=x_grid)


# ── Laplace transform pricing (Section 3.1) ─────────────────────────────
def f_tilde_call(z, K):
    """Bilateral Laplace transform of (e^x - K)^+, valid for Re(z) = R > 1.

    The put (K - e^x)^+ has the same expression, but on the strip Re(z) = R < 0."""
    return K ** (1 - z) / (z * (z - 1))


def laplace_price(K, char, r, T, R=1.5, u_max=100.0, f_tilde=f_tilde_call):
    """V(0) = e^{-rT}/pi * int_0^inf Re( f~(R + iu) chi(u - iR) ) du."""
    val = quad(lambda u: np.real(f_tilde(R + 1j * u, K) * char(u - 1j * R)), 0, u_max, limit=400)[0]
    return np.exp(-r * T) / np.pi * val


def laplace_delta(K, char, S0, r, T, R=1.5, u_max=100.0):
    """d/dS0 of the Laplace price: d chi(u - iR)/dS0 = (iu + R)/S0 * chi(u - iR)."""
    def integrand(u):
        z = R + 1j * u
        return np.real(f_tilde_call(z, K) * z / S0 * char(u - 1j * R))
    return np.exp(-r * T) / np.pi * quad(integrand, 0, u_max, limit=400)[0]


# ── FFT pricing (Section 3.2) ───────────────────────────────────────────
def fft_call_prices(K_array, char, r, T, R=1.5, M=5000.0, N=2**15):
    """Call prices for all strikes at once via the FFT algorithm of Section 3.2."""
    K_array = np.atleast_1d(np.asarray(K_array, dtype=float))
    delta = M / N
    n = np.arange(1, N + 1)
    d_kappa = 2 * np.pi / M
    kappa_1 = np.log(K_array.min())
    kappa_m = kappa_1 + (n - 1) * d_kappa

    def g(u):
        z = R + 1j * u
        return 1 / (z * (z - 1)) * char(u - 1j * R)

    x = g((n - 0.5) * delta) * delta * np.exp(-1j * (n - 1) * delta * kappa_1)
    x_hat = np.fft.fft(x)
    V_kappa = np.exp(-r * T + (1 - R) * kappa_m) / np.pi * np.real(x_hat * np.exp(-1j * delta / 2 * kappa_m))
    return np.interp(K_array, np.exp(kappa_m), V_kappa), np.exp(kappa_m), V_kappa


def heston_fft_prices(S0, K, T, r, gam0, kappa, lamb, sig_tilde, rho, R=1.5, M=5000.0, N=2**15):
    char = lambda u: heston_char(u, S0, T, r, gam0, kappa, lamb, sig_tilde, rho)
    return fft_call_prices(K, char, r, T, R, M, N)[0]


def heston_implied_vols(S0, K, T, r, params, N=2**15):
    prices = heston_fft_prices(S0, K, T, r, *params, N=N)
    return implied_vol(prices, S0, K, T, r, call=True)


# ── Simulation ──────────────────────────────────────────────────────────
def simulate_variance_exact(gam0, kappa, lamb, sig_tilde, T, m, N, rng):
    """Bias-free simulation: gamma(t+dt) | gamma(t) ~ Y / (2c), Y non-central chi^2."""
    dt = T / m
    c = 2 * lamb / ((1 - np.exp(-lamb * dt)) * sig_tilde**2)
    df = 4 * kappa / sig_tilde**2
    gamma = np.empty((N, m + 1))
    gamma[:, 0] = gam0
    for i in range(1, m + 1):
        nc = 2 * c * np.exp(-lamb * dt) * gamma[:, i - 1]
        gamma[:, i] = rng.noncentral_chisquare(df, nc) / (2 * c)
    return gamma


def simulate_variance_euler(gam0, kappa, lamb, sig_tilde, T, m, N, rng, fix="none"):
    """Plain Euler scheme for the variance; can go negative (fix='none') unless truncated."""
    dt = T / m
    gamma = np.empty((N, m + 1))
    gamma[:, 0] = gam0
    for i in range(1, m + 1):
        g_prev = gamma[:, i - 1]
        g_pos = np.maximum(g_prev, 0.0)
        gamma[:, i] = g_prev + (kappa - lamb * (g_pos if fix == "full truncation" else g_prev)) * dt \
            + sig_tilde * np.sqrt(g_pos * dt) * rng.standard_normal(N)
        if fix == "reflection":
            gamma[:, i] = np.abs(gamma[:, i])
    return gamma


def simulate_heston_paths(S0, T, r, gam0, kappa, lamb, sig_tilde, rho, m, N, rng):
    """Log-Euler for S with full-truncation Euler for the variance."""
    dt = T / m
    S = np.empty((N, m + 1))
    gamma = np.empty((N, m + 1))
    S[:, 0], gamma[:, 0] = S0, gam0
    for i in range(1, m + 1):
        z1 = rng.standard_normal(N)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(N)
        g = np.maximum(gamma[:, i - 1], 0.0)
        S[:, i] = S[:, i - 1] * np.exp((r - 0.5 * g) * dt + np.sqrt(g * dt) * z1)
        gamma[:, i] = gamma[:, i - 1] + (kappa - lamb * g) * dt + sig_tilde * np.sqrt(g * dt) * z2
    return S, np.maximum(gamma, 0.0)
