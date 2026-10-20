import numpy as np


def crr_params(r, sigma, dt, c=1.0):
    """u, d, q from moment matching (Section 1.1); c = 1 is CRR, c != 1 the anchored tree (ud = c)."""
    beta = 0.5 * (c * np.exp(-r * dt) + np.exp((r + sigma**2) * dt))
    u = beta + np.sqrt(beta**2 - c)
    d = c / u
    q = (np.exp(r * dt) - d) / (u - d)
    return u, d, q


def payoff_fn(kind, K, power=1.0):
    if kind == "call":
        return lambda s: np.maximum(s**power - K, 0.0)
    if kind == "put":
        return lambda s: np.maximum(K - s**power, 0.0)
    if kind == "digital":
        return lambda s: (s >= K).astype(float)
    raise ValueError(kind)


def crr_tree(S0, T, r, sigma, M, g, american=False, barrier=None, c=1.0):
    """Full (M+1) x (M+1) stock and option matrices, S[j, i] = S0 u^j d^(i-j) (NaN above diagonal)."""
    dt = T / M
    u, d, q = crr_params(r, sigma, dt, c)
    disc = np.exp(-r * dt)
    S = np.full((M + 1, M + 1), np.nan)
    V = np.full((M + 1, M + 1), np.nan)
    for i in range(M + 1):
        j = np.arange(i + 1)
        S[: i + 1, i] = S0 * np.exp(j * np.log(u) + (i - j) * np.log(d))
    V[:, M] = g(S[:, M])
    if barrier is not None:
        V[S[:, M] < barrier, M] = 0.0
    for i in range(M - 1, -1, -1):
        cont = disc * (q * V[1 : i + 2, i + 1] + (1 - q) * V[: i + 1, i + 1])
        if american:
            cont = np.maximum(cont, g(S[: i + 1, i]))
        if barrier is not None:
            cont[S[: i + 1, i] < barrier] = 0.0
        V[: i + 1, i] = cont
    return S, V, (u, d, q)


def crr_bermudan(S0, T, r, sigma, m_dates, g, steps_per_date=50):
    """Bermudan option: exercise allowed only at t_k = k T / m_dates (k = 1..m_dates), via a CRR tree."""
    M = m_dates * steps_per_date
    dt = T / M
    u, d, q = crr_params(r, sigma, dt)
    disc = np.exp(-r * dt)
    lu, ld = np.log(u), np.log(d)
    j = np.arange(M + 1)
    V = g(S0 * np.exp(j * lu + (M - j) * ld))
    for i in range(M - 1, -1, -1):
        V = disc * (q * V[1:] + (1 - q) * V[:-1])
        if i > 0 and i % steps_per_date == 0:
            jj = j[: i + 1]
            V = np.maximum(V, g(S0 * np.exp(jj * lu + (i - jj) * ld)))
    return V[0]


def crr_price(S0, T, r, sigma, M, g, american=False, c=1.0, barrier=None):
    """Memory-light backward induction (only one column kept) - fast enough for M in the tens of thousands."""
    dt = T / M
    u, d, q = crr_params(r, sigma, dt, c)
    disc = np.exp(-r * dt)
    lu, ld = np.log(u), np.log(d)
    j = np.arange(M + 1)
    S_T = S0 * np.exp(j * lu + (M - j) * ld)
    V = g(S_T)
    if barrier is not None:
        V[S_T < barrier] = 0.0
    for i in range(M - 1, -1, -1):
        V = disc * (q * V[1:] + (1 - q) * V[:-1])
        if american or barrier is not None:
            jj = j[: i + 1]
            S_i = S0 * np.exp(jj * lu + (i - jj) * ld)
            if american:
                V = np.maximum(V, g(S_i))
            if barrier is not None:
                V[S_i < barrier] = 0.0
    return V[0]


def crr_delta_tree(S, V):
    """Hedge ratio phi_1(t_i) = (V_up - V_down) / (S_up - S_down) for every node at t_{i-1} (eq. 1.12)."""
    M = S.shape[1] - 1
    D = np.full((M, M), np.nan)
    for i in range(M):
        D[: i + 1, i] = (V[1 : i + 2, i + 1] - V[: i + 1, i + 1]) / (S[1 : i + 2, i + 1] - S[: i + 1, i + 1])
    return D
