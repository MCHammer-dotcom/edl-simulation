# ---------------------------- ecosim.py ----------------------------
"""
Ecosystem-Dominant Logic simulation engine
==========================================

Dependencies: numpy, pandas, networkx
Author: ChatGPT
Date: 2025-05-28  (patched: safe RNG handling)

Core idea
---------
Simulate how operand, operant and orchestrant resources co-evolve on a social
network of actors.  Outputs a tidy pandas DataFrame for plotting or dashboards.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx

# ------------------ defaults -------------------------------------------------
_DEFAULTS = dict(
    w_O=1.0, w_P=1.0,           # weights in production
    eta=0.3,                    # externality intensity
    theta=0.1,                  # neighbour influence on production
    rho=0.8,                    # memory in R recursion
    phi=0.2, psi=0.3,           # R amplifiers
    lam=0.1,                    # belonging dividend λ
    alpha=0.5, beta=0.3, gamma=0.2,  # value weights
    epsilon=0.4,                # elasticity / AI diffusion
    sigma=0.4                   # Gaussian attenuation width
)

# ------------------ helpers --------------------------------------------------
def _safe_rng(seed_or_rng: int | np.random.RandomState | None = None) -> np.random.RandomState:
    """
    Always return a NumPy RandomState.
      • int  -> RandomState(seed)
      • RandomState -> unchanged
      • None -> fresh, unseeded RandomState
    """
    if isinstance(seed_or_rng, np.random.RandomState):
        return seed_or_rng
    if seed_or_rng is None:
        return np.random.RandomState()
    if isinstance(seed_or_rng, (int, np.integer)):
        return np.random.RandomState(int(seed_or_rng))
    raise TypeError("Seed must be int, RandomState, or None.")

def initialize_network(n_actors: int, density: float = 0.05, *,
                       seed: int | None = None) -> nx.Graph:
    """
    Watts-Strogatz small-world graph; edge weights ∈ (0,1] proportional to
    relational proximity. Raises ValueError if graph is disconnected.
    """
    if not (0 < density < 1):
        raise ValueError("density must be in (0,1)")
    rng = _safe_rng(seed)
    k = max(1, int(density * n_actors))
    G = nx.watts_strogatz_graph(n_actors, k, p=0.1, seed=rng)
    if not nx.is_connected(G):
        raise ValueError("Generated network is disconnected – raise density")
    for u, v in G.edges:
        G.edges[u, v]["weight"] = rng.uniform(0.5, 1.0)
    return G

def initialize_resources(G: nx.Graph, *, rng: int | np.random.RandomState | None = None):
    """
    Adds node attributes:
        O   – operand resource stock
        P_u – unique operant stock
        P_s – shared operant baseline (common to all)
        R   – orchestrant stock (scalar)
    """
    rng = _safe_rng(rng)
    P_u = rng.normal(1, 0.4, size=len(G))
    P_shared = P_u.mean()
    for idx, n in enumerate(G.nodes):
        G.nodes[n]["O"] = rng.normal(1, 0.2)
        G.nodes[n]["P_u"] = P_u[idx]
        G.nodes[n]["P_s"] = P_shared
        G.nodes[n]["R"] = 0.0

def update_operant_resources(G: nx.Graph, *, epsilon: float):
    "Blend unique and shared operant resources using elasticity ε."
    for n in G.nodes:
        P_u, P_s = G.nodes[n]["P_u"], G.nodes[n]["P_s"]
        G.nodes[n]["P"] = (1 - epsilon) * P_u + epsilon * P_s

def compute_externalities(G: nx.Graph, y_vec: np.ndarray, *,
                          eta: float, sigma: float):
    """Return e_ij matrix with Gaussian attenuation."""
    w = nx.to_numpy_array(G, weight="weight")
    decay = np.exp(-(1 - w) / sigma)
    return eta * y_vec[:, None] * decay

def update_orchestrant_resources(R_prev: float, E_t: float, *, rho: float):
    "Recursive stock update R_t = ρ R_{t-1} + (1-ρ) E_t"
    return rho * R_prev + (1 - rho) * E_t

def compute_value(y_tilde: np.ndarray, e_scaled: np.ndarray, R_t: float, *,
                  alpha, beta, gamma, lam):
    """Return V and CA vectors."""
    direct = alpha * y_tilde
    relational = beta * e_scaled.sum(axis=0)
    orchestrant = gamma * R_t
    V = direct + relational + orchestrant + lam * R_t
    CA = V - V.mean()
    return V, CA

# ------------------ wrapper --------------------------------------------------
def run_simulation(n_actors: int = 500, n_steps: int = 500, density: float = 0.05,
                   seed: int = 42, **params) -> pd.DataFrame:
    """
    Simulate ecosystem dynamics; return tidy DataFrame with per-actor metrics.
    """
    p = _DEFAULTS | params
    rng = _safe_rng(seed)

    # 1. network & stocks
    G = initialize_network(n_actors, density, seed=rng)
    initialize_resources(G, rng=rng)
    S = nx.to_numpy_array(G, weight="weight")

    R_t = 0.0
    y_prev = np.zeros(n_actors)
    logs = []

    # 2. main loop
    for t in range(n_steps):
        update_operant_resources(G, epsilon=p["epsilon"])
        O = np.array([G.nodes[n]["O"] for n in G.nodes])
        P = np.array([G.nodes[n]["P"] for n in G.nodes])

        y_core = p["w_O"] * O + p["w_P"] * P
        y = y_core + p["theta"] * S.dot(y_prev)

        e = compute_externalities(G, y, eta=p["eta"], sigma=p["sigma"])
        E_t = e.sum()
        R_t = update_orchestrant_resources(R_t, E_t, rho=p["rho"])

        y_tilde = (1 + p["phi"] * R_t) * y
        e_scaled = (1 + p["psi"] * R_t) * e

        V, CA = compute_value(y_tilde, e_scaled, R_t,
                              alpha=p["alpha"], beta=p["beta"],
                              gamma=p["gamma"], lam=p["lam"])

        for idx, n in enumerate(G.nodes):
            logs.append(dict(time=t, actor_id=n,
                             operand_value=p["w_O"] * O[idx],
                             operant_value=p["w_P"] * P[idx],
                             orchestrant_value=p["gamma"] * R_t,
                             total_value=V[idx],
                             competitive_advantage=CA[idx]))
        y_prev = y

    return pd.DataFrame(logs)

# ------------------ CLI smoke-test ------------------------------------------
if __name__ == "__main__":
    demo = run_simulation(n_actors=10, n_steps=20, density=0.2, seed=1)
    print(demo.head(), "\nShape:", demo.shape)
