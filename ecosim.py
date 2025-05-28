# ---------------------------- ecosim.py (1/2) ----------------------------
"""
Ecosystem-Dominant Logic simulation engine
==========================================

Dependencies: numpy, pandas, networkx
Author: ChatGPT
Date: 2025-05-28

Core idea
---------
Simulate how operand, operant and orchestrant resources co-evolve on a social
network of actors.  Outputs a tidy pandas DataFrame for plotting or Dash/Voilà
dashboards.

Public API
----------
initialize_network          – build weighted graph
initialize_resources        – seed operand / operant / orchestrant stocks
update_operant_resources    – elasticity / AI diffusion
compute_externalities       – linear Gaussian spill-over kernel
update_orchestrant_resources– recursive accumulation (memory ρ)
compute_value               – direct + relational + orchestrant value
run_simulation              – wrapper that returns a tidy DataFrame
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx

# ------------------ defaults -------------------------------------------------
_DEFAULTS = dict(
    w_O=1.0, w_P=1.0,           # weight on operand vs operant in production
    eta=0.3,                    # externality intensity
    theta=0.1,                  # neighbour influence on production
    rho=0.8,                    # memory in R recursion
    phi=0.2, psi=0.3,           # R-to-output / R-to-externality amplifiers
    lam=0.1,                    # belonging dividend λ
    alpha=0.5, beta=0.3, gamma=0.2,  # value decomposition weights
    epsilon=0.4,                # elasticity / AI diffusion (0…1)
    sigma=0.4                   # σ in Gaussian attenuation
)

# ------------------ helpers --------------------------------------------------
def _safe_rng(seed: int | None):
    "Return a RandomState (legacy API to stay with NumPy<2)"
    return np.random.RandomState(seed) if seed is not None else np.random

def initialize_network(n_actors: int, density: float = 0.05, *,
                       seed: int | None = None) -> nx.Graph:
    """
    Watts-Strogatz small-world graph; edge weights ∈ (0,1] proportional to
    relational proximity.  Raises ValueError if graph is disconnected.
    """
    if not (0 < density < 1):
        raise ValueError("density must be in (0,1)")
    rng = _safe_rng(seed)
    k = max(1, int(density * n_actors))           # mean degree
    G = nx.watts_strogatz_graph(n_actors, k, p=0.1, seed=seed)
    if not nx.is_connected(G):
        raise ValueError("Generated network is disconnected – try higher density")
    # assign weights (distance → proximity)
    for u, v in G.edges:
        G.edges[u, v]['weight'] = rng.uniform(0.5, 1.0)
    return G

def initialize_resources(G: nx.Graph, *, rng=None):
    """
    Adds node attributes:
        O  – operand   drawn N(1,0.2)
        P_u– unique operant N(1,0.4)
        P_s– shared operant (mean of P_u)
        R  – orchestrant (scalar, start at 0)
    """
    rng = _safe_rng(rng)
    P_u = rng.normal(1, 0.4, size=len(G))
    P_shared = P_u.mean()
    for i, n in enumerate(G.nodes):
        G.nodes[n]['O'] = rng.normal(1, 0.2)
        G.nodes[n]['P_u'] = P_u[i]
        G.nodes[n]['P_s'] = P_shared
        G.nodes[n]['R'] = 0.0

# ------------------ core dynamics -------------------------------------------
def update_operant_resources(G: nx.Graph, *, epsilon: float):
    """
    Blend unique and shared operant resources using elasticity ε.
    """
    for n in G.nodes:
        P_u, P_s = G.nodes[n]['P_u'], G.nodes[n]['P_s']
        G.nodes[n]['P'] = (1 - epsilon) * P_u + epsilon * P_s

def compute_externalities(G: nx.Graph, y_vec: np.ndarray, *, eta: float, sigma: float):
    """
    Returns NxN matrix e_ij = η y_i exp( - (1-w_ij)/σ ).
    """
    N = len(G)
    w = nx.to_numpy_array(G, weight='weight')      # proximity matrix
    decay = np.exp(-(1 - w) / sigma)
    return eta * y_vec[:, None] * decay

def update_orchestrant_resources(R_prev: float, E_t: float, *, rho: float):
    "Recursive: R_t = ρ R_{t-1} + (1-ρ) E_t"
    return rho * R_prev + (1 - rho) * E_t
# ---------------------------- ecosim.py (2/2) ----------------------------
def compute_value(G: nx.Graph, y_tilde: np.ndarray, e_scaled: np.ndarray,
                  R_t: float, *, alpha, beta, gamma, lam):
    """
    Returns tuple (V_vec, CA_vec) – value and competitive advantage per actor.
    """
    N = len(G)
    direct = alpha * y_tilde
    relational = beta * e_scaled.sum(axis=0)
    orchestrant = gamma * R_t
    V = direct + relational + orchestrant + lam * R_t
    CA = V - V.mean()
    return V, CA

# ------------------ high-level wrapper --------------------------------------
def run_simulation(n_actors: int = 500, n_steps: int = 500, density: float = 0.05,
                   seed: int = 42, **params) -> pd.DataFrame:
    """
    Simulates the ecosystem and returns a tidy DataFrame with per-actor,
    per-period metrics.  Missing params default to _DEFAULTS.
    """
    p = _DEFAULTS | params                         # merge dicts (py3.9+)
    rng = _safe_rng(seed)

    # 1. network + initial stocks ------------------------------------------------
    G = initialize_network(n_actors, density, seed=seed)
    initialize_resources(G, rng=rng)

    # Allocate logs
    logs = []

    # Precompute adjacency weight matrix & neighbour index list
    S = nx.to_numpy_array(G, weight='weight')

    # 2. main loop --------------------------------------------------------------
    R_t = 0.0
    y_prev = np.zeros(n_actors)

    for t in range(n_steps):
        update_operant_resources(G, epsilon=p['epsilon'])

        # build current resource vectors
        O = np.array([G.nodes[n]['O'] for n in G.nodes])
        P = np.array([G.nodes[n]['P'] for n in G.nodes])

        # direct output: f_O + f_P + θ S y_prev
        y_core = p['w_O'] * O + p['w_P'] * P
        y = y_core + p['theta'] * S.dot(y_prev)

        # externalities & orchestrant update
        e = compute_externalities(G, y, eta=p['eta'], sigma=p['sigma'])
        E_t = e.sum()
        R_t = update_orchestrant_resources(R_t, E_t, rho=p['rho'])

        # R-scaled outputs and spillovers
        y_tilde = (1 + p['phi'] * R_t) * y
        e_scaled = (1 + p['psi'] * R_t) * e

        # value + advantage
        V, CA = compute_value(G, y_tilde, e_scaled, R_t,
                              alpha=p['alpha'], beta=p['beta'],
                              gamma=p['gamma'], lam=p['lam'])

        # ---- log everything ---------------------------------------------------
        for idx, n in enumerate(G.nodes):
            logs.append(dict(time=t, actor_id=n,
                             operand_value=p['w_O'] * O[idx],
                             operant_value=p['w_P'] * P[idx],
                             orchestrant_value=p['gamma'] * R_t,
                             total_value=V[idx],
                             competitive_advantage=CA[idx]))

        # update state
        y_prev = y

    df = pd.DataFrame(logs)
    return df

# ------------------ quick CLI smoke-test ------------------------------------
if __name__ == "__main__":
    demo = run_simulation(n_actors=10, n_steps=20, density=0.2, seed=1)
    print(demo.head())
    print("\nShape:", demo.shape)
