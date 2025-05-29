# ---------------------------- ecosim.py ----------------------------
"""
Ecosystem-Dominant Logic simulation engine
=========================================

**NEW (v1.2) — Orchestrant-enabled productivity**

  • Optional coupling of orchestrant stock *R_t* to both operant and
    operand resource contributions.

  • Flags:
        orchestrant_boost_operant : bool  (default False)
        orchestrant_boost_operand : bool  (default False)

  • Strength parameters:
        alpha  – boost multiplier on operant  (default 0.10)
        beta   – boost multiplier on operand  (default 0.05)

    If enabled:
        operant_term  = w_P * P * (1 + alpha * R_t)
        operand_term  = w_O * O * (1 + beta  * R_t)

    Otherwise falls back to the linear terms used in prior versions.

The remainder of the API is unchanged, so existing field-experiment
scripts and plotting notebooks will continue to run. Passing the new
flags simply activates the productivity effect.

Dependencies: numpy, pandas, networkx
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx

# ------------------ defaults -------------------------------------------------
_DEFAULTS = dict(
    w_O=1.0, w_P=1.0,
    eta=0.3,
    theta=0.1,
    rho=0.8,
    phi=0.2, psi=0.3,
    lam=0.1,
    alpha_val=0.5, beta_val=0.3, gamma=0.2,   # value-decomp weights
    epsilon=0.4,
    sigma=0.4,
    # NEW orchestrant-productivity controls
    orchestrant_boost_operant=False,
    orchestrant_boost_operand=False,
    alpha=0.10,          # strength of boost on operant term
    beta=0.05            # strength of boost on operand term
)

# ------------------ helpers --------------------------------------------------
def _safe_rng(seed_or_rng=None):
    """Return a NumPy RandomState from int, RandomState, or None."""
    if isinstance(seed_or_rng, np.random.RandomState):
        return seed_or_rng
    if seed_or_rng is None:
        return np.random.RandomState()
    if isinstance(seed_or_rng, (int, np.integer)):
        return np.random.RandomState(int(seed_or_rng))
    raise TypeError("Seed must be int, RandomState, or None.")

def initialize_network(n_actors: int, density: float = 0.05, *, seed=None):
    """Watts-Strogatz small-world graph with [0.5,1] proximity weights."""
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

def initialize_resources(G: nx.Graph, *, rng=None):
    """Seed operand (O), unique operant (P_u), shared P_s, and R=0."""
    rng = _safe_rng(rng)
    P_u = rng.normal(1, 0.4, size=len(G))
    P_shared = P_u.mean()
    for idx, n in enumerate(G.nodes):
        G.nodes[n]["O"] = rng.normal(1, 0.2)
        G.nodes[n]["P_u"] = P_u[idx]
        G.nodes[n]["P_s"] = P_shared
        G.nodes[n]["R"] = 0.0

def update_operant_resources(G: nx.Graph, *, epsilon: float):
    """Blend unique and shared operant resources via elasticity ε."""
    for n in G.nodes:
        G.nodes[n]["P"] = (1 - epsilon) * G.nodes[n]["P_u"] + epsilon * G.nodes[n]["P_s"]

def compute_externalities(G: nx.Graph, y_vec: np.ndarray, *, eta: float, sigma: float):
    """e_ij = η · y_i · exp( − (1 − w_ij)/σ )."""
    w = nx.to_numpy_array(G, weight="weight")
    decay = np.exp(-(1 - w) / sigma)
    return eta * y_vec[:, None] * decay

def update_orchestrant_resources(R_prev: float, E_t: float, *, rho: float):
    """R_t = ρ R_{t−1} + (1−ρ) E_t"""
    return rho * R_prev + (1 - rho) * E_t

def compute_value(y_tilde: np.ndarray, e_scaled: np.ndarray, R_t: float, *, 
                  alpha_val, beta_val, gamma, lam):
    """Return V and CA vectors (actor-level)."""
    direct = alpha_val * y_tilde
    relational = beta_val * e_scaled.sum(axis=0)
    orchestrant = gamma * R_t
    V = direct + relational + orchestrant + lam * R_t
    return V, V - V.mean()

# ------------------ main wrapper --------------------------------------------
def run_simulation(n_actors: int = 500, n_steps: int = 500, density: float = 0.05,
                   seed: int = 42, **params) -> pd.DataFrame:
    """
    Simulate the EDL ecosystem.

    New options
    -----------
    orchestrant_boost_operant : bool
        If True, operant productivity scales with `(1 + alpha * R_t)`.
    orchestrant_boost_operand : bool
        If True, operand productivity scales with `(1 + beta  * R_t)`.
    alpha : float
        Strength of orchestrant boost on operant term.
    beta  : float
        Strength of orchestrant boost on operand term.
    """
    p = {**_DEFAULTS, **params}          # merge defaults with overrides
    rng = _safe_rng(seed)

    # initialise
    G = initialize_network(n_actors, density, seed=rng)
    initialize_resources(G, rng=rng)
    S = nx.to_numpy_array(G, weight="weight")

    R_t = 0.0
    y_prev = np.zeros(n_actors)
    logs = []

    for t in range(n_steps):
        update_operant_resources(G, epsilon=p["epsilon"])
        O = np.array([G.nodes[n]["O"] for n in G.nodes])
        P = np.array([G.nodes[n]["P"] for n in G.nodes])

        # ---------- orchestrant-enabled productivity -----------------------
        operand_term = p["w_O"] * O
        operant_term = p["w_P"] * P
        if p["orchestrant_boost_operand"]:
            operand_term *= (1.0 + p["beta"] * R_t)
        if p["orchestrant_boost_operant"]:
            operant_term *= (1.0 + p["alpha"] * R_t)

        # core production and neighbour influence
        y_core = operand_term + operant_term
        y = y_core + p["theta"] * S.dot(y_prev)

        # externalities & orchestrant update
        e = compute_externalities(G, y, eta=p["eta"], sigma=p["sigma"])
        E_t = e.sum()
        R_t = update_orchestrant_resources(R_t, E_t, rho=p["rho"])

        # orchestrant scaling
        y_tilde = (1 + p["phi"] * R_t) * y
        e_scaled = (1 + p["psi"] * R_t) * e

        # value & advantage
        V, CA = compute_value(
            y_tilde, e_scaled, R_t,
            alpha_val=p["alpha_val"], beta_val=p["beta_val"],
            gamma=p["gamma"], lam=p["lam"]
        )

        # ------------------- logging --------------------------------------
        for idx, n in enumerate(G.nodes):
            logs.append(dict(
                time=t, actor_id=n,
                operand_value=operand_term[idx],
                operant_value=operant_term[idx],
                orchestrant_value=p["gamma"] * R_t,
                total_value=V[idx],
                competitive_advantage=CA[idx]
            ))
        y_prev = y

    return pd.DataFrame(logs)

# ------------------ smoke-test ---------------------------------------------
if __name__ == "__main__":
    df_demo = run_simulation(
        n_actors=10, n_steps=20, density=0.2, seed=1,
        orchestrant_boost_operant=True,
        orchestrant_boost_operand=True
    )
    print(df_demo.head())
    print("Shape:", df_demo.shape)
