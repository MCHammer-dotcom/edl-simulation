# ---------------------------- ecosim.py ----------------------------
"""
Ecosystem-Dominant Logic simulation engine   ·   v1.3 (bounded boosts)
=====================================================================

NEW IN THIS VERSION
-------------------
• **Saturating orchestrant boosts** – avoids runaway exponential growth:
      boost = 1 + (α·R_t) / (1 + α·R_t)          # asymptote at 2
  The same form is used for both operant (α) and operand (β) terms.

• **Optional R-cap** – safety ceiling for the orchestrant stock:
      R_t = min(R_t, R_cap)            # default: no cap

• **Debug printing** – set `print_growth_debug=True` to emit per-tick
  diagnostics (R_t and boost factors).

The public API is unchanged, so existing experiment scripts still work.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx

# ------------------ defaults -------------------------------------------------
_DEFAULTS = dict(
    # production weights & externalities
    w_O=1.0, w_P=1.0, eta=0.3, theta=0.1,
    # orchestrant dynamics
    rho=0.8, phi=0.2, psi=0.3, lam=0.1,
    gamma=0.2, alpha_val=0.5, beta_val=0.3,
    epsilon=0.4, sigma=0.4,
    # orchestrant-productivity controls
    orchestrant_boost_operant=False,
    orchestrant_boost_operand=False,
    alpha=0.10,          # operant boost strength
    beta=0.05,           # operand boost strength
    R_cap=None,          # set e.g. 10.0 to clip R_t
    print_growth_debug=False
)

# ------------------ helpers --------------------------------------------------
def _safe_rng(seed_or_rng=None):
    if isinstance(seed_or_rng, np.random.RandomState):
        return seed_or_rng
    if seed_or_rng is None:
        return np.random.RandomState()
    return np.random.RandomState(int(seed_or_rng))

def initialize_network(n_actors: int, density: float, *, seed):
    rng = _safe_rng(seed)
    k = max(1, int(density * n_actors))
    G = nx.watts_strogatz_graph(n_actors, k, p=0.1, seed=rng)
    if not nx.is_connected(G):
        raise ValueError("Network disconnected – raise density")
    for u, v in G.edges:
        G.edges[u, v]["weight"] = rng.uniform(0.5, 1.0)
    return G

def initialize_resources(G: nx.Graph, *, rng):
    rng = _safe_rng(rng)
    P_u = rng.normal(1, 0.4, size=len(G))
    P_shared = P_u.mean()
    for idx, n in enumerate(G.nodes):
        G.nodes[n]["O"] = rng.normal(1, 0.2)
        G.nodes[n]["P_u"] = P_u[idx]
        G.nodes[n]["P_s"] = P_shared
        G.nodes[n]["R"] = 0.0

def update_operant_resources(G: nx.Graph, *, epsilon: float):
    for n in G.nodes:
        G.nodes[n]["P"] = (1 - epsilon) * G.nodes[n]["P_u"] + epsilon * G.nodes[n]["P_s"]

def compute_externalities(G: nx.Graph, y: np.ndarray, *, eta: float, sigma: float):
    w = nx.to_numpy_array(G, weight="weight")
    decay = np.exp(-(1 - w) / sigma)
    return eta * y[:, None] * decay

def update_R(R_prev: float, E_t: float, rho: float, cap: float | None):
    R = rho * R_prev + (1 - rho) * E_t
    if cap is not None:
        R = min(R, cap)
    return R

def compute_value(y_tilde, e_scaled, R, *, alpha_val, beta_val, gamma, lam):
    direct = alpha_val * y_tilde
    relational = beta_val * e_scaled.sum(axis=0)
    V = direct + relational + gamma * R + lam * R
    return V, V - V.mean()

# ------------------ simulation wrapper --------------------------------------
def run_simulation(n_actors: int = 500, n_steps: int = 500, density: float = 0.05,
                   seed: int = 42, **kwargs) -> pd.DataFrame:
    """
    Run the EDL simulation.

    Parameters added in v1.3
    ------------------------
    orchestrant_boost_operant : bool, default False
    orchestrant_boost_operand : bool, default False
    alpha : float  – operant boost strength
    beta  : float  – operand boost strength
    R_cap : float | None – upper limit for R_t (None = no cap)
    print_growth_debug : bool – print R_t and boost factors each tick
    """
    p = {**_DEFAULTS, **kwargs}
    rng = _safe_rng(seed)

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

        # ---------- bounded boost factors ----------------------------------
        boost_operant  = 1.0
        boost_operand  = 1.0
        if p["orchestrant_boost_operant"]:
            boost_operant = 1 + (p["alpha"] * R_t) / (1 + p["alpha"] * R_t)
        if p["orchestrant_boost_operand"]:
            boost_operand = 1 + (p["beta"] * R_t)  / (1 + p["beta"]  * R_t)

        operand_term = p["w_O"] * O * boost_operand
        operant_term = p["w_P"] * P * boost_operant

        if p["print_growth_debug"]:
            print(f"t={t:3d}  R={R_t:.3f}  "
                  f"boost_operant={boost_operant:.3f}  "
                  f"boost_operand={boost_operand:.3f}")

        y_core = operand_term + operant_term
        y = y_core + p["theta"] * S.dot(y_prev)

        e = compute_externalities(G, y, eta=p["eta"], sigma=p["sigma"])
        E_t = e.sum()
        R_t = update_R(R_t, E_t, rho=p["rho"], cap=p["R_cap"])

        y_tilde = (1 + p["phi"] * R_t) * y
        e_scaled = (1 + p["psi"] * R_t) * e

        V, CA = compute_value(
            y_tilde, e_scaled, R_t,
            alpha_val=p["alpha_val"], beta_val=p["beta_val"],
            gamma=p["gamma"], lam=p["lam"]
        )

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
    df = run_simulation(
        n_actors=10, n_steps=15, density=0.2, seed=1,
        orchestrant_boost_operant=True,
        orchestrant_boost_operand=True,
        print_growth_debug=True,
        R_cap=10.0
    )
    print(df.head())
