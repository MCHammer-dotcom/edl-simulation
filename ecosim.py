# ---------------------------- ecosim.py ----------------------------
"""
Ecosystem-Dominant Logic simulation engine      ·      v1.4
===============================================================

New in 1.4
----------
Actor heterogeneity, negative externalities, adaptive orchestrant
memory, tipping-point penalties, and optional diagnostics — while
retaining full backward compatibility.

Actor roles (assigned on init)
    • Generator  – 60 %   (operant output ×1.2)
    • Absorber   – 30 %   (operant output ×0.7)
    • Suppressor – 10 %   (operant ×1.0, but sends negative externalities)

Key options (all False by default to preserve prior behaviour)
    enable_actor_types          – activate role logic
    enable_negative_externality – flip up to 20 % of edges negative
    enable_volatility_memory    – adapt ρ_t to rolling inflow volatility
    enable_tipping_penalty      – apply −20 % to next y_core if V_i > tipping_thr
    enable_diagnostics          – add extra fields & optional printouts

Previous features
    • Orchestrant-boosted productivity (bounded formula)
    • R_cap ceiling, bounded boosts, debug printing

API compatibility: older experiment scripts need no change.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
from collections import deque

# ------------------ defaults -------------------------------------------------
_DEFAULTS = dict(
    # structural + externality params
    w_O=1.0, w_P=1.0, eta=0.3, theta=0.1,
    # orchestrant dynamics
    rho_base=0.8,           # baseline memory
    phi=0.2, psi=0.3, lam=0.1, gamma=0.2,
    alpha_val=0.5, beta_val=0.3,               # value decomposition
    epsilon=0.4, sigma=0.4,
    # boosts
    orchestrant_boost_operant=False,
    orchestrant_boost_operand=False,
    alpha=0.05, beta=0.025,
    R_cap=None,
    print_growth_debug=False,
    # new behavioural toggles
    enable_actor_types=False,
    enable_negative_externality=False,
    enable_volatility_memory=False,
    enable_tipping_penalty=False,
    enable_diagnostics=False,
    tipping_thr=10.0,
)

# ------------------ helpers --------------------------------------------------
def _safe_rng(seed_or_rng=None):
    if isinstance(seed_or_rng, np.random.RandomState):
        return seed_or_rng
    return np.random.RandomState(seed_or_rng)

# ---------- network + resource initialisation -------------------------------
def initialize_network(n: int, density: float, *, rng):
    k = max(1, int(density * n))
    G = nx.watts_strogatz_graph(n, k, p=0.1, seed=rng)
    if not nx.is_connected(G):
        raise ValueError("Network disconnected; raise density.")
    for u, v in G.edges:
        G.edges[u, v]['weight'] = rng.uniform(0.5, 1.0)
    return G

def assign_actor_types(nodes, *, rng):
    n = len(nodes)
    roles = (["Generator"]  * int(0.6*n) +
             ["Absorber"]   * int(0.3*n) +
             ["Suppressor"] * (n - int(0.9*n)))
    rng.shuffle(roles)
    return dict(zip(nodes, roles))

def initialize_resources(G: nx.Graph, *, rng, enable_actor_types: bool):
    rng = _safe_rng(rng)
    types = assign_actor_types(G.nodes, rng=rng) if enable_actor_types else {}
    P_u = rng.normal(1, 0.4, size=len(G))
    P_shared = P_u.mean()
    for idx, n in enumerate(G.nodes):
        G.nodes[n]['O'] = rng.normal(1, 0.2)
        G.nodes[n]['P_u'] = P_u[idx]
        G.nodes[n]['P_s'] = P_shared
        G.nodes[n]['R'] = 0.0
        if enable_actor_types:
            G.nodes[n]['type'] = types[n]

# ---------- main simulation --------------------------------------------------
def run_simulation(n_actors: int = 500, n_steps: int = 300, density: float = 0.05,
                   seed: int | None = 42, **opts) -> pd.DataFrame:
    p = {**_DEFAULTS, **opts}
    rng = _safe_rng(seed)

    # network and resources
    G = initialize_network(n_actors, density, rng=rng)
    initialize_resources(G, rng=rng, enable_actor_types=p['enable_actor_types'])
    S = nx.to_numpy_array(G, weight='weight')

    # optionally mark some negative edges (except suppressor rule handled later)
    if p['enable_negative_externality']:
        neg_mask = rng.rand(*S.shape) < 0.2
        S = np.where(neg_mask, -S, S)

    R_t, y_prev = 0.0, np.zeros(n_actors)
    inflow_window: deque[float] = deque(maxlen=5)
    logs = []
    actor_types = np.array([G.nodes[n].get('type', 'None') for n in G.nodes])

    for t in range(n_steps):
        # elasticity blending
        for n in G.nodes:
            G.nodes[n]['P'] = (1-p['epsilon'])*G.nodes[n]['P_u'] + p['epsilon']*G.nodes[n]['P_s']
        O = np.array([G.nodes[n]['O'] for n in G.nodes])
        P_vec = np.array([G.nodes[n]['P'] for n in G.nodes])

        # role-based output multiplier
        operant_multiplier = np.ones(n_actors)
        if p['enable_actor_types']:
            operant_multiplier[actor_types == 'Generator']  = 1.2
            operant_multiplier[actor_types == 'Absorber']   = 0.7
            # Suppressor stays 1.0

        # bounded boosts
        boost_operant = 1 + (p['alpha']*R_t) / (1 + p['alpha']*R_t) \
                        if p['orchestrant_boost_operant'] else 1.0
        boost_operand = 1 + (p['beta'] *R_t) / (1 + p['beta'] *R_t) \
                        if p['orchestrant_boost_operand'] else 1.0

        operand_term = p['w_O'] * O * boost_operand
        operant_term = p['w_P'] * P_vec * boost_operant * operant_multiplier

        # tipping-point penalty applied to next core output
        if p['enable_tipping_penalty'] and t > 0:
            penal_mask = prev_total > p['tipping_thr']
            operand_term[penal_mask] *= 0.8
            operant_term[penal_mask] *= 0.8

        y_core = operand_term + operant_term
        y = y_core + p['theta'] * S.dot(y_prev)

        # compute externalities with sign adjustments for suppressors
        e = p['eta'] * y[:, None] * np.exp(-(1 - np.abs(S)) / p['sigma']) * np.sign(S)
        if p['enable_actor_types']:
            sup_idx = np.where(actor_types == 'Suppressor')[0]
            # suppressor sends negative stronger spillovers
            e[sup_idx, :] *= -1.5
        E_t = e.sum()
        if p['enable_diagnostics']:
            ext_received = e.sum(axis=0)

        # orchestrant memory adaptation
        inflow_window.append(E_t)
        if p['enable_volatility_memory'] and len(inflow_window) == inflow_window.maxlen:
            vol = np.std(inflow_window)
            rho_t = max(0.5, min(0.95, p['rho_base'] * (1 - vol / (abs(E_t)+1e-6))))
        else:
            vol, rho_t = 0.0, p['rho_base']

        R_t = rho_t * R_t + (1 - rho_t) * E_t
        if p['R_cap'] is not None:
            R_t = min(R_t, p['R_cap'])

        y_tilde  = (1 + p['phi'] * R_t) * y
        e_scaled = (1 + p['psi'] * R_t) * e

        V, CA = _compute_value(
            y_tilde, e_scaled, R_t, p['alpha_val'], p['beta_val'], p['gamma'], p['lam']
        )

        if p['print_growth_debug']:
            print(f"t={t:3d} R={R_t:.2f}  boost_op={boost_operant:.2f}  "
                  f"boost_ov={boost_operand:.2f}")

        # ------- logging ----------------------------------------------------
        for idx, n in enumerate(G.nodes):
            record = dict(time=t, actor_id=n,
                          operand_value=operand_term[idx],
                          operant_value=operant_term[idx],
                          orchestrant_value=p['gamma']*R_t,
                          total_value=V[idx],
                          competitive_advantage=CA[idx])
            if p['enable_actor_types']:
                record['actor_type'] = actor_types[idx]
            if p['enable_diagnostics']:
                record['externality_received'] = ext_received[idx]
                record['volatility'] = vol
                record['rho_t'] = rho_t
            logs.append(record)

        y_prev, prev_total = y, V

    return pd.DataFrame(logs)

# ---------- helper ----------------------------------------------------------
def _compute_value(y_til, e_scaled, R, a, b, gam, lam):
    direct = a * y_til
    rel    = b * e_scaled.sum(axis=0)
    V = direct + rel + gam * R + lam * R
    return V, V - V.mean()

# ----------------- smoke-test (comment out in production) -------------------
if __name__ == "__main__":
    df = run_simulation(n_actors=20, n_steps=30, density=0.1, seed=0,
                        orchestrant_boost_operant=True,
                        orchestrant_boost_operand=True,
                        enable_actor_types=True,
                        enable_negative_externality=True,
                        enable_volatility_memory=True,
                        enable_tipping_penalty=True,
                        enable_diagnostics=True,
                        R_cap=10.0,
                        print_growth_debug=False)
    print(df.head())
