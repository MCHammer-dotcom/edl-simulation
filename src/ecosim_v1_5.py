# ---------------------------- ecosim.py ----------------------------
"""
Ecosystem-Dominant Logic simulation engine               •  v1.5  (May 2025)
====================================================================

Backward-compatible with v1.4  ➜  new behaviour is gated by flags.

NEW OPTIONAL CAPABILITIES
-------------------------
1. Receiver-sensitivity            (enable_receiver_sensitivity)
   • Each actor draws  s_i ~ U[0.8,1.2]  applied to inbound externalities.

2. Externality balance tracking    (enable_externality_balance)
   • Logs  net_ext_i(t) = Σ_j e_ij − Σ_j e_ji.

3. Tipping-recovery mechanism      (enable_tipping_recovery)
   • Boolean state is_tipped  ⇒  −20 % productivity until value < 0.8·thr.

4. Local orchestrant field         (enable_local_orchestrant_field)
   • Maintains vector R_i(t)  (personal stock based on inflow);
     global scalar R̄_t for backward-compatibility.

Diagnostics (all in output when enable_diagnostics=True)
    receiver_sensitivity, externality_received, net_ext,
    volatility, rho_t, is_tipped
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
from collections import deque

# ------------------------------------------------------------------ defaults
_DEFAULTS = dict(
    # core params (as in v1.4)
    w_O=1.0, w_P=1.0, eta=0.3, theta=0.1,
    rho_base=0.8, phi=0.2, psi=0.3, lam=0.1, gamma=0.2,
    alpha_val=0.5, beta_val=0.3, epsilon=0.4, sigma=0.4,
    orchestrant_boost_operant=False, orchestrant_boost_operand=False,
    alpha=0.05, beta=0.025, R_cap=None, print_growth_debug=False,
    enable_actor_types=False, enable_negative_externality=False,
    enable_volatility_memory=False, enable_tipping_penalty=False,
    enable_diagnostics=False, tipping_thr=10.0,
    # -------- v1.5 additions ----------
    enable_receiver_sensitivity=False,
    enable_externality_balance=False,
    enable_tipping_recovery=False,
    enable_local_orchestrant_field=False,
)

# ------------------------------------------------------------------ helpers
def _rng(x=None):
    return x if isinstance(x, np.random.RandomState) else np.random.RandomState(x)

def _init_network(n, dens, rng):
    k = max(1, int(dens * n))
    G = nx.watts_strogatz_graph(n, k, p=0.1, seed=rng)
    if not nx.is_connected(G):
        raise ValueError("Network disconnected – raise density.")
    for u, v in G.edges:
        G.edges[u, v]["weight"] = rng.uniform(0.5, 1.0)
    return G

def _assign_types(nodes, rng):
    n = len(nodes)
    roles = (["Generator"]  * int(0.6*n) +
             ["Absorber"]   * int(0.3*n) +
             ["Suppressor"] * (n - int(0.9*n)))
    rng.shuffle(roles)
    return dict(zip(nodes, roles))

def _init_resources(G, rng, actor_types_on, recv_sense_on):
    rng = _rng(rng)
    types = _assign_types(G.nodes, rng) if actor_types_on else {}
    sens  = rng.uniform(0.8, 1.2, size=len(G)) if recv_sense_on else np.ones(len(G))
    P_u = rng.normal(1, 0.4, size=len(G))
    P_s = P_u.mean()
    for i, n in enumerate(G.nodes):
        G.nodes[n].update({
            "O": rng.normal(1, 0.2),
            "P_u": P_u[i],
            "P_s": P_s,
            "R": 0.0,
            "type": types.get(n, "None"),
            "sensitivity": sens[i],
            "is_tipped": False
        })

# ------------------------------------------------------------------ main
def run_simulation(n_actors: int=500, n_steps: int=300, density: float=0.05,
                   seed: int|None=42, **user) -> pd.DataFrame:
    p = {**_DEFAULTS, **user}
    rng = _rng(seed)

    # network & resources ----------------------------------------------------
    G = _init_network(n_actors, density, rng)
    _init_resources(G, rng,
                    p['enable_actor_types'],
                    p['enable_receiver_sensitivity'])
    S = nx.to_numpy_array(G, weight="weight")

    # negative externalities (random 20 % unless suppressor rule flips sign)
    if p['enable_negative_externality']:
        neg_mask = rng.rand(*S.shape) < 0.2
        S = np.where(neg_mask, -S, S)

    # pre-compute arrays that don’t change
    actor_types = np.array([G.nodes[n]["type"] for n in G.nodes])
    sensitivity = np.array([G.nodes[n]["sensitivity"] for n in G.nodes])

    # orchestrant stocks
    R_vec = np.zeros(n_actors)                      # local if enabled
    R_scalar = 0.0
    inflow_win: deque[float] = deque(maxlen=5)

    # diagnostics accumulators
    logs, tipped = [], np.zeros(n_actors, dtype=bool)

    for t in range(n_steps):
        # elasticity blend ---------------------------------------------------
        for n in G.nodes:
            g = G.nodes[n]
            g["P"] = (1 - p['epsilon']) * g["P_u"] + p['epsilon'] * g["P_s"]
        O = np.array([G.nodes[n]["O"] for n in G.nodes])
        P_vec = np.array([G.nodes[n]["P"] for n in G.nodes])

        # role multipliers on operant term
        op_mult = np.ones(n_actors)
        if p['enable_actor_types']:
            op_mult[actor_types == "Generator"]  = 1.2
            op_mult[actor_types == "Absorber"]   = 0.7

        # bounded boosts (scalar or per-actor)
        boost_opern = 1 + (p['alpha']* (R_vec if p['enable_local_orchestrant_field']
                                        else R_scalar)) \
                      / (1 + p['alpha']* (R_vec if p['enable_local_orchestrant_field']
                                           else R_scalar)) \
                      if p['orchestrant_boost_operant'] else 1.0
        boost_operd = 1 + (p['beta'] * (R_vec if p['enable_local_orchestrant_field']
                                        else R_scalar)) \
                      / (1 + p['beta'] * (R_vec if p['enable_local_orchestrant_field']
                                           else R_scalar)) \
                      if p['orchestrant_boost_operand'] else 1.0

        # apply tipping-recovery penalty
        pen = np.where(tipped & p['enable_tipping_recovery'], 0.8, 1.0)

        operand_term = p['w_O'] * O           * boost_operd * pen
        operant_term = p['w_P'] * P_vec * op_mult * boost_opern * pen

        # simple tipping penalty from v1.4 (kept for backwards flag)
        if p['enable_tipping_penalty'] and t > 0:
            penal_mask = prev_V > p['tipping_thr']
            operand_term[penal_mask] *= 0.8
            operant_term[penal_mask] *= 0.8

        y_core = operand_term + operant_term
        y_prev_arr = np.array([G.nodes[n].get("y_prev", 0.0) for n in G.nodes])
        y = y_core + p['theta'] * S.dot(y_prev_arr)

        # externalities ------------------------------------------------------
        e = p['eta'] * y[:, None] * np.exp(-(1 - np.abs(S)) / p['sigma']) \
            * np.sign(S)
        if p['enable_actor_types']:
            sup_idx = np.where(actor_types == "Suppressor")[0]
            e[sup_idx, :] *= -1.5

        ext_received = (e * sensitivity[None, :]).sum(axis=0) \
                       if p['enable_receiver_sensitivity'] else e.sum(axis=0)

        if p['enable_externality_balance']:
            net_ext = e.sum(axis=1) - e.sum(axis=0)

        # orchestrant updates -----------------------------------------------
        if p['enable_local_orchestrant_field']:
            rho_t = np.full(n_actors, p['rho_base'])
            if p['enable_volatility_memory']:
                inflow_win.append(ext_received.sum())
                if len(inflow_win) == inflow_win.maxlen:
                    vol = np.std(inflow_win)
                    rho_t = np.clip(p['rho_base'] * (1 - vol /
                               (abs(ext_received.mean())+1e-6)), 0.5, 0.95)
            R_vec = rho_t * R_vec + (1 - rho_t) * ext_received
            if p['R_cap'] is not None:
                R_vec = np.clip(R_vec, None, p['R_cap'])
            R_scalar = R_vec.mean()
        else:
            inflow_win.append(e.sum())
            if p['enable_volatility_memory'] and len(inflow_win) == inflow_win.maxlen:
                vol = np.std(inflow_win)
                rho_t_scalar = np.clip(p['rho_base'] * (1 - vol /
                                 (abs(e.sum())+1e-6)), 0.5, 0.95)
            else:
                vol, rho_t_scalar = 0.0, p['rho_base']
            R_scalar = rho_t_scalar * R_scalar + (1 - rho_t_scalar) * e.sum()
            if p['R_cap'] is not None:
                R_scalar = min(R_scalar, p['R_cap'])

        # value & advantage --------------------------------------------------
        y_tilde = (1 + p['phi'] * (R_vec if p['enable_local_orchestrant_field']
                                   else R_scalar)) * y
        e_scaled = (1 + p['psi'] * (R_vec[:, None] if p['enable_local_orchestrant_field']
                                    else R_scalar)) * e
        V = (p['alpha_val']*y_tilde +
             p['beta_val'] * e_scaled.sum(axis=0) +
             p['gamma'] * (R_vec if p['enable_local_orchestrant_field'] else R_scalar) +
             p['lam']   * (R_vec if p['enable_local_orchestrant_field'] else R_scalar))
        CA = V - V.mean()

        # update tipping state for next round
        if p['enable_tipping_recovery']:
            tipped = np.where(V > p['tipping_thr'], True,
                     np.where(V < 0.8*p['tipping_thr'], False, tipped))

        # debug --------------------------------------------------------------
        if p['print_growth_debug']:
            print(f"t={t:3d}  R̄={R_scalar:.2f}  "
                  f"boost_Oprnt={np.mean(boost_opern):.2f}")

        # logging ------------------------------------------------------------
        for i, n in enumerate(G.nodes):
            rec = dict(time=t, actor_id=n,
                       operand_value=operand_term[i],
                       operant_value=operant_term[i],
                       orchestrant_value=p['gamma']*
                         (R_vec[i] if p['enable_local_orchestrant_field']
                                   else R_scalar),
                       total_value=V[i],
                       competitive_advantage=CA[i])
            if p['enable_actor_types']:
                rec['actor_type'] = actor_types[i]
            if p['enable_diagnostics']:
                rec['externality_received'] = ext_received[i]
                if p['enable_externality_balance']:
                    rec['net_ext'] = net_ext[i]
                rec['volatility'] = np.std(inflow_win) if inflow_win else 0.0
                rec['rho_t'] = (rho_t[i] if p['enable_local_orchestrant_field']
                                else rho_t_scalar)
                rec['receiver_sensitivity'] = sensitivity[i]
                if p['enable_tipping_recovery']:
                    rec['is_tipped'] = bool(tipped[i])
            logs.append(rec)

            # store previous y for neighbour influence
            G.nodes[n]["y_prev"] = y[i]
        prev_V = V

    return pd.DataFrame(logs)

# ------------------------------------------------------------------ demo
if __name__ == "__main__":
    df_demo = run_simulation(n_actors=15, n_steps=25, density=0.1, seed=1,
                             enable_actor_types=True,
                             enable_receiver_sensitivity=True,
                             enable_externality_balance=True,
                             enable_local_orchestrant_field=True,
                             enable_tipping_recovery=True,
                             enable_diagnostics=True,
                             orchestrant_boost_operant=True,
                             orchestrant_boost_operand=True,
                             R_cap=10)
    print(df_demo.head())
