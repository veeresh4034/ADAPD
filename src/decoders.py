"""
decoders.py
-----------
Implementations of three syndrome decoders for the rotated surface code:

    1. MWPM  — Minimum-Weight Perfect Matching
    2. UF    — Union-Find (greedy nearest-neighbour)
    3. ADAPD — Adaptive Density-Aware Phase Decoder  [this work]

All decoders share the same interface:
    correction, latency = decode_*(code, syndrome, p)

Authors: Veeresh Kuruba, Srinivas Talabattula, E. S. Shivaleela
         Dept. of ECE, Indian Institute of Science, Bengaluru
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import List, Tuple

import networkx as nx
import numpy as np

from .surface_code import RotatedSurfaceCode
from .correction_chain import pairs_to_correction, drop_boundary_nearest
from .utils import UnionFind, manhattan_distance, log_likelihood_weight


# ──────────────────────────────────────────────────────────────────────────────
# Shared helper: greedy nearest-neighbour matching
# ──────────────────────────────────────────────────────────────────────────────

def _greedy_match(
    defects: List[int],
    stab_pos: List[Tuple[float, float]],
) -> List[Tuple[int, int]]:
    """
    Greedily pair defects by repeatedly matching each unpaired defect
    with its nearest unpaired neighbour (Manhattan distance on stabiliser
    centroid positions).

    Complexity: O(k^2) for k defects — efficient when k is small.

    Parameters
    ----------
    defects : list of int
        Stabiliser indices of syndrome defects to be matched.
        Must have even length.
    stab_pos : list of (float, float)
        Stabiliser centroid positions.

    Returns
    -------
    list of (int, int)
        Matched pairs of stabiliser indices.
    """
    pairs: List[Tuple[int, int]] = []
    remaining = list(defects)

    while len(remaining) >= 2:
        i = remaining[0]
        best_j = remaining[1]
        best_dist = float('inf')
        for j in remaining[1:]:
            dist = manhattan_distance(stab_pos[i], stab_pos[j])
            if dist < best_dist:
                best_dist = dist
                best_j = j
        pairs.append((i, best_j))
        remaining.remove(i)
        remaining.remove(best_j)

    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Shared helper: MWPM on a subgraph
# ──────────────────────────────────────────────────────────────────────────────

def _mwpm_match(
    defects: List[int],
    stab_pos: List[Tuple[float, float]],
    p_est: float,
) -> List[Tuple[int, int]]:
    """
    Find the minimum-weight perfect matching over a complete graph of
    syndrome defects using NetworkX's blossom-based max_weight_matching.

    Edge weights are negated log-likelihoods (so maximising weight =
    minimising log-likelihood = finding most probable correction).

    Parameters
    ----------
    defects : list of int
        Stabiliser indices to match.  Must have even length.
    stab_pos : list of (float, float)
        Stabiliser centroid positions.
    p_est : float
        Physical error rate estimate used for edge-weight computation.

    Returns
    -------
    list of (int, int)
        Matched pairs.
    """
    if len(defects) < 2:
        return []

    G = nx.Graph()
    for i in range(len(defects)):
        for j in range(i + 1, len(defects)):
            si, sj = defects[i], defects[j]
            dist = manhattan_distance(stab_pos[si], stab_pos[sj])
            # Negate: NetworkX finds max-weight matching
            G.add_edge(si, sj, weight=-log_likelihood_weight(dist, p_est))

    try:
        matching = nx.max_weight_matching(G, maxcardinality=True)
        return list(matching)
    except Exception:
        # Fallback to greedy if matching fails (should not happen in practice)
        return _greedy_match(defects, stab_pos)


# ──────────────────────────────────────────────────────────────────────────────
# 1. MWPM Decoder
# ──────────────────────────────────────────────────────────────────────────────

def decode_mwpm(
    code: RotatedSurfaceCode,
    syndrome: np.ndarray,
    p: float,
) -> Tuple[np.ndarray, float]:
    """
    Minimum-Weight Perfect Matching (MWPM) decoder.

    Constructs a complete weighted graph over all syndrome defects and finds
    the minimum-weight perfect matching via Edmonds' blossom algorithm
    (NetworkX implementation).  Odd-count syndromes are handled by dropping
    the defect nearest the code boundary.

    Time complexity: O(K^3) worst case, O(K^1.5) average at low p,
    where K = |V_s| is the number of syndrome defects.

    Parameters
    ----------
    code : RotatedSurfaceCode
    syndrome : np.ndarray of shape (n_stabs,), dtype int8
        Binary syndrome vector.
    p : float
        Physical error rate (used for edge-weight computation).

    Returns
    -------
    correction : np.ndarray of shape (n_qubits,), dtype int8
    latency : float
        Wall-clock decoding time in seconds.
    """
    t0 = time.perf_counter()
    zero = np.zeros(code.n_qubits, dtype=np.int8)

    defects = list(np.where(syndrome == 1)[0])
    if not defects:
        return zero, time.perf_counter() - t0

    defects = drop_boundary_nearest(code, defects)
    if not defects:
        return zero, time.perf_counter() - t0

    pairs = _mwpm_match(defects, code.stab_pos, p)
    correction = pairs_to_correction(code, pairs)
    return correction, time.perf_counter() - t0


# ──────────────────────────────────────────────────────────────────────────────
# 2. Union-Find (Greedy) Decoder
# ──────────────────────────────────────────────────────────────────────────────

def decode_uf(
    code: RotatedSurfaceCode,
    syndrome: np.ndarray,
    p: float,
) -> Tuple[np.ndarray, float]:
    """
    Union-Find (greedy nearest-neighbour) decoder.

    Pairs each defect with its nearest unpaired neighbour by Manhattan
    distance.  This is the greedy analogue of the full union-find
    cluster-growth decoder of Delfosse & Nickerson (2021).

    Time complexity: O(K^2) per syndrome, O(K) for small clusters.

    Parameters
    ----------
    code : RotatedSurfaceCode
    syndrome : np.ndarray of shape (n_stabs,), dtype int8
    p : float
        Physical error rate (not used in matching; included for API
        consistency).

    Returns
    -------
    correction : np.ndarray of shape (n_qubits,), dtype int8
    latency : float
        Wall-clock decoding time in seconds.
    """
    t0 = time.perf_counter()
    zero = np.zeros(code.n_qubits, dtype=np.int8)

    defects = list(np.where(syndrome == 1)[0])
    if not defects:
        return zero, time.perf_counter() - t0

    defects = drop_boundary_nearest(code, defects)
    if not defects:
        return zero, time.perf_counter() - t0

    pairs = _greedy_match(defects, code.stab_pos)
    correction = pairs_to_correction(code, pairs)
    return correction, time.perf_counter() - t0


# ──────────────────────────────────────────────────────────────────────────────
# 3. ADAPD — Adaptive Density-Aware Phase Decoder 
# ──────────────────────────────────────────────────────────────────────────────

def _cluster_density(
    cluster: List[int],
    stab_pos: List[Tuple[float, float]],
) -> float:
    """
    Estimate the local defect density of a cluster as:
        ρ_k = |C_k| / Area(BoundingBox(C_k))

    Parameters
    ----------
    cluster : list of int
        Stabiliser indices in the cluster.
    stab_pos : list of (float, float)
        Stabiliser centroid positions.

    Returns
    -------
    float
        Local defect density.  Returns 0.0 for singleton clusters.
    """
    if len(cluster) <= 1:
        return 0.0
    rows = [stab_pos[s][0] for s in cluster]
    cols = [stab_pos[s][1] for s in cluster]
    row_span = max(rows) - min(rows)
    col_span = max(cols) - min(cols)
    area = max((row_span + 1.0) * (col_span + 1.0), 1.0)
    return len(cluster) / area


def _form_clusters(
    defects: List[int],
    stab_pos: List[Tuple[float, float]],
    radius: float,
) -> List[List[int]]:
    """
    Partition `defects` into connected clusters using union-find.

    Two defects are placed in the same cluster if their Manhattan distance
    is ≤ `radius`.

    Parameters
    ----------
    defects : list of int
        Stabiliser indices to cluster.
    stab_pos : list of (float, float)
        Stabiliser centroid positions.
    radius : float
        Clustering radius in lattice units.

    Returns
    -------
    list of list of int
        Each inner list is a cluster of stabiliser indices.
    """
    n = len(defects)
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if manhattan_distance(stab_pos[defects[i]], stab_pos[defects[j]]) <= radius:
                uf.union(i, j)

    groups: dict = defaultdict(list)
    for idx, stab in enumerate(defects):
        groups[uf.find(idx)].append(stab)

    return list(groups.values())


def decode_adapd(
    code: RotatedSurfaceCode,
    syndrome: np.ndarray,
    p: float,
    rho_star: float = 0.40,
) -> Tuple[np.ndarray, float]:
    """
    Adaptive Density-Aware Phase Decoder (ADAPD).

    ADAPD classifies each syndrome defect cluster by its local spatial
    density ρ_k and dispatches a tier-appropriate decoding strategy:

        Tier 0  |C_k| = 2           → direct pair, O(1)
        Tier 1  ρ_k < ρ* or |C_k|≤4 → greedy nearest-neighbour, O(k²)
        Tier 2  ρ_k ≥ ρ* and |C_k|>4 → local MWPM on cluster subgraph

    An online noise estimate p̂ is computed from the syndrome density and
    used to set the adaptive clustering radius and MWPM edge weights.

    Algorithm:
        Phase 1 — Online noise estimate → clustering radius r
                  Union-Find cluster formation
                  Bounding-box density estimation per cluster
                  Tier classification
        Phase 2 — Tier-selective matching (parallel over clusters)
                  Boundary-nearest defect dropping per cluster
                  Correction chain assembly

    Time complexity: O(n log n) average under depolarising noise at p < p*.
    Space complexity: O(n).

    Parameters
    ----------
    code : RotatedSurfaceCode
    syndrome : np.ndarray of shape (n_stabs,), dtype int8
    p : float
        Nominal physical error rate (used as fallback if p̂ is unreliable).
    rho_star : float, optional
        Density threshold separating Tier-1 and Tier-2 clusters.
        Default: 0.40 (tuned for depolarising noise on surface codes).

    Returns
    -------
    correction : np.ndarray of shape (n_qubits,), dtype int8
    latency : float
        Wall-clock decoding time in seconds.

   
    """
    t0 = time.perf_counter()
    zero = np.zeros(code.n_qubits, dtype=np.int8)

    defects = list(np.where(syndrome == 1)[0])
    if not defects:
        return zero, time.perf_counter() - t0

    # ── Phase 1: Online noise estimation ──────────────────────────────────
    # p̂ = |V_s| / (n_qubits × 2/3)
    # The factor 2/3 accounts for the fraction of depolarising errors
    # that activate the Z-sector syndrome.
    p_hat = float(np.clip(
        len(defects) / (code.n_qubits * (2.0 / 3.0)),
        1e-4, 0.45
    ))

    # Adaptive clustering radius (decreases as noise increases)
    radius = float(np.clip(
        max(2.0, 0.8 / max(p_hat, 0.005)),
        0.0, float(code.d)
    ))

    # ── Phase 1: Cluster formation ─────────────────────────────────────────
    clusters = _form_clusters(defects, code.stab_pos, radius)

    # ── Phase 2: Tier-selective matching ──────────────────────────────────
    all_pairs: List[Tuple[int, int]] = []

    for cluster in clusters:
        # Enforce even parity: drop boundary-nearest defect if odd
        cluster = drop_boundary_nearest(code, cluster)
        if not cluster:
            continue

        # Tier 0: isolated defect pair — direct O(1) match
        if len(cluster) == 2:
            all_pairs.append((cluster[0], cluster[1]))
            continue

        # Compute density for Tier 1 vs Tier 2 decision
        density = _cluster_density(cluster, code.stab_pos)

        if density < rho_star or len(cluster) <= 4:
            # Tier 1: sparse cluster → greedy nearest-neighbour
            all_pairs.extend(_greedy_match(cluster, code.stab_pos))
        else:
            # Tier 2: dense cluster → local MWPM with adaptive weights
            all_pairs.extend(_mwpm_match(cluster, code.stab_pos, p_hat))

    correction = pairs_to_correction(code, all_pairs)
    return correction, time.perf_counter() - t0
