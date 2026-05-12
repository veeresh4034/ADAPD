"""
correction_chain.py
-------------------
Translates a list of matched (stabiliser, stabiliser) pairs into a binary
qubit correction vector by walking a path between each matched pair on the
physical qubit lattice.

Strategy:
    For each matched pair (s_i, s_j), walk right along the row of s_i
    until the column of s_j is reached, then walk down (or up) to the row
    of s_j.  Apply XOR corrections to every qubit along this path.

    This "L-shaped" path is a simple shortest-path heuristic on the
    rectangular lattice.  It is not unique (an L-path going column-first
    is equally valid), but is consistent across all decoders in this
    codebase.

Authors: Veeresh Kuruba, Srinivas Talabattula, E. S. Shivaleela
         Dept. of ECE, Indian Institute of Science, Bengaluru
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np

from .surface_code import RotatedSurfaceCode


def pairs_to_correction(
    code: RotatedSurfaceCode,
    pairs: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Convert a list of matched stabiliser pairs to a qubit correction vector.

    For each pair (s_i, s_j) of stabiliser indices, an L-shaped path of
    qubits is found on the d × d lattice connecting the centroid of s_i to
    the centroid of s_j.  Every qubit on the path has its correction bit
    flipped (XOR).

    Parameters
    ----------
    code : RotatedSurfaceCode
        The code instance providing lattice geometry.
    pairs : list of (int, int)
        Each tuple is a pair of stabiliser indices to be matched.
        May be empty (returns zero correction).

    Returns
    -------
    correction : np.ndarray of shape (n_qubits,), dtype int8
        Binary correction vector.  correction[q] = 1 means apply X to qubit q.
    """
    correction = np.zeros(code.n_qubits, dtype=np.int8)
    d = code.d

    for (si, sj) in pairs:
        ri = int(round(code.stab_pos[si][0]))
        ci = int(round(code.stab_pos[si][1]))
        rj = int(round(code.stab_pos[sj][0]))
        cj = int(round(code.stab_pos[sj][1]))

        # Walk horizontally first (row ri, columns ci → cj)
        step_c = 1 if cj >= ci else -1
        for col in range(ci, cj, step_c):
            if 0 <= ri < d and 0 <= col < d:
                correction[code.qubit_index(ri, col)] ^= 1

        # Walk vertically second (column cj, rows ri → rj)
        step_r = 1 if rj >= ri else -1
        for row in range(ri, rj, step_r):
            if 0 <= row < d and 0 <= cj < d:
                correction[code.qubit_index(row, cj)] ^= 1

    return correction


def drop_boundary_nearest(
    code: RotatedSurfaceCode,
    defects: List[int],
) -> List[int]:
    """
    If `defects` has odd length, remove the defect whose stabiliser is
    closest to the code boundary and return the even-length remainder.

    The boundary distance of stabiliser s at position (r, c) is:
        min(r, d-1-r, c, d-1-c)

    This is used as the boundary-pairing approximation: a defect near
    the boundary is paired with the virtual boundary node (i.e., dropped).

    Parameters
    ----------
    code : RotatedSurfaceCode
    defects : list of int
        Stabiliser indices of current syndrome defects.

    Returns
    -------
    list of int
        Defect list with even length (at most one element removed).
    """
    if len(defects) % 2 == 0:
        return defects

    d = code.d
    best_idx = 0
    best_dist = float('inf')

    for i, stab in enumerate(defects):
        r, c = code.stab_pos[stab]
        boundary_dist = min(r, d - 1.0 - r, c, d - 1.0 - c)
        if boundary_dist < best_dist:
            best_dist = boundary_dist
            best_idx = i

    return [x for i, x in enumerate(defects) if i != best_idx]
