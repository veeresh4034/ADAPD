"""
surface_code.py
---------------
Rotated surface code geometry, stabiliser construction, syndrome extraction,
and logical failure detection.

Reference:
    Fowler et al., "Surface codes: Towards practical large-scale quantum
    computation", Phys. Rev. A 86, 032324 (2012).

Authors: Veeresh Kuruba, Srinivas Talabattula, E. S. Shivaleela
         Dept. of ECE, Indian Institute of Science, Bengaluru
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple


class RotatedSurfaceCode:
    """
    Distance-d rotated surface code on a d x d physical qubit lattice.

    Encoding:
        - n_qubits = d^2 data qubits on a d × d grid (row-major indexing).
        - Z-type stabilisers detect X (and Y) errors.
        - X-type stabilisers detect Z (and Y) errors.
        - We simulate the Z-sector (detecting X errors) throughout; the
          X-sector is symmetric and identical in structure.

    Stabiliser layout:
        - Interior plaquettes: weight-4 Z-stabilisers on all (d-1)^2
          interior squares of the lattice.
        - Boundary half-plaquettes: weight-2 Z-stabilisers on the top and
          bottom boundary edges (alternating, per the rotated construction).

    Logical operators:
        - Logical X-bar: top row of qubits (row 0).
        - Logical Z-bar: left column of qubits (column 0).
        - Logical failure is detected via the left column (Z-bar).

    Parameters
    ----------
    d : int
        Code distance. Must be an odd integer ≥ 3.

    Attributes
    ----------
    d : int
        Code distance.
    n_qubits : int
        Number of physical data qubits (= d^2).
    n_stabs : int
        Number of Z-type stabilisers.
    z_stabs : List[List[int]]
        Each element is a list of qubit indices in the support of one
        Z-stabiliser.
    stab_pos : List[Tuple[float, float]]
        (row, col) centroid of each stabiliser in lattice coordinates.
    """

    def __init__(self, d: int) -> None:
        if d < 3:
            raise ValueError(f"Code distance must be ≥ 3, got {d}.")
        self.d = d
        self.n_qubits = d * d
        self._build_stabilisers()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def qubit_index(self, row: int, col: int) -> int:
        """Return the linear index of the qubit at lattice position (row, col)."""
        return row * self.d + col

    def qubit_position(self, index: int) -> Tuple[int, int]:
        """Return the (row, col) lattice position of qubit `index`."""
        return divmod(index, self.d)

    def _build_stabilisers(self) -> None:
        """Construct the Z-stabiliser list and their centroid positions."""
        d = self.d
        self.z_stabs: List[List[int]] = []

        # --- Interior weight-4 plaquette stabilisers ---
        for r in range(d - 1):
            for c in range(d - 1):
                self.z_stabs.append([
                    self.qubit_index(r,     c),
                    self.qubit_index(r,     c + 1),
                    self.qubit_index(r + 1, c),
                    self.qubit_index(r + 1, c + 1),
                ])

        # --- Boundary weight-2 stabilisers (top row, even columns) ---
        for c in range(0, d - 1, 2):
            self.z_stabs.append([
                self.qubit_index(0, c),
                self.qubit_index(0, c + 1),
            ])

        # --- Boundary weight-2 stabilisers (bottom row, odd columns) ---
        for c in range(1, d - 1, 2):
            self.z_stabs.append([
                self.qubit_index(d - 1, c),
                self.qubit_index(d - 1, c + 1),
            ])

        self.n_stabs = len(self.z_stabs)

        # Stabiliser centroids in (row, col) coordinates
        self.stab_pos: List[Tuple[float, float]] = []
        for qubits in self.z_stabs:
            rows = [q // d for q in qubits]
            cols = [q  % d for q in qubits]
            self.stab_pos.append((float(np.mean(rows)), float(np.mean(cols))))

    # ------------------------------------------------------------------
    # Syndrome measurement
    # ------------------------------------------------------------------

    def measure_syndrome(self, error: np.ndarray) -> np.ndarray:
        """
        Compute the Z-sector syndrome for a given X-error pattern.

        Parameters
        ----------
        error : np.ndarray of shape (n_qubits,), dtype int8 or bool
            Binary X-error vector; error[i] = 1 if qubit i has an X error.

        Returns
        -------
        syndrome : np.ndarray of shape (n_stabs,), dtype int8
            syndrome[i] = 1 if stabiliser i is violated (defect present).
        """
        error = np.asarray(error, dtype=np.int8)
        syndrome = np.zeros(self.n_stabs, dtype=np.int8)
        for i, qubits in enumerate(self.z_stabs):
            syndrome[i] = int(np.sum(error[qubits]) % 2)
        return syndrome

    # ------------------------------------------------------------------
    # Logical failure check
    # ------------------------------------------------------------------

    def logical_failure(self, residual: np.ndarray) -> bool:
        """
        Check whether a residual error causes a logical X failure.

        A logical failure occurs when the residual has odd overlap with the
        logical Z-bar operator (the left column of the lattice).

        Parameters
        ----------
        residual : np.ndarray of shape (n_qubits,), dtype int8 or bool
            Residual X-error after applying the decoder's correction:
            residual = (original_error + correction) mod 2.

        Returns
        -------
        bool
            True if a logical failure occurred; False otherwise.
        """
        logical_z = [self.qubit_index(r, 0) for r in range(self.d)]
        return bool(np.sum(residual[logical_z]) % 2)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RotatedSurfaceCode(d={self.d}, "
            f"n_qubits={self.n_qubits}, "
            f"n_stabs={self.n_stabs})"
        )
