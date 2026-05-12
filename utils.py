"""
utils.py
--------
Utility data structures and helper functions used by the decoders.

Contents:
    - UnionFind: path-compressed, union-by-rank union-find data structure
    - manhattan_distance: stabiliser-centroid Manhattan distance

Authors: Veeresh Kuruba, Srinivas Talabattula, E. S. Shivaleela
         Dept. of ECE, Indian Institute of Science, Bengaluru
"""

from __future__ import annotations
from typing import List, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Union-Find with path compression and union by rank
# ──────────────────────────────────────────────────────────────────────────────

class UnionFind:
    """
    Disjoint-set (union-find) data structure with path compression
    and union by rank.

    Amortised per-operation complexity: O(α(n)), where α is the inverse
    Ackermann function — effectively constant for all practical n.

    Parameters
    ----------
    n : int
        Number of elements (labelled 0 … n-1).

    Examples
    --------
    >>> uf = UnionFind(5)
    >>> uf.union(0, 1)
    >>> uf.union(2, 3)
    >>> uf.find(0) == uf.find(1)
    True
    >>> uf.find(0) == uf.find(2)
    False
    """

    def __init__(self, n: int) -> None:
        self._parent: List[int] = list(range(n))
        self._rank:   List[int] = [0] * n

    def find(self, x: int) -> int:
        """
        Return the representative (root) of the set containing x.
        Uses iterative path compression (path halving).
        """
        while self._parent[x] != x:
            # Path halving: point each node to its grandparent
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        """
        Merge the sets containing x and y.

        Returns
        -------
        bool
            True if x and y were in different sets (merge performed);
            False if they were already in the same set.
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        # Union by rank: attach smaller tree under larger tree
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """Return True if x and y are in the same set."""
        return self.find(x) == self.find(y)

    def components(self, n: int) -> dict:
        """
        Return a dict mapping root → list of members for all n elements.

        Parameters
        ----------
        n : int
            Total number of elements (must equal the n passed to __init__).
        """
        from collections import defaultdict
        groups: dict = defaultdict(list)
        for i in range(n):
            groups[self.find(i)].append(i)
        return dict(groups)


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def manhattan_distance(
    pos_a: Tuple[float, float],
    pos_b: Tuple[float, float],
) -> float:
    """
    Manhattan (L1) distance between two 2-D points.

    Parameters
    ----------
    pos_a, pos_b : (float, float)
        Positions in (row, col) stabiliser-centroid coordinates.

    Returns
    -------
    float
        |row_a - row_b| + |col_a - col_b|
    """
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


def log_likelihood_weight(distance: float, p: float) -> float:
    """
    Compute the MWPM edge weight for a given Manhattan distance and
    physical error rate.

    The weight is the negative log-likelihood of the error chain:
        w = distance * log[(1 - p/3) / (p/3)]

    Edges with lower weight correspond to more likely error paths.

    Parameters
    ----------
    distance : float
        Manhattan distance between two syndrome defects (stabiliser
        centroid distance).
    p : float
        Physical error rate per qubit per round.

    Returns
    -------
    float
        Non-negative edge weight (larger = less likely error chain).
    """
    eps = 1e-9  # Avoid log(0)
    return distance * float(__import__('math').log(
        (1.0 - p / 3.0 + eps) / (p / 3.0 + eps)
    ))
