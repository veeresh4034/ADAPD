"""
noise_models.py
---------------
Pauli noise models for Monte Carlo surface-code simulation.

Implemented models:
    1. Independent depolarising noise
    2. Z-biased Pauli noise (parameterised by bias η)

Authors: Veeresh Kuruba, Srinivas Talabattula, E. S. Shivaleela
         Dept. of ECE, Indian Institute of Science, Bengaluru
"""

from __future__ import annotations
import numpy as np
from numpy.random import Generator


# ──────────────────────────────────────────────────────────────────────────────
# 1. Independent depolarising noise
# ──────────────────────────────────────────────────────────────────────────────

def depolarising_errors(
    n_qubits: int,
    p: float,
    rng: Generator,
) -> np.ndarray:
    """
    Sample X errors under the independent depolarising channel.

    Under the depolarising channel with rate p, each qubit independently
    suffers an X, Y, or Z error with probability p/3 each.  Only the
    X-sector syndrome is simulated here (detecting X and Y errors), so
    the effective X-error probability per qubit is p/3.

    Parameters
    ----------
    n_qubits : int
        Number of physical qubits.
    p : float
        Total depolarising error rate per qubit per round.  Must be in [0, 1].
    rng : numpy.random.Generator
        NumPy random generator instance (for reproducibility, pass a seeded
        generator: ``np.random.default_rng(seed)``).

    Returns
    -------
    errors : np.ndarray of shape (n_qubits,), dtype int8
        Binary vector; errors[i] = 1 if qubit i has an X error.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> errors = depolarising_errors(n_qubits=25, p=0.01, rng=rng)
    >>> errors.shape
    (25,)
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Error rate p must be in [0, 1], got {p}.")
    return (rng.random(n_qubits) < p / 3.0).astype(np.int8)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Z-biased Pauli noise
# ──────────────────────────────────────────────────────────────────────────────

def biased_errors(
    n_qubits: int,
    p: float,
    eta: float,
    rng: Generator,
) -> np.ndarray:
    """
    Sample X errors under a Z-biased Pauli noise channel.

    Under Z-biased noise with bias parameter η:
        P(Z error) = η * p / (2η + 1)
        P(X error) = p / (2 * (2η + 1))
        P(Y error) = p / (2 * (2η + 1))

    For η = 1: recovers equal-probability depolarising (p/3 each).
    For η >> 1: Z errors dominate; X errors are rare (P(X) ≈ p / (4η)).

    Only X errors activate the Z-sector syndrome, so only P(X) matters
    for our simulation.

    Parameters
    ----------
    n_qubits : int
        Number of physical qubits.
    p : float
        Total error rate per qubit per round.  Must be in [0, 1].
    eta : float
        Pauli bias parameter (η ≥ 1).  η = 1 ↔ depolarising.
    rng : numpy.random.Generator
        NumPy random generator instance.

    Returns
    -------
    errors : np.ndarray of shape (n_qubits,), dtype int8
        Binary X-error vector.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> errors = biased_errors(n_qubits=25, p=0.01, eta=10, rng=rng)
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Error rate p must be in [0, 1], got {p}.")
    if eta < 1.0:
        raise ValueError(f"Bias η must be ≥ 1, got {eta}.")

    p_x = p / (2.0 * (2.0 * eta + 1.0))
    return (rng.random(n_qubits) < p_x).astype(np.int8)


# ──────────────────────────────────────────────────────────────────────────────
# Utility: effective X-error probability
# ──────────────────────────────────────────────────────────────────────────────

def effective_x_rate(p: float, eta: float = 1.0) -> float:
    """
    Return the effective X-error probability per qubit for the given
    noise model parameters.

    For depolarising noise (η = 1):   P(X) = p / 3
    For Z-biased noise (η ≥ 1):       P(X) = p / (2*(2η+1))

    Parameters
    ----------
    p : float
        Total error rate.
    eta : float
        Bias parameter (default 1.0 = depolarising).

    Returns
    -------
    float
        Effective X-error probability per qubit.
    """
    if eta == 1.0:
        return p / 3.0
    return p / (2.0 * (2.0 * eta + 1.0))
