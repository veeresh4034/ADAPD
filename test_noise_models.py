"""
test_noise_models.py
--------------------
Unit tests for depolarising and biased noise models.

Run with:
    python -m pytest tests/ -v

Authors: Veeresh Kuruba, Srinivas Talabattula, E. S. Shivaleela
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from src.noise_models import depolarising_errors, biased_errors, effective_x_rate


N = 10_000   # Use large n for statistical tests
RNG = np.random.default_rng(0)


# ── depolarising_errors ───────────────────────────────────────────────────────

def test_depol_shape():
    errors = depolarising_errors(N, p=0.01, rng=RNG)
    assert errors.shape == (N,)


def test_depol_binary():
    errors = depolarising_errors(N, p=0.01, rng=RNG)
    assert set(errors).issubset({0, 1})


def test_depol_zero_rate():
    errors = depolarising_errors(N, p=0.0, rng=RNG)
    assert np.all(errors == 0), "p=0 should produce no errors"


def test_depol_full_rate():
    errors = depolarising_errors(N, p=1.0, rng=RNG)
    # p=1 → p/3 ≈ 0.333, so ~1/3 of qubits should have errors
    frac = errors.mean()
    assert abs(frac - 1 / 3) < 0.02, \
        f"At p=1, expected ~1/3 X errors, got {frac:.3f}"


def test_depol_rate_accuracy():
    """Check that empirical X-error rate ≈ p/3."""
    for p in [0.01, 0.05, 0.10]:
        rng = np.random.default_rng(42)
        errors = depolarising_errors(N, p=p, rng=rng)
        empirical = errors.mean()
        expected  = p / 3
        assert abs(empirical - expected) < 3 * np.sqrt(expected * (1 - expected) / N), \
            f"Empirical rate {empirical:.4f} ≠ expected {expected:.4f} at p={p}"


def test_depol_invalid_rate():
    with pytest.raises(ValueError):
        depolarising_errors(N, p=1.5, rng=RNG)
    with pytest.raises(ValueError):
        depolarising_errors(N, p=-0.1, rng=RNG)


# ── biased_errors ─────────────────────────────────────────────────────────────

def test_biased_shape():
    errors = biased_errors(N, p=0.01, eta=10, rng=RNG)
    assert errors.shape == (N,)


def test_biased_binary():
    errors = biased_errors(N, p=0.01, eta=10, rng=RNG)
    assert set(errors).issubset({0, 1})


def test_biased_rate_lower_than_depol():
    """Z-biased noise (η>1) should produce fewer X errors than depolarising."""
    rng = np.random.default_rng(1)
    p = 0.05
    e_depol = depolarising_errors(N, p=p, rng=rng).mean()
    rng2 = np.random.default_rng(1)
    e_bias  = biased_errors(N, p=p, eta=10, rng=rng2).mean()
    assert e_bias < e_depol, \
        "Biased noise (η=10) should produce fewer X errors than depolarising"


def test_biased_eta1_approx_depol():
    """η=1 should give P(X) ≈ p/3 (same as depolarising)."""
    rng = np.random.default_rng(2)
    p = 0.06
    e_bias = biased_errors(N, p=p, eta=1, rng=rng).mean()
    expected = p / (2 * 3)  # P(X) = p / (2*(2*1+1)) = p/6
    # Note: for eta=1: P(X) = p/(2*(2+1)) = p/6, not p/3
    # (depolarising has P(X) = p/3, biased eta=1 gives p/6)
    assert abs(e_bias - expected) < 3 * np.sqrt(expected * (1 - expected) / N)


def test_biased_invalid_eta():
    with pytest.raises(ValueError):
        biased_errors(N, p=0.01, eta=0.5, rng=RNG)


# ── effective_x_rate ──────────────────────────────────────────────────────────

def test_effective_x_rate_depol():
    assert abs(effective_x_rate(0.09, eta=1.0) - 0.09 / 3) < 1e-9


def test_effective_x_rate_biased():
    p, eta = 0.10, 10
    expected = p / (2 * (2 * eta + 1))
    assert abs(effective_x_rate(p, eta) - expected) < 1e-9


def test_effective_x_rate_zero():
    assert effective_x_rate(0.0) == 0.0
