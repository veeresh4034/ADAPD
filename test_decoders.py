"""
test_decoders.py
----------------
Unit tests for MWPM, Union-Find, and ADAPD decoders.

Run with:
    python -m pytest tests/ -v

Authors: Veeresh Kuruba, Srinivas Talabattula, E. S. Shivaleela
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from src.surface_code import RotatedSurfaceCode
from src.noise_models import depolarising_errors
from src.decoders import decode_mwpm, decode_uf, decode_adapd


DECODERS = [
    ('MWPM',      decode_mwpm),
    ('UnionFind', decode_uf),
    ('ADAPD',     decode_adapd),
]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def code3():
    return RotatedSurfaceCode(d=3)

@pytest.fixture
def code5():
    return RotatedSurfaceCode(d=5)


# ── Interface tests (all decoders) ────────────────────────────────────────────

@pytest.mark.parametrize("dec_name,dec_fn", DECODERS)
def test_trivial_syndrome_returns_zero(dec_name, dec_fn, code5):
    """Empty syndrome → zero correction vector."""
    syndrome = np.zeros(code5.n_stabs, dtype=np.int8)
    correction, lat = dec_fn(code5, syndrome, p=0.01)
    assert np.all(correction == 0), \
        f"{dec_name}: empty syndrome should return zero correction"
    assert lat >= 0.0, f"{dec_name}: latency should be non-negative"


@pytest.mark.parametrize("dec_name,dec_fn", DECODERS)
def test_correction_shape(dec_name, dec_fn, code5):
    """Correction must have shape (n_qubits,)."""
    rng = np.random.default_rng(10)
    errors = depolarising_errors(code5.n_qubits, p=0.05, rng=rng)
    syndrome = code5.measure_syndrome(errors)
    correction, _ = dec_fn(code5, syndrome, p=0.05)
    assert correction.shape == (code5.n_qubits,), \
        f"{dec_name}: correction shape mismatch"


@pytest.mark.parametrize("dec_name,dec_fn", DECODERS)
def test_correction_binary(dec_name, dec_fn, code5):
    """Correction must be a binary vector."""
    rng = np.random.default_rng(20)
    errors = depolarising_errors(code5.n_qubits, p=0.05, rng=rng)
    syndrome = code5.measure_syndrome(errors)
    correction, _ = dec_fn(code5, syndrome, p=0.05)
    assert set(correction).issubset({0, 1}), \
        f"{dec_name}: correction must be binary"


@pytest.mark.parametrize("dec_name,dec_fn", DECODERS)
def test_latency_positive(dec_name, dec_fn, code5):
    rng = np.random.default_rng(30)
    errors = depolarising_errors(code5.n_qubits, p=0.05, rng=rng)
    syndrome = code5.measure_syndrome(errors)
    _, lat = dec_fn(code5, syndrome, p=0.05)
    assert lat >= 0.0


@pytest.mark.parametrize("dec_name,dec_fn", DECODERS)
@pytest.mark.parametrize("d", [3, 5, 7])
def test_decoder_runs_on_all_distances(dec_name, dec_fn, d):
    code = RotatedSurfaceCode(d=d)
    rng  = np.random.default_rng(100 + d)
    errors   = depolarising_errors(code.n_qubits, p=0.01, rng=rng)
    syndrome = code.measure_syndrome(errors)
    correction, lat = dec_fn(code, syndrome, p=0.01)
    assert correction.shape == (code.n_qubits,)


# ── Correctness smoke tests ───────────────────────────────────────────────────

@pytest.mark.parametrize("dec_name,dec_fn", DECODERS)
def test_low_error_rate_high_success(dec_name, dec_fn):
    """At very low error rate, decoder should succeed most of the time."""
    code = RotatedSurfaceCode(d=5)
    rng  = np.random.default_rng(42)
    n_trials = 200
    failures = 0
    for _ in range(n_trials):
        errors   = depolarising_errors(code.n_qubits, p=0.005, rng=rng)
        syndrome = code.measure_syndrome(errors)
        correction, _ = dec_fn(code, syndrome, p=0.005)
        residual = (errors + correction) % 2
        if code.logical_failure(residual):
            failures += 1
    ler = failures / n_trials
    # At p=0.5% on d=5, LER should be well below 10%
    assert ler < 0.10, \
        f"{dec_name}: LER={ler:.3f} at p=0.5% is unexpectedly high"


# ── ADAPD-specific tests ──────────────────────────────────────────────────────

def test_adapd_rho_star_zero_matches_mwpm_closely(code5):
    """With rho_star=0 all clusters → Tier 2 → MWPM. Results should match."""
    rng = np.random.default_rng(99)
    n_matches = 0
    n_trials  = 50
    for _ in range(n_trials):
        errors   = depolarising_errors(code5.n_qubits, p=0.01, rng=rng)
        syndrome = code5.measure_syndrome(errors)
        c_mwpm,  _ = decode_mwpm(code5, syndrome, p=0.01)
        c_adapd, _ = decode_adapd(code5, syndrome, p=0.01, rho_star=0.0)
        if np.array_equal(c_mwpm, c_adapd):
            n_matches += 1
    # With rho_star=0 ADAPD should use Tier 2 (MWPM) for all non-trivial clusters
    # Expect high but not necessarily 100% match due to p_hat vs fixed-p weights
    assert n_matches >= n_trials * 0.5, \
        f"ADAPD with rho_star=0 matched MWPM only {n_matches}/{n_trials} times"


def test_adapd_rho_star_large_is_fast(code5):
    """With rho_star very large, all clusters → Tier 1 (greedy), which is fast."""
    rng = np.random.default_rng(77)
    lats_greedy, lats_mwpm = [], []
    for _ in range(30):
        errors   = depolarising_errors(code5.n_qubits, p=0.02, rng=rng)
        syndrome = code5.measure_syndrome(errors)
        _, lat_a = decode_adapd(code5, syndrome, p=0.02, rho_star=1e6)
        _, lat_m = decode_mwpm(code5, syndrome, p=0.02)
        lats_greedy.append(lat_a)
        lats_mwpm.append(lat_m)
    # ADAPD with giant rho_star (all greedy) should be at most as slow as MWPM on avg
    assert np.mean(lats_greedy) <= np.mean(lats_mwpm) * 1.2, \
        "ADAPD (all greedy) should not be slower than MWPM"
