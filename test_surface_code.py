"""
test_surface_code.py
--------------------
Unit tests for the RotatedSurfaceCode class.

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


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(params=[3, 5, 7])
def code(request):
    return RotatedSurfaceCode(d=request.param)


# ── Basic construction ────────────────────────────────────────────────────────

def test_qubit_count(code):
    assert code.n_qubits == code.d ** 2


def test_stab_count_positive(code):
    assert code.n_stabs > 0


def test_stab_positions_have_correct_count(code):
    assert len(code.stab_pos) == code.n_stabs


def test_qubit_index_roundtrip(code):
    d = code.d
    for r in range(d):
        for c in range(d):
            idx = code.qubit_index(r, c)
            r2, c2 = code.qubit_position(idx)
            assert (r2, c2) == (r, c), \
                f"Roundtrip failed at ({r},{c}): got ({r2},{c2})"


def test_invalid_distance():
    with pytest.raises(ValueError):
        RotatedSurfaceCode(d=2)


# ── Syndrome extraction ───────────────────────────────────────────────────────

def test_no_error_trivial_syndrome(code):
    errors = np.zeros(code.n_qubits, dtype=np.int8)
    syndrome = code.measure_syndrome(errors)
    assert np.all(syndrome == 0), "Zero error should give trivial syndrome"


def test_syndrome_binary(code):
    rng = np.random.default_rng(0)
    errors = (rng.random(code.n_qubits) < 0.1).astype(np.int8)
    syndrome = code.measure_syndrome(errors)
    assert set(syndrome).issubset({0, 1}), "Syndrome must be binary"
    assert syndrome.shape == (code.n_stabs,)


def test_single_qubit_error_triggers_defects():
    """A single qubit X error should trigger 1 or 2 syndrome defects."""
    code = RotatedSurfaceCode(d=5)
    for q in range(code.n_qubits):
        errors = np.zeros(code.n_qubits, dtype=np.int8)
        errors[q] = 1
        syndrome = code.measure_syndrome(errors)
        n_defects = int(np.sum(syndrome))
        assert n_defects in (1, 2), \
            f"Single qubit error at q={q} triggered {n_defects} defects (expected 1 or 2)"


def test_same_error_applied_twice_is_trivial(code):
    rng = np.random.default_rng(1)
    errors = (rng.random(code.n_qubits) < 0.05).astype(np.int8)
    double = (errors * 2) % 2  # = zero
    syndrome = code.measure_syndrome(double)
    assert np.all(syndrome == 0)


# ── Logical failure detection ─────────────────────────────────────────────────

def test_no_error_no_failure(code):
    residual = np.zeros(code.n_qubits, dtype=np.int8)
    assert not code.logical_failure(residual)


def test_logical_x_causes_failure():
    """Applying the logical X operator (top row) should NOT cause a logical
    failure — it is a logical operation, not an error.  But flipping the
    logical Z (left column) should cause a Z-type failure, detected by
    checking parity with logical Z-bar."""
    code = RotatedSurfaceCode(d=5)
    # Flip the left column — this IS the logical Z-bar, so it should fail
    residual = np.zeros(code.n_qubits, dtype=np.int8)
    for r in range(code.d):
        residual[code.qubit_index(r, 0)] = 1
    assert code.logical_failure(residual), \
        "Left column error should cause logical failure"


def test_right_column_no_failure():
    """Flipping the right column does not traverse the logical Z-bar."""
    code = RotatedSurfaceCode(d=5)
    residual = np.zeros(code.n_qubits, dtype=np.int8)
    for r in range(code.d):
        residual[code.qubit_index(r, code.d - 1)] = 1
    # Should not cause a logical failure on the left-column logical Z-bar
    assert not code.logical_failure(residual)
