"""
run_simulation.py
-----------------
Main Monte Carlo simulation runner for the ADAPD paper.

Evaluates MWPM, Union-Find, and ADAPD on rotated surface codes with:
    - Distances d ∈ {3, 5, 7, 9}
    - Depolarising error rates p ∈ {0.4%, …, 3.0%}
    - Z-biased noise (η=10) at p ∈ {0.5%, …, 3.0%}

Saves results to JSON files in the specified output directory.

Usage
-----
    # Full simulation 
    python scripts/run_simulation.py --output data/

    # Quick test with reduced shots (fast, fewer samples)
    python scripts/run_simulation.py --quick --output data/

    # Custom parameters
    python scripts/run_simulation.py \\
        --distances 3 5 7 \\
        --prates 0.005 0.010 0.020 \\
        --shots 10000 \\
        --seed 42 \\
        --output data/

Authors: Veeresh Kuruba, Srinivas Talabattula, E. S. Shivaleela
         Dept. of ECE, Indian Institute of Science, Bengaluru
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.surface_code import RotatedSurfaceCode
from src.noise_models import depolarising_errors, biased_errors
from src.decoders import decode_mwpm, decode_uf, decode_adapd


# ──────────────────────────────────────────────────────────────────────────────
# Simulation parameters
# ──────────────────────────────────────────────────────────────────────────────

DISTANCES     = [3, 5, 7, 9]
PRATES_DEPOL  = [0.004, 0.006, 0.008, 0.010, 0.013, 0.016, 0.020, 0.025, 0.030]
PRATES_BIASED = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]
BIAS_ETA      = 10.0

# Shot counts for full simulation (more shots at lower p for statistical power)
SHOTS_DEPOL = {
    0.004: 8000, 0.006: 7000, 0.008: 6000, 0.010: 6000,
    0.013: 5000, 0.016: 5000, 0.020: 5000, 0.025: 4000, 0.030: 4000,
}
SHOTS_BIASED = {p: 4000 for p in PRATES_BIASED}

# Shot counts for quick test
SHOTS_QUICK = {p: 500 for p in PRATES_DEPOL + PRATES_BIASED}

DECODERS = {
    'MWPM':      decode_mwpm,
    'UnionFind': decode_uf,
    'ADAPD':     decode_adapd,
}


# ──────────────────────────────────────────────────────────────────────────────
# Core simulation function
# ──────────────────────────────────────────────────────────────────────────────

def simulate_one(
    code: RotatedSurfaceCode,
    decoder_fn,
    p: float,
    n_shots: int,
    rng: np.random.Generator,
    noise: str = 'depol',
    eta: float = 10.0,
) -> Dict:
    """
    Run `n_shots` Monte Carlo trials and return a results dictionary.

    Parameters
    ----------
    code         : RotatedSurfaceCode instance
    decoder_fn   : callable — one of decode_mwpm, decode_uf, decode_adapd
    p            : physical error rate
    n_shots      : number of independent trials
    rng          : seeded NumPy random generator
    noise        : 'depol' for depolarising, 'biased' for Z-biased
    eta          : bias parameter (only used when noise='biased')

    Returns
    -------
    dict with keys:
        ler       — logical error rate (failures / n_shots)
        se        — 95% Wilson confidence interval half-width
        lm        — mean decoding latency (μs)
        lp        — 95th-percentile decoding latency (μs)
        ns        — number of shots
        nf        — number of logical failures (integer)
    """
    failures = 0
    latencies = []

    for _ in range(n_shots):
        # Sample errors
        if noise == 'depol':
            errors = depolarising_errors(code.n_qubits, p, rng)
        else:
            errors = biased_errors(code.n_qubits, p, eta, rng)

        # Measure syndrome
        syndrome = code.measure_syndrome(errors)

        # Decode
        correction, lat_s = decoder_fn(code, syndrome, p)
        latencies.append(lat_s * 1e6)  # convert to μs

        # Check logical failure
        residual = (errors + correction) % 2
        if code.logical_failure(residual):
            failures += 1

    ler = failures / n_shots
    se  = 1.96 * float(np.sqrt(max(ler * (1.0 - ler) / n_shots, 1e-10)))

    return {
        'ler': float(ler),
        'se':  float(se),
        'lm':  float(np.mean(latencies)),
        'lp':  float(np.percentile(latencies, 95)),
        'ns':  n_shots,
        'nf':  int(failures),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Serialisation helper
# ──────────────────────────────────────────────────────────────────────────────

def _serialise(results: dict) -> dict:
    """Convert results dict with int/float keys to JSON-serialisable strings."""
    out: dict = {}
    for dec in results:
        out[dec] = {}
        for d in results[dec]:
            out[dec][str(d)] = {}
            for p in results[dec][d]:
                out[dec][str(d)][str(p)] = results[dec][d][p]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    distances     = args.distances
    prates_depol  = args.prates if args.prates else PRATES_DEPOL
    prates_biased = PRATES_BIASED
    shots_map     = SHOTS_QUICK if args.quick else SHOTS_DEPOL
    shots_biased  = SHOTS_QUICK if args.quick else SHOTS_BIASED

    if args.shots:
        shots_map     = {p: args.shots for p in prates_depol}
        shots_biased  = {p: args.shots for p in prates_biased}

    total_depol  = len(DECODERS) * len(distances) * len(prates_depol)
    total_biased = len(DECODERS) * len(distances) * len(prates_biased)
    total        = total_depol + total_biased
    idx          = 0

    RD: dict = {}  # Depolarising results
    RB: dict = {}  # Biased results

    t_start = time.time()

    for dec_name, dec_fn in DECODERS.items():
        RD[dec_name] = {}
        RB[dec_name] = {}

        for d in distances:
            RD[dec_name][d] = {}
            RB[dec_name][d] = {}
            code = RotatedSurfaceCode(d)

            # ── Depolarising noise ─────────────────────────────────────
            for p in prates_depol:
                idx += 1
                ns = shots_map.get(p, 5000)
                elapsed = time.time() - t_start
                print(
                    f"[{idx:3d}/{total}] DEPOL  {dec_name:10s}  d={d}  "
                    f"p={p:.3f}  shots={ns:5d}  "
                    f"(elapsed {elapsed:.0f}s)",
                    flush=True
                )
                result = simulate_one(code, dec_fn, p, ns, rng,
                                      noise='depol')
                RD[dec_name][d][p] = result
                print(
                    f"         LER={result['ler']:.4e}  "
                    f"±{result['se']:.1e}  "
                    f"lat={result['lm']:.3f}μs  "
                    f"failures={result['nf']}",
                    flush=True
                )

            # ── Z-biased noise ─────────────────────────────────────────
            for p in prates_biased:
                idx += 1
                ns = shots_biased.get(p, 4000)
                print(
                    f"[{idx:3d}/{total}] BIASED {dec_name:10s}  d={d}  "
                    f"p={p:.3f}  eta={BIAS_ETA:.0f}  shots={ns:5d}",
                    flush=True
                )
                result = simulate_one(code, dec_fn, p, ns, rng,
                                      noise='biased', eta=BIAS_ETA)
                RB[dec_name][d][p] = result
                print(
                    f"         LER={result['ler']:.4e}  "
                    f"±{result['se']:.1e}  "
                    f"lat={result['lm']:.3f}μs",
                    flush=True
                )

    # ── Save results ───────────────────────────────────────────────────
    rd_path = Path(args.output) / 'RD.json'
    rb_path = Path(args.output) / 'RB.json'

    with open(rd_path, 'w') as f:
        json.dump(_serialise(RD), f, indent=2)
    with open(rb_path, 'w') as f:
        json.dump(_serialise(RB), f, indent=2)

    total_time = time.time() - t_start
    total_shots = sum(
        RD[d][dist][p]['ns']
        for d in RD for dist in RD[d] for p in RD[d][dist]
    )
    total_failures = sum(
        RD[d][dist][p]['nf']
        for d in RD for dist in RD[d] for p in RD[d][dist]
    )

    print(f"\n{'='*60}")
    print(f"SIMULATION COMPLETE")
    print(f"  Total time:        {total_time:.1f} s")
    print(f"  Total shots:       {total_shots:,}")
    print(f"  Total failures:    {total_failures:,}")
    print(f"  Results saved to:  {rd_path}")
    print(f"                     {rb_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run ADAPD Monte Carlo simulation.'
    )
    parser.add_argument(
        '--output', type=str, default='data/',
        help='Output directory for JSON results (default: data/)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Use reduced shot counts for a fast test run'
    )
    parser.add_argument(
        '--distances', type=int, nargs='+', default=DISTANCES,
        help='Code distances to simulate (default: 3 5 7 9)'
    )
    parser.add_argument(
        '--prates', type=float, nargs='+', default=None,
        help='Depolarising error rates to simulate (default: full list)'
    )
    parser.add_argument(
        '--shots', type=int, default=None,
        help='Fixed shot count (overrides per-p defaults)'
    )
    parser.add_argument(
        '--seed', type=int, default=2024,
        help='NumPy random seed for reproducibility (default: 2024)'
    )
    args = parser.parse_args()
    main(args)
