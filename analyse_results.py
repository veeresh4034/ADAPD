"""
analyse_results.py
------------------
Statistical analysis of ADAPD simulation results.
Prints summary tables, computes scaling exponents, and saves stats.json.

Usage
-----
    python scripts/analyse_results.py --data data/

Authors: Veeresh Kuruba, Srinivas Talabattula, E. S. Shivaleela
         Dept. of ECE, Indian Institute of Science, Bengaluru
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import linregress

DISTANCES = [3, 5, 7, 9]
PRATES    = [0.004, 0.006, 0.008, 0.010, 0.013, 0.016, 0.020, 0.025, 0.030]
PRATES_B  = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]
DEC_NAMES = ['MWPM', 'UnionFind', 'ADAPD']


def load(data_dir):
    def _cvt(raw):
        out = {}
        for dn in raw:
            out[dn] = {}
            for d in raw[dn]:
                out[dn][int(d)] = {}
                for p in raw[dn][d]:
                    out[dn][int(d)][float(p)] = raw[dn][d][p]
        return out
    with open(Path(data_dir) / 'RD.json') as f: RD = _cvt(json.load(f))
    with open(Path(data_dir) / 'RB.json') as f: RB = _cvt(json.load(f))
    return RD, RB


def main(args):
    RD, RB = load(args.data)

    # ── 1. Full LER table ────────────────────────────────────────────────
    print("\n" + "="*80)
    print("TABLE 1: LOGICAL ERROR RATE — DEPOLARISING NOISE")
    print("="*80)
    header = f"{'Dec':10s} {'d':3s} " + "  ".join([f"p={p*100:.1f}%" for p in PRATES])
    print(header)
    print("-" * len(header))
    for dn in DEC_NAMES:
        for d in DISTANCES:
            row = f"{dn:10s} {d:3d} "
            for p in PRATES:
                v = RD[dn][d][p]
                row += f"  {v['ler']:.3e}"
            print(row)
        print()

    # ── 2. Latency table ─────────────────────────────────────────────────
    print("\n" + "="*80)
    print("TABLE 2: MEAN DECODING LATENCY (μs) — DEPOLARISING NOISE")
    print("="*80)
    for d in DISTANCES:
        print(f"\n  d = {d}")
        print(f"  {'p':7s}  {'MWPM':10s}  {'UF':10s}  {'ADAPD':10s}  {'Speedup (MWPM/ADAPD)':22s}")
        for p in [0.008, 0.010, 0.016, 0.020, 0.030]:
            m = RD['MWPM'][d][p]['lm']
            u = RD['UnionFind'][d][p]['lm']
            a = RD['ADAPD'][d][p]['lm']
            print(f"  {p*100:.1f}%    {m:8.2f}μs  {u:8.2f}μs  {a:8.2f}μs  {m/a:.2f}×")

    # ── 3. Latency scaling exponents ─────────────────────────────────────
    print("\n" + "="*80)
    print("TABLE 3: LATENCY SCALING EXPONENTS  (lat ∝ d^α at p=1.0%)")
    print("="*80)
    fp = 0.010
    exponents = {}
    for dn in DEC_NAMES:
        ys = np.array([RD[dn][d][fp]['lm'] for d in DISTANCES])
        xs = np.log(np.array(DISTANCES, dtype=float))
        slope, intercept, r, _, _ = linregress(xs, np.log(ys))
        exponents[dn] = float(slope)
        lats = [f"{RD[dn][d][fp]['lm']:.2f}" for d in DISTANCES]
        print(f"  {dn:12s}  α={slope:.2f}  (R²={r**2:.3f})  "
              f"lats: {' / '.join(lats)} μs")

    # ── 4. Aggregate improvements ────────────────────────────────────────
    print("\n" + "="*80)
    print("TABLE 4: AGGREGATE PERFORMANCE vs MWPM")
    print("="*80)
    for dn in ['UnionFind', 'ADAPD']:
        ler_impr, lat_spd = [], []
        for d in DISTANCES:
            for p in PRATES:
                m_ler = RD['MWPM'][d][p]['ler']
                a_ler = RD[dn][d][p]['ler']
                m_lat = RD['MWPM'][d][p]['lm']
                a_lat = RD[dn][d][p]['lm']
                if m_ler > 0:
                    ler_impr.append((m_ler - a_ler) / m_ler * 100)
                lat_spd.append(m_lat / a_lat)
        print(f"\n  {dn}:")
        print(f"    LER improvement vs MWPM: "
              f"mean={np.mean(ler_impr):.1f}%  "
              f"min={min(ler_impr):.1f}%  max={max(ler_impr):.1f}%")
        print(f"    Latency speedup vs MWPM: "
              f"mean={np.mean(lat_spd):.2f}×  "
              f"min={min(lat_spd):.2f}×  max={max(lat_spd):.2f}×")

    # ── 5. Key single points ─────────────────────────────────────────────
    print("\n" + "="*80)
    print("KEY RESULT: d=9, p=1.0%")
    print("="*80)
    for dn in DEC_NAMES:
        v = RD[dn][9][0.010]
        print(f"  {dn:12s}  LER={v['ler']:.4e}  lat={v['lm']:.2f}μs  failures={v['nf']}")

    # ── 6. Biased noise summary ──────────────────────────────────────────
    print("\n" + "="*80)
    print("TABLE 5: LER UNDER Z-BIASED NOISE (η=10)")
    print("="*80)
    for d in [5, 7]:
        print(f"\n  d = {d}")
        print(f"  {'p':7s}  {'MWPM':12s}  {'UF':12s}  {'ADAPD':12s}  {'Winner':10s}")
        for p in PRATES_B:
            vals = {dn: RB[dn][d][p]['ler'] for dn in DEC_NAMES}
            best = min(vals, key=vals.get)
            dn_label = {'MWPM': 'MWPM', 'UnionFind': 'UF', 'ADAPD': 'ADAPD'}
            print(f"  {p*100:.1f}%    {vals['MWPM']:.3e}    {vals['UnionFind']:.3e}    "
                  f"{vals['ADAPD']:.3e}    {dn_label[best]}")

    # ── 7. Save stats.json ───────────────────────────────────────────────
    stats = {'exponents': exponents}
    for dn in DEC_NAMES:
        ler_impr, lat_spd = [], []
        for d in DISTANCES:
            for p in PRATES:
                m = RD['MWPM'][d][p]['ler']
                a = RD[dn][d][p]['ler']
                if m > 0:
                    ler_impr.append((m - a) / m * 100)
                lat_spd.append(RD['MWPM'][d][p]['lm'] / RD[dn][d][p]['lm'])
        stats[f'{dn}_mean_ler_improvement_vs_mwpm'] = float(np.mean(ler_impr))
        stats[f'{dn}_mean_latency_speedup_vs_mwpm'] = float(np.mean(lat_spd))

    stats_path = Path(args.data) / 'stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Print summary statistics from ADAPD simulation data.'
    )
    parser.add_argument(
        '--data', type=str, default='data/',
        help='Directory containing RD.json and RB.json (default: data/)'
    )
    args = parser.parse_args()
    main(args)
