"""
generate_figures.py
-------------------
Reproduce all 8 figures from the ADAPD paper using simulation data
stored in RD.json and RB.json.

Figures produced:
    fig1_ler_vs_p          — LER vs p (2×2 subplots, all distances)
    fig2_ler_vs_d          — LER vs d at fixed p values
    fig3_latency_vs_p      — Decoding latency vs p
    fig4_latency_scaling   — Latency scaling with d (linear + log-log)
    fig5_adapd_advantage   — ADAPD speedup and LER improvement over MWPM
    fig6_biased_noise      — LER under Z-biased noise (η=10)
    fig7_heatmap           — LER ratio heatmap (ADAPD vs MWPM and UF)
    fig8_latency_dist      — Latency mean + 95th percentile at d=7

Usage
-----
    python scripts/generate_figures.py --data data/ --output figures/

Authors: Veeresh Kuruba, Srinivas Talabattula, E. S. Shivaleela
         Dept. of ECE, Indian Institute of Science, Bengaluru
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import linregress

# ── Visual style ──────────────────────────────────────────────────────────────
COLORS  = {'MWPM': '#2166ac', 'UnionFind': '#d6604d', 'ADAPD': '#1a9641'}
MARKERS = {'MWPM': 'o',       'UnionFind': 's',        'ADAPD': '^'}
LSTYLES = {'MWPM': '-',       'UnionFind': '--',        'ADAPD': '-'}
LABELS  = {'MWPM': 'MWPM',   'UnionFind': 'Union-Find','ADAPD': 'ADAPD'}
LW, MS  = 2.0, 6

DEC_NAMES     = ['MWPM', 'UnionFind', 'ADAPD']
DISTANCES     = [3, 5, 7, 9]
PRATES        = [0.004, 0.006, 0.008, 0.010, 0.013, 0.016, 0.020, 0.025, 0.030]
PRATES_BIASED = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(data_dir: str):
    """Load RD.json (depolarising) and RB.json (biased) and convert keys."""
    def _cvt(raw):
        out = {}
        for dn in raw:
            out[dn] = {}
            for d in raw[dn]:
                out[dn][int(d)] = {}
                for p in raw[dn][d]:
                    out[dn][int(d)][float(p)] = raw[dn][d][p]
        return out

    with open(Path(data_dir) / 'RD.json') as f:
        RD = _cvt(json.load(f))
    with open(Path(data_dir) / 'RB.json') as f:
        RB = _cvt(json.load(f))
    return RD, RB


def save_fig(fig, name: str, out_dir: str):
    """Save figure as both PDF and PNG."""
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = Path(out_dir) / f'{name}.{ext}'
        fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved {name}.pdf / .png')


# ── Figure 1: LER vs p (2×2 subplots) ────────────────────────────────────────

def fig1_ler_vs_p(RD, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for idx, d in enumerate(DISTANCES):
        ax = axes[idx // 2][idx % 2]
        for dn in DEC_NAMES:
            ys = [RD[dn][d][p]['ler'] for p in PRATES]
            es = [RD[dn][d][p]['se']  for p in PRATES]
            ax.errorbar(
                np.array(PRATES) * 100, ys, yerr=es,
                color=COLORS[dn], marker=MARKERS[dn],
                label=LABELS[dn], lw=LW, ms=MS,
                ls=LSTYLES[dn], capsize=3,
            )
        ax.set_yscale('log')
        ax.set_xlabel('Physical Error Rate p (%)', fontsize=11)
        ax.set_ylabel('Logical Error Rate (LER)', fontsize=11)
        ax.set_title(f'd = {d}  (n = {d*d} qubits)', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, which='both', ls=':', alpha=0.5)
    fig.suptitle(
        'LER vs Physical Error Rate — Depolarising Noise',
        fontsize=13, fontweight='bold'
    )
    save_fig(fig, 'fig1_ler_vs_p', out_dir)


# ── Figure 2: LER vs d ────────────────────────────────────────────────────────

def fig2_ler_vs_d(RD, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, fp in zip(axes, [0.008, 0.013, 0.020]):
        for dn in DEC_NAMES:
            ys = [RD[dn][d][fp]['ler'] for d in DISTANCES]
            es = [RD[dn][d][fp]['se']  for d in DISTANCES]
            ax.errorbar(
                DISTANCES, ys, yerr=es,
                color=COLORS[dn], marker=MARKERS[dn],
                label=LABELS[dn], lw=LW, ms=MS,
                ls=LSTYLES[dn], capsize=3,
            )
        ax.set_xlabel('Code Distance d', fontsize=11)
        ax.set_ylabel('Logical Error Rate', fontsize=11)
        ax.set_title(f'p = {fp*100:.1f}%', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, ls=':', alpha=0.5)
        ax.set_xticks(DISTANCES)
    fig.suptitle(
        'LER vs Code Distance — Depolarising Noise',
        fontsize=13, fontweight='bold'
    )
    save_fig(fig, 'fig2_ler_vs_d', out_dir)


# ── Figure 3: Latency vs p ────────────────────────────────────────────────────

def fig3_latency_vs_p(RD, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, d in zip(axes, [5, 7, 9]):
        for dn in DEC_NAMES:
            ys = [RD[dn][d][p]['lm'] for p in PRATES]
            ax.plot(
                np.array(PRATES) * 100, ys,
                color=COLORS[dn], marker=MARKERS[dn],
                label=LABELS[dn], lw=LW, ms=MS, ls=LSTYLES[dn],
            )
        ax.set_xlabel('Physical Error Rate p (%)', fontsize=11)
        ax.set_ylabel('Mean Decoding Latency (μs)', fontsize=11)
        ax.set_title(f'd = {d}', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, ls=':', alpha=0.5)
    fig.suptitle(
        'Decoding Latency vs Physical Error Rate',
        fontsize=13, fontweight='bold'
    )
    save_fig(fig, 'fig3_latency_vs_p', out_dir)


# ── Figure 4: Latency scaling ─────────────────────────────────────────────────

def fig4_latency_scaling(RD, out_dir):
    fp = 0.010
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: linear scale
    ax = axes[0]
    for dn in DEC_NAMES:
        ys = [RD[dn][d][fp]['lm'] for d in DISTANCES]
        ax.plot(DISTANCES, ys, color=COLORS[dn], marker=MARKERS[dn],
                label=LABELS[dn], lw=LW, ms=MS, ls=LSTYLES[dn])
    ax.set_xlabel('Code Distance d', fontsize=11)
    ax.set_ylabel('Mean Latency (μs)', fontsize=11)
    ax.set_title(f'Latency vs d  (p = {fp*100:.1f}%)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', alpha=0.5)
    ax.set_xticks(DISTANCES)

    # Right: log-log with fitted exponents
    ax2 = axes[1]
    for dn in DEC_NAMES:
        ys = np.array([RD[dn][d][fp]['lm'] for d in DISTANCES])
        xs = np.log(np.array(DISTANCES, dtype=float))
        slope, _, _, _, _ = linregress(xs, np.log(ys))
        ax2.plot(xs, np.log(ys), color=COLORS[dn], marker=MARKERS[dn],
                 label=f'{LABELS[dn]}  α={slope:.2f}', lw=LW, ms=MS,
                 ls=LSTYLES[dn])
    ax2.set_xlabel('log(d)', fontsize=11)
    ax2.set_ylabel('log(Latency)', fontsize=11)
    ax2.set_title('Log-Log Scaling', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, ls=':', alpha=0.5)

    fig.suptitle('Latency Scaling with Code Distance',
                 fontsize=13, fontweight='bold')
    save_fig(fig, 'fig4_latency_scaling', out_dir)


# ── Figure 5: ADAPD advantage ─────────────────────────────────────────────────

def fig5_adapd_advantage(RD, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for d in DISTANCES:
        ys = [RD['MWPM'][d][p]['lm'] / RD['ADAPD'][d][p]['lm']
              for p in PRATES]
        ax.plot(np.array(PRATES) * 100, ys, marker='o',
                lw=LW, ms=MS, label=f'd={d}')
    ax.axhline(1.0, ls='--', color='gray', lw=1)
    ax.set_xlabel('Physical Error Rate p (%)', fontsize=11)
    ax.set_ylabel('Latency Speedup  (MWPM / ADAPD)', fontsize=11)
    ax.set_title('ADAPD Latency Speedup over MWPM', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', alpha=0.5)

    ax = axes[1]
    for d in DISTANCES:
        ys = [
            (RD['MWPM'][d][p]['ler'] - RD['ADAPD'][d][p]['ler'])
            / max(RD['MWPM'][d][p]['ler'], 1e-8) * 100
            for p in PRATES
        ]
        ax.plot(np.array(PRATES) * 100, ys, marker='^',
                lw=LW, ms=MS, label=f'd={d}')
    ax.axhline(0, ls='--', color='gray', lw=1)
    ax.set_xlabel('Physical Error Rate p (%)', fontsize=11)
    ax.set_ylabel('LER Improvement over MWPM (%)', fontsize=11)
    ax.set_title('ADAPD LER Improvement over MWPM', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', alpha=0.5)

    fig.suptitle('ADAPD vs MWPM — Speedup and LER Reduction',
                 fontsize=13, fontweight='bold')
    save_fig(fig, 'fig5_adapd_advantage', out_dir)


# ── Figure 6: Biased noise ────────────────────────────────────────────────────

def fig6_biased_noise(RB, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, d in zip(axes, [5, 7]):
        for dn in DEC_NAMES:
            ys = [RB[dn][d][p]['ler'] for p in PRATES_BIASED]
            es = [RB[dn][d][p]['se']  for p in PRATES_BIASED]
            ax.errorbar(
                np.array(PRATES_BIASED) * 100, ys, yerr=es,
                color=COLORS[dn], marker=MARKERS[dn],
                label=LABELS[dn], lw=LW, ms=MS,
                ls=LSTYLES[dn], capsize=3,
            )
        ax.set_yscale('log')
        ax.set_xlabel('Physical Error Rate p (%)', fontsize=11)
        ax.set_ylabel('LER (Z-Biased Noise, η=10)', fontsize=11)
        ax.set_title(f'd = {d}', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, which='both', ls=':', alpha=0.5)
    fig.suptitle('LER under Z-Biased Noise (η = 10)',
                 fontsize=13, fontweight='bold')
    save_fig(fig, 'fig6_biased_noise', out_dir)


# ── Figure 7: LER ratio heatmap ───────────────────────────────────────────────

def fig7_heatmap(RD, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_i, (baseline, title) in enumerate([
        ('MWPM',      'ADAPD / MWPM LER Ratio'),
        ('UnionFind', 'ADAPD / UF LER Ratio'),
    ]):
        mat = np.zeros((len(DISTANCES), len(PRATES)))
        for di, d in enumerate(DISTANCES):
            for pi, p in enumerate(PRATES):
                a = RD['ADAPD'][d][p]['ler']
                b = RD[baseline][d][p]['ler']
                mat[di, pi] = a / max(b, 1e-8)

        ax = axes[ax_i]
        norm = mcolors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=1.5)
        im = ax.imshow(mat, aspect='auto', cmap='RdYlGn_r', norm=norm)
        ax.set_xticks(range(len(PRATES)))
        ax.set_xticklabels([f'{p*100:.1f}' for p in PRATES],
                           fontsize=8, rotation=30)
        ax.set_yticks(range(len(DISTANCES)))
        ax.set_yticklabels([f'd={d}' for d in DISTANCES])
        ax.set_xlabel('p (%)')
        ax.set_title(title, fontweight='bold', fontsize=11)
        plt.colorbar(im, ax=ax, label='Ratio  (< 1.0 = ADAPD better)')
        for di in range(len(DISTANCES)):
            for pi in range(len(PRATES)):
                val = mat[di, pi]
                color = 'white' if val < 0.7 or val > 1.3 else 'black'
                ax.text(pi, di, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)

    fig.suptitle('LER Ratio Heatmap — ADAPD vs Baselines  (< 1 = ADAPD wins)',
                 fontsize=13, fontweight='bold')
    save_fig(fig, 'fig7_heatmap', out_dir)


# ── Figure 8: Latency distribution ───────────────────────────────────────────

def fig8_latency_dist(RD, out_dir):
    d_plot = 7
    fig, ax = plt.subplots(figsize=(9, 5))
    for dn in DEC_NAMES:
        means = [RD[dn][d_plot][p]['lm'] for p in PRATES]
        p95s  = [RD[dn][d_plot][p]['lp'] for p in PRATES]
        xp    = np.array(PRATES) * 100
        ax.fill_between(xp, means, p95s, color=COLORS[dn], alpha=0.15)
        ax.plot(xp, means, color=COLORS[dn], marker=MARKERS[dn],
                label=f'{LABELS[dn]} mean', lw=LW, ms=MS)
        ax.plot(xp, p95s,  color=COLORS[dn], ls=':', lw=1.2, ms=4,
                marker=MARKERS[dn], label=f'{LABELS[dn]} p95')
    ax.set_xlabel('Physical Error Rate p (%)', fontsize=11)
    ax.set_ylabel('Latency (μs)', fontsize=11)
    ax.set_title(
        f'Latency Distribution: Mean and 95th Percentile  (d={d_plot})',
        fontweight='bold'
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, ls=':', alpha=0.5)
    save_fig(fig, 'fig8_latency_dist', out_dir)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    print(f"Loading data from {args.data} ...")
    RD, RB = load_data(args.data)
    print(f"Generating figures → {args.output}/")

    fig1_ler_vs_p(RD, args.output)
    fig2_ler_vs_d(RD, args.output)
    fig3_latency_vs_p(RD, args.output)
    fig4_latency_scaling(RD, args.output)
    fig5_adapd_advantage(RD, args.output)
    fig6_biased_noise(RB, args.output)
    fig7_heatmap(RD, args.output)
    fig8_latency_dist(RD, args.output)

    print("\nAll 8 figures generated successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate all ADAPD paper figures from simulation data.'
    )
    parser.add_argument(
        '--data', type=str, default='data/',
        help='Directory containing RD.json and RB.json (default: data/)'
    )
    parser.add_argument(
        '--output', type=str, default='figures/',
        help='Output directory for figures (default: figures/)'
    )
    args = parser.parse_args()
    main(args)
