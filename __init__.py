"""
ADAPD — Adaptive Density-Aware Phase Decoder for Quantum Error Correction

Package structure:
    src.surface_code      — Rotated surface code geometry
    src.noise_models      — Depolarising and biased Pauli noise
    src.utils             — Union-Find data structure and helpers
    src.correction_chain  — Matching-to-correction chain translation
    src.decoders          — MWPM, Union-Find, and ADAPD decoders
"""

from .surface_code import RotatedSurfaceCode
from .noise_models import depolarising_errors, biased_errors, effective_x_rate
from .utils import UnionFind, manhattan_distance, log_likelihood_weight
from .correction_chain import pairs_to_correction, drop_boundary_nearest
from .decoders import decode_mwpm, decode_uf, decode_adapd

__all__ = [
    "RotatedSurfaceCode",
    "depolarising_errors",
    "biased_errors",
    "effective_x_rate",
    "UnionFind",
    "manhattan_distance",
    "log_likelihood_weight",
    "pairs_to_correction",
    "drop_boundary_nearest",
    "decode_mwpm",
    "decode_uf",
    "decode_adapd",
]
