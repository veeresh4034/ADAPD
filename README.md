# ADAPD
ADAPD: Adaptive Density-Aware Phase Decoder for Quantum Error Correction on Rotated Surface Codes
# ADAPD: Adaptive Density-Aware Phase Decoder for Quantum Error Correction

## Overview

This repository contains the implementation and simulation framework for:

**ADAPD: Adaptive Density-Aware Phase Decoder for Quantum Error Correction on Rotated Surface Codes**

The project focuses on adaptive decoding strategies for quantum error correction (QEC), particularly for rotated surface codes under various quantum noise models. The repository includes decoder implementations, simulation scripts, benchmarking utilities, and reproducibility resources associated with the manuscript submitted to Quantum Reports.

---

## Features

* Adaptive Density-Aware Phase Decoder (ADAPD)
* Rotated Surface Code simulations
* Multiple quantum noise models
* Monte Carlo simulation framework
* Benchmarking and performance evaluation
* Figure generation scripts
* Modular and extensible decoder architecture
* Reproducible experimental pipeline

---

## Repository Structure

```text
ADAPD/
├── src/                 # Core source code
│   ├── decoders.py
│   ├── surface_code.py
│   ├── noise_models.py
│   ├── correction_chain.py
│   └── utils.py
│
├── scripts/             # Simulation and analysis scripts
│   ├── run_simulation.py
│   ├── analyse_results.py
│   └── generate_figures.py
│
├── data/                # Experimental datasets and statistics
│
├── figures/             # Generated plots and figures
│
├── tests/               # Unit and validation tests
│
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Simulations

Execute Monte Carlo simulations:

```bash
python scripts/run_simulation.py
```

Analyse generated results:

```bash
python scripts/analyse_results.py
```

Generate figures:

```bash
python scripts/generate_figures.py
```

---

## Noise Models

The framework supports multiple quantum noise configurations, including:

* Bit-flip noise
* Phase-flip noise
* Depolarizing noise
* Measurement noise
* Custom noise model extensions

---

## Research Objectives

This work investigates:

* Adaptive decoding for sparse quantum error patterns
* Density-aware correction mechanisms
* Improved logical error rate performance
* Efficient decoding strategies for scalable quantum systems
* Practical simulation methodologies for quantum LDPC and surface-code-inspired architectures

---

## Reproducibility

This repository includes:

* Source code
* Simulation scripts
* Experimental configurations
* Benchmark datasets
* Figure generation utilities

to support reproducibility of the results presented in the associated manuscript.

---

## Citation

If you use this repository in your research, please cite:

```text
Veeresh Kuruba, Srinivas Talabattula, and E.S. Shivaleela,
"ADAPD: Adaptive Density-Aware Phase Decoder for Quantum Error Correction on Rotated Surface Codes",
Quantum Reports, 2026.
```

---

## License

This project is released under the MIT License.

---

## Contact

For questions, collaborations, or research discussions:

Veeresh Kuruba

GitHub: https://github.com/YOUR_USERNAME

```
```
