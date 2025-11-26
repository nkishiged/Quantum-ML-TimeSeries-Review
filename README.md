# Quantum Machine Learning for Time Series Forecasting - Systematic Review

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PRISMA](https://img.shields.io/badge/PRISMA-Systematic%20Review-blue)](http://prisma-statement.org/)

This repository contains the companion code and materials for the systematic review:

> **"Quantum Machine Learning Techniques for Time Series Forecasting: A Systematic Review with Emulation Focus"**

## 📚 Overview

This repository provides:
- **Reference implementations** of key QML architectures discussed in the review
- **Original Hybrid NeuroSymbolic CeNN architecture** - our conceptual contribution
- **Reproducible benchmarking framework** for quantum-inspired models
- **Complete PRISMA protocol materials** for transparency
- **Educational notebooks** demonstrating quantum ML concepts
- **Synthetic time series data** for testing and validation

## 🏗️ Repository Structure
Quantum-ML-TimeSeries-Review/
├── implementations/ # Reference implementations
│ ├── qelm/ # Quantum Extreme Learning Machines
│ ├── qnn/ # Quantum Neural Networks
│ ├── qrc/ # Quantum Reservoir Computing
│ └── cenn/ # Hybrid NeuroSymbolic CeNN (OUR CONTRIBUTION)
├── benchmarks/ # Benchmarking framework and results
├── scripts/ # Analysis and processing scripts
├── notebooks/ # Jupyter notebooks with demonstrations
├── data/ # Synthetic data and external links
└── prisma/ # PRISMA systematic review materials


## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Quantum-ML-TimeSeries-Review.git
   cd Quantum-ML-TimeSeries-Review

2. **Install dependenciesy**:
    pip install -r requirements.txt

3. **Generate synthetic data**:
    cd data/synthetic
    python generate_data.py

Basic Usage
Run EQELM demonstration:
    python implementations/qelm/eqelm_implementation.py
Explore our CeNN contribution:
    jupyter notebook notebooks/01_cenn_demonstration.ipynb
Test QRC implementation:
    python implementations/qrc/quantum_reservoir.py