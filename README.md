# spatiotemporal-simulation-prediction-SIF

This repository provides the official implementation of the models presented in the paper:

**“A Multi-Source Spatiotemporal Prediction Framework for Large-Scale Heterogeneous Grassland SIF”**
https://doi.org/10.1080/15481603.2025.2573565

The proposed framework integrates deep learning, statistical modeling, and interpretable machine learning to achieve accurate and stable spatiotemporal prediction of solar-induced chlorophyll fluorescence (SIF) across heterogeneous grassland ecosystems.

---

## 🔍 Overview

Accurate prediction of SIF is critical for understanding vegetation productivity and ecosystem responses to climate change. However, existing methods often struggle to simultaneously address:

- Long-term temporal dependencies
- Nonlinear interactions among climatic and anthropogenic drivers
- Large-scale spatial heterogeneity

To overcome these limitations, this repository implements a **hybrid spatiotemporal modeling framework** that couples:

- Temporal memory accumulation (LSTM)
- Global temporal attention (Transformer)
- Residual volatility modeling (GARCH)
- Interpretable nonlinear representation (KAN)

---

## 🧠 Implemented Models

### Baseline Models
- Random Forest (RF)
- Long Short-Term Memory (LSTM)
- Transformer
- GARCH
- Kolmogorov–Arnold Network (KAN)

### Hybrid Models
- LSTM + RF
- LSTM + GARCH
- LSTM + KAN
- **LSTM + Transformer (Main Model)**

Each model follows a unified input–output interface for fair comparison.

---

## 📁 Repository Structure

```text
src/
 ├── data_utils.py        # Time-series construction
 ├── metrics.py           # Evaluation metrics
 ├── models/              # All baseline and hybrid models
 ├── train.py             # Unified training pipeline
 └── evaluate.py          # Model evaluation
