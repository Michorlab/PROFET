# Particle-based Reconstruction Of generative Force-matched Expression Trajectories (PROFET)
**PROFET** (Particle-based Reconstruction Of generative Force-matched Expression Trajectories) is a computational framework for reconstructing continuous gene expression dynamics from static, time-stamped single-cell RNA sequencing (scRNA-seq) data. Unlike conventional methods that rely on discrete timepoints or assume linear transitions, PROFET models cell state evolution as a two-step generative process grounded in mathematical formalisms.

## Method Overview
1. **Step 1: Particle-based transport generation**
PROFET constructs transport plans between empirical distributions at different timepoints using a **Lipschitz-regularized** gradient flow formulation, yielding temporally smooth and distribution-consistent particle trajectories.
2. **Step 2: Force-matched velocity field learning**
A **time-dependent velocity field** is estimated using **force matching**, resulting in a global vector field that best explains the sampled particle flows across time.

This two-stage approach enables nonlinear, continuous trajectory reconstruction from sparsely sampled gene expression snapshots.

PROFET has been validated on both synthetic and experimental datasets and applied to uncover treatment-induced heterogeneity in breast cancer. By recovering dynamic expression trajectories from static scRNA-seq data, PROFET provides a scalable and principled tool for modeling cell state transitions in development, disease, and therapeutic response.



## Installation
```bash
git clone https://github.com/yourusername/PROFET.git
cd PROFET
pip install -r requirements.txt
```

## Getting started
Here’s how to run a basic example:


## Project Structure
PROFET/
├── scripts/                            # Main training and utility scripts
│   ├── train_time_dep_vectorfields.py  # Training script for time-dependent vector fields (force-matching, step 2)
│   ├── GPA_NN/                          # GPA (Generative Particle Algorithm) implementation
│   │   └── GPA_NN.py                    # Training script for GPA (step 1)
│   └── util/                            # Utilities and helper functions for training
│       └── ...                          # e.g., data loading, argparsing, plotting
│
├── configs/                            # YAML configuration files for training and model setup
│   └── gpa_config.yaml                 # Example config for GPA_NN
│
├── notebooks/                          # Jupyter notebooks for end-to-end workflows or experiments
│   └── example_pipeline.ipynb          # 
│
├── data/                               # Datasets and its preprocessed files 
│
├── assets/                             # Output directory for results and plots
│
└── README.md                           # Project overview and usage instructions


## Examples

* Epithelial-Mesenchymal Transition (EMT)
* Synthetic Trajectories
* Stem Cell Differentiation

## Citation
If you use PROFET in your research, please cite:
```bibtex
@article{cheng2025profet,
  title={PROFET Predicts Continuous Gene Expression Dynamics
from scRNA-seq Data to Elucidate Resistance to Cancer Therapy},
  author={},
  journal={Preprint},
  year={2025}
}
```
