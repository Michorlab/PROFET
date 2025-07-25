# Particle-based Reconstruction Of generative Force-matched Expression Trajectories (PROFET)
**PROFET** (Particle-based Reconstruction Of generative Force-matched Expression Trajectories) is a computational framework for reconstructing continuous gene expression dynamics from static, time-stamped single-cell RNA sequencing (scRNA-seq) data. Unlike conventional methods that rely on discrete timepoints or assume linear transitions, PROFET models cell state evolution as a two-step generative process grounded in mathematical formalisms.

## Method Overview
1. **Step 1: Particle-based transport generation**
PROFET constructs transport plans between empirical distributions at different timepoints using a **Lipschitz-regularized** gradient flow formulation, yielding temporally smooth and distribution-consistent particle trajectories.
2. **Step 2: Force-matched velocity field learning**
A **time-dependent velocity field** is estimated using **force matching**, resulting in a global vector field that best explains the sampled particle flows across time.

This two-stage approach enables nonlinear, continuous trajectory reconstruction from sparsely sampled gene expression snapshots.

PROFET has been validated on both synthetic and experimental datasets and applied to uncover treatment-induced heterogeneity in breast cancer. By recovering dynamic expression trajectories from static scRNA-seq data, PROFET provides a scalable and principled tool for modeling cell state transitions in development, disease, and therapeutic response.


## How to Run PROFET: Notebook Workflow and File Structure

To use PROFET and perform downstream analysis, follow the instructions below. Note that two key notebooks operate **in parallel**, both using the same raw input data.

---

### 📁 Input Data Structure

All datasets are stored in the `data/` folder. Each dataset is organized into its own subfolder and should include:

- A **gene expression matrix** (e.g., `.txt`)
- A corresponding **cell time annotation file**

These files serve as inputs for both notebooks described below.

---

### 1. Run `Preprocess_datasets.ipynb`  *(for downstream analysis)*

This notebook performs **PCA** and saves the preprocessed version of each dataset.

- **Input:** Raw gene expression matrix and time annotations from the `data/` subfolders.
- **Output:** Preprocessed dataset saved in `.pkl` format in the `data/` folder.
- These `.pkl` files are required for **visualization and trajectory validation**.

---

### 2. Run `PROFET_full_pipeline.ipynb`  *(for trajectory reconstruction)*

This notebook runs the full PROFET pipeline:
- **Step 1:** Load raw gene expression matrix and time annotations (from `data/` subfolders)
- **Step 2:** Run GPA to generate particle-based trajectories
- **Step 3:** Apply force-matching to estimate the time-dependent vector field

- **Output:** A `.pickle` file containing the reconstructed single-cell trajectory, saved in the `assets/` folder.

⚠️ **Note:** This notebook also uses the raw data from the `data/` subfolders (not the `.pkl` files from preprocessing). It runs independently of `Preprocess_datasets.ipynb`, but both notebooks must be executed for full analysis.

---

### ✅ After Running Both Notebooks

At this stage, you should have:
- A **preprocessed `.pkl` file** (from `Preprocess_datasets.ipynb`) in the `data/` folder
- A **reconstructed `.pickle` trajectory file** (from `PROFET_full_pipeline.ipynb`) in the `assets/` folder

These two files are used together for:
- Visualization
- Subtrajectory classification
- Validation and downstream analysis

You can now proceed to:
- `Trajectory_Visualization_and_Subtrajectory_Analysis.ipynb` (for plotting and clustering trajectories)



## Installation
```bash
git clone https://github.com/yourusername/PROFET.git
cd PROFET
pip install -r requirements.txt
```

## Project Structure

```
PROFET/
├── scripts/                            # Main training and utility scripts
│   ├── train_time_dep_vectorfields.py  # Step 2: train time-dependent vector fields (force matching)
│   ├── GPA_NN/                          # Step 1: Generative Particle Algorithm (GPA) implementation
│   │   └── GPA_NN.py                    # Main GPA training script
│   └── util/                            # Helper functions: data loading, plotting, etc.
│
├── configs/                            # YAML configuration files
│   └── gpa_config.yaml                 # Example configuration for GPA
│
├── notebooks/                          # Jupyter notebooks for complete workflows
│   └── example_pipeline.ipynb          # End-to-end example pipeline
│
├── data/                               # Input datasets and preprocessed files
│
├── assets/                             # Output directory for results and plots
│
└── README.md                           # Project overview and usage instructions
```


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
