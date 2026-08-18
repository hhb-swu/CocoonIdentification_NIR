# Dual-Representation NIR Cocoon Identification

## Overview

This repository provides the data organization, recurrence-plot (RP) generation code, released model weights, and test/evaluation code associated with a dual-representation weighted-fusion framework for near-infrared (NIR) spectral identification of fresh-cocoon sex and imprinted-dead cocoons.

The framework combines:

- **1D spectral-sequence representation**, using vector-normalized NIR spectra;
- **2D recurrence-plot representation**, generated from the corresponding 1D spectra;
- **Dual-representation weighted fusion**, which combines the two complementary representations for three-class identification.

The repository is intended to support transparent evaluation of the finalized model and reproduction of the RP-generation procedure. It does **not** include the complete training, hyperparameter-search, or ablation-study pipeline.

---

## Repository Structure

```text
<repository>/
├── data/
│   ├── Spectral_Sequences/
│   │   └── TestSet/
│   │       ├── Day1/
│   │       │   ├── 0/
│   │       │   ├── 1/
│   │       │   └── 2/
│   │       ├── Day2/
│   │       ├── ...
│   │       └── Day10/
│   │
│   └── Recurrence_Plots/
│       └── TestSet/
│           ├── Day1/
│           │   ├── 0/
│           │   ├── 1/
│           │   └── 2/
│           ├── Day2/
│           ├── ...
│           └── Day10/
│
├── weights/
│   ├── Optimal_1D-CNN_Weights.pth
│   ├── Optimal_2D-CNN_Weights.pth
│   └── DualRepresentation_Weighted-Fusion_Network_Weights.pth
│
├── results/
│
├── DualRepresentation_WeightedFusion_Test.py
├── DualRepresentation_WeightedFusion_Test_Path_Guide.txt
├── Generate_Recurrence_Plots.py
└── Recurrence_Plot_Generation_Path_Guide.txt
```

All paths used by the released scripts are repository-relative. No machine-specific absolute Windows path is required.

---

## Class Labels

The class encoding is consistent throughout the repository:

| Label | Class |
|---|---|
| `0` | Female cocoon |
| `1` | Male cocoon |
| `2` | Imprinted-dead cocoon |

---

## Released Test Set

The final test set contains **868 samples**.

### Day1-Day9

For each day:

- 36 female-cocoon samples (`0`)
- 36 male-cocoon samples (`1`)
- 22 imprinted-dead-cocoon samples (`2`)

### Day10

- 0 female-cocoon samples
- 0 male-cocoon samples
- 22 imprinted-dead-cocoon samples (`2`)

### Total

| Class | Number of samples |
|---|---:|
| Female cocoon | 324 |
| Male cocoon | 324 |
| Imprinted-dead cocoon | 220 |
| **Total** | **868** |

---

## 1D Spectral Sequences

The released 1D spectral sequences are stored as NumPy `.npy` files.

Each spectrum contains:

- **922 spectral variables**
- expected shape: `(922,)` or `(1, 922)`
- data type used by the evaluation script: `float32`

The released spectra have already undergone the vector-normalization preprocessing used for the finalized model.

**No additional vector normalization is applied by the released test script or RP-generation script.**

---

## Recurrence-Plot Generation

Recurrence plots can be generated directly from the released 1D spectral sequences using:

```bash
python Generate_Recurrence_Plots.py
```

Default input:

```text
data/Spectral_Sequences/TestSet
```

Default output:

```text
data/Recurrence_Plots/TestSet
```

The script preserves the complete directory structure and file names.

Example:

```text
Input:
data/Spectral_Sequences/TestSet/Day1/0/000001.npy

Output:
data/Recurrence_Plots/TestSet/Day1/0/000001.npy
```

### RP Parameters

The released RP-generation procedure uses:

- spectral length: `922`
- embedding dimension: `2`
- delay: `6`
- reconstructed phase-space length: `916`
- Euclidean distance matrix
- min-max scaling to `[0, 1]`
- original RP size: `916 × 916`
- resize scale: `0.5`
- final RP size: `458 × 458`
- resize interpolation: OpenCV `INTER_AREA`

No additional spectral preprocessing is performed during RP generation.

For additional path and usage details, see:

```text
Recurrence_Plot_Generation_Path_Guide.txt
```

---

## Model Weights

The `weights/` directory contains the released checkpoints required by the test script:

```text
Optimal_1D-CNN_Weights.pth
Optimal_2D-CNN_Weights.pth
DualRepresentation_Weighted-Fusion_Network_Weights.pth
```

The evaluation script loads these checkpoints using strict PyTorch state-dictionary matching.

---

## Test and Evaluation

Run the finalized dual-representation model with:

```bash
python DualRepresentation_WeightedFusion_Test.py
```

The script automatically uses:

```text
1D spectra:
data/Spectral_Sequences/TestSet

2D recurrence plots:
data/Recurrence_Plots/TestSet

Model weights:
weights

Output directory:
results
```

The 1D and 2D inputs are paired only when they have:

1. the same `Day` folder;
2. the same class folder;
3. exactly the same file stem.

For example:

```text
data/Spectral_Sequences/TestSet/Day1/0/000001.npy
data/Recurrence_Plots/TestSet/Day1/0/000001.npy
```

are treated as one paired sample.

The released evaluation script checks for unmatched files and verifies the expected test-set distribution before model evaluation.

---

## Evaluation Outputs

The test script reports:

- overall accuracy;
- class-specific accuracy;
- confusion matrix;
- one-vs-rest ROC curves;


By default, figures are saved to:

```text
results/
```

Typical output files include:

```text
DualRepresentationWeightedFusion_ConfusionMatrix.png
DualRepresentationWeightedFusion_ROC.png
DualRepresentationWeightedFusion_UMAP.png
```

UMAP analysis requires the `umap-learn` package. It can be skipped with:

```bash
python DualRepresentation_WeightedFusion_Test.py --skip-umap
```

The evaluation batch size can also be specified from the command line, for example:

```bash
python DualRepresentation_WeightedFusion_Test.py --batch-size 1
```

---

## Required Python Packages

The main dependencies are:

```text
numpy
torch
matplotlib
scikit-learn
opencv-python
umap-learn
```

`umap-learn` is required only for UMAP visualization.

---

## Recommended Workflow

### 1. Check the released 1D spectra

Confirm that:

```text
data/Spectral_Sequences/TestSet/
```

contains the expected `Day1`-`Day10` and class folders.

### 2. Generate recurrence plots

```bash
python Generate_Recurrence_Plots.py
```

### 3. Run model evaluation

```bash
python DualRepresentation_WeightedFusion_Test.py
```

or, without UMAP:

```bash
python DualRepresentation_WeightedFusion_Test.py --skip-umap
```

### 4. Check the results

Evaluation figures are written to:

```text
results/
```

---

## Important Notes

1. The released spectra are already vector-normalized.
2. The RP-generation script does not repeat vector normalization.
3. The test script does not perform additional spectral preprocessing.
4. The 1D and 2D files must remain strictly paired by Day, class, and file name.
5. The released evaluation script is intended for predictive-performance evaluation of the finalized model.
6. The separate network-inference timing protocol reported in the manuscript is **not** reproduced by this script.
7. The complete training, hyperparameter-search, and ablation-study code is not included in this repository.

---

## File-Specific Guides

For more detailed instructions, refer to:

```text
DualRepresentation_WeightedFusion_Test_Path_Guide.txt
Recurrence_Plot_Generation_Path_Guide.txt
```

These files provide additional information on path configuration, expected input structure, and script usage.
