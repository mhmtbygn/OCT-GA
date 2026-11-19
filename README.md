# OCT-GA: frozen deep features, ReliefF and genetic-algorithm selection for retinal OCT classification

Code for the manuscript *Automated Retinal Disease Classification from OCT Images via Deep Feature Extraction and Genetic Algorithm-based Feature Selection* (under review, revision 1).

This repository has code only. It contains no images, extracted features or results. Both datasets are public and are downloaded from their original sources:

- **OCT2017** (Kermany et al., Cell 2018), Mendeley Data v2: https://doi.org/10.17632/rscbjbr9sj.2. The development subsets were drawn from the training folder only.
- **OCTDL** (Kulyabin et al., Sci Data 2024), Zenodo: https://doi.org/10.5281/zenodo.10839556. It has 2064 images from 820 patients, and `OCTDL_labels.csv` gives the patient identifier of each image.

## Pipeline

1. Frozen ImageNet features: DarkNet53 (`avg1` + `conv53`, 2024 dimensions) and DenseNet201 (`avg_pool` + `fc1000`, 2920 dimensions).
2. ReliefF for each architecture (k = 10, keep W > 0).
3. Concatenation, then genetic-algorithm search (N = 100, at most 500 generations, CR = 0.8, MR = 0.3, stop after 100 generations without improvement). The fitness is a k-NN classifier (k = 5) on a patient-grouped 20% hold-out.
4. SVM. The kernel (linear, quadratic, cubic or Gaussian) and C (0.1, 1 or 10) are chosen by inner patient-grouped 5-fold cross-validation. The binary task adds Platt scaling.

**Evaluation on OCTDL (patient-level nested protocol).** Folds are stratified by class and grouped by patient: 10 folds × 5 repeats. Z-scoring, ReliefF, the GA, SVM tuning and calibration are all fitted inside each outer training fold.

## Repository layout

| Folder | Content |
|---|---|
| `original_submission/` | MATLAB script used for the experiments of the original submission. Selection precedes cross-validation in this script, so the revision reports its results as image-level development estimates. |
| `matlab/` | MATLAB R2025a implementation of the revision protocol. `run_all_revision.m` is the main driver, `config_revision.m` holds paths and settings, `m00`–`m08` are the stages and `lib/` holds the helpers. `run_m05_m07.m` runs the fine-tuned ResNet-18 and the latency measurement. |
| `python/` | Python implementation of the same protocol, used for the reported results. `octga.py` is the core library, `run_full.py` the resumable job runner, `aggregate_full.py` the aggregation into the result tables, and `tests/` the porting checks. |

## Experiments

| Code | Analysis |
|---|---|
| E0 | OCTDL inventory: images, patients and eyes per class |
| E2/E4/E5 | Proposed pipeline (F); ablations A, B, C, D, G; references E, Etuned, L1, FZ, a random subset of equal size (RP) and fine-tuned ResNet-18 (FT); corrected resampled t-tests with Holm adjustment |
| E3 | Protocol optimism: patient-grouped (PG), image-level (IL) and replica of the original code (ORIG) |
| E6/E7 | Selection behaviour (feature counts, Jaccard stability, kernel choice) and backbone complementarity |
| E8 | Label-permutation test of the complete pipeline (B = 49) |
| E9 | Calibration (Brier score, ECE) and sensitivity at fixed specificity |
| E10 | Learning curves |
| E11 | Measured cost and end-to-end latency |
| E12 | GA sensitivity analysis |
| E13 | Frozen OCT2017 → OCTDL transfer |
| E14 | Five repeated balanced OCT2017 subsets |

## Running

The data paths are hard-coded for the original machine: `BASE` in `python/octga.py`, `RES` in `python/run_full.py`, the paths at the top of `python/aggregate_full.py`, and `cfg.projDir` in `matlab/config_revision.m`. Edit them before running.

- **MATLAB.** Install the Deep Learning, Statistics and Machine Learning, Parallel Computing and Image Processing toolboxes, plus the DarkNet-53, DenseNet-201 and ResNet-18 support packages. Run `matlab/run_all_revision.m`: first with `SMOKE = true` as a smoke test, then the full run. Finished jobs are skipped on restart.
- **Python.** Run `pip install -r python/requirements.txt`. The features are extracted by `matlab/m01_extract_features.m` (stored in `Revision_Features/`). Then run `python python/run_full.py` (several independent worker processes may run at once; they claim jobs through lock files), followed by `python python/aggregate_full.py`.

Seeds are derived from the job identifier (`job_seed`), so every outer fold is reproducible. The reported results used Python 3.12.16, NumPy 2.5.5, SciPy 1.18.1, scikit-learn 1.9.1, Numba 0.68.1 and pandas 3.1.1 on an Intel Core i9-14900K with 64 GB RAM. The MATLAB and Python implementations agreed within 0.7 percentage points of balanced accuracy on the first repeat.
