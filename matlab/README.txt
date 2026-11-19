OCT-GA R1 REVISION - EXPERIMENT SCRIPTS
=======================================

Run
---
1. Open MATLAB R2025a (R2024a or newer) in this folder.
2. Open run_all_revision.m, keep SMOKE = true, press Run (smoke test).
3. Check D:\MBayrakOCTGA\Revision_Results_SMOKE\REPORT.txt and the log; send both back.
4. Set SMOKE = false, run again (full run). It can be stopped (Ctrl+C) and restarted;
   finished jobs in Revision_Results\jobs are skipped.

Requirements: Deep Learning, Statistics and Machine Learning, Parallel Computing and
Image Processing Toolboxes; support packages "Deep Learning Toolbox Model for
DarkNet-53 Network", "... DenseNet-201 Network" and "... ResNet-18 Network"
(Home > Add-Ons). Data paths are set in config_revision.m.

What is run (OCTDL, 2064 images, patient_id from OCTDL_labels.csv)
-------------------------------------------------------------------
Tasks: T2 = Normal vs Diseased, T7 = 7 classes.
Outer CV: 10 folds x 5 repeats, patient-grouped and stratified (strat_group_kfold).
Inside each outer TRAINING set only: z-score, ReliefF (k = 10, W > 0, classification),
GA (N 100, maxIter 500, CR 0.8, MR 0.3, stall 100; k-NN k = 5 fitness on an inner
patient-grouped 20% hold-out), SVM kernel {linear, poly2, poly3, rbf} x C {0.1, 1, 10}
chosen by inner 5-fold patient-grouped CV (balanced accuracy), Platt scaling.

E0  inventory (images / patients / eyes per class, duplicates, fold composition)
E1  feature extraction once (label-free), timing
E2  proposed pipeline (F)
E3  protocol comparison: PG (patient-grouped nested), IL (image-level nested),
    ORIG (replica of Kod\GA_CNN_FeatureSelection_Code.m: global z-score + RReliefF,
    GA with targetAcc 0.95 and the fold-1 hold-out, cubic SVM, image-level CV)
E4  ablation A-G (+E, Etuned) with corrected resampled t-tests (Nadeau-Bengio) + Holm
E5  baselines: L1 logistic (L1), single frozen CNN + linear SVM (FZ), fine-tuned ResNet-18 (FT)
E6  selection counts per stage/backbone, Jaccard stability, ReliefF weight distribution, GA curves
E7  backbone complementarity (error overlap, McNemar, Q) and same-size random-subset control (RP)
E8  label-permutation test (B = 100, patient level, T2)
E9  ROC/PR AUC, Brier, ECE, reliability, sensitivity at 90/95% specificity (T2)
E10 learning curve (25/50/75/100% of training patients)
E11 cost: per-stage times, end-to-end single-image latency, peak RAM, hardware
E12 GA sensitivity (MR 0.1, 0.01, 1/D; N 50)
E13 frozen transfer: pipeline fitted on local Kermany binary images, applied unchanged to OCTDL

Outputs (D:\MBayrakOCTGA\Revision_Results)
-----------------------------------------
REPORT.txt, summary_tables.xlsx, tables\*.csv, figures\*.png (600 dpi),
jobs\*.mat (one per outer fold), models\, E11_latency.mat, log_*.txt.
Feature cache shared by smoke and full runs: D:\MBayrakOCTGA\Revision_Features.

Optional: to report latency on the laptop named in the manuscript, copy this folder,
Revision_Features and Revision_Results\models to the laptop and run
    addpath(pwd, fullfile(pwd,'lib')); cfg = config_revision(false); m07_latency(cfg)
