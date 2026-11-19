OCT-GA revision - Python pipeline (code only, no data)
=====================================================

FILES
  octga.py        Core library: data loading (load_data, load_kermany), job_seed, strat_group_kfold,
                  ReliefF / RReliefF (numba: _l1, _relief_cls, _relief_reg, relieff, relieff_pos),
                  GA (knn_err, ga_select), SVM with precomputed kernels (KSVM, svm_tune_fit),
                  L1 logistic (fit_l1, lin_tune_fit), label permutation (permute_group_labels),
                  run_job (one outer fold of any config/protocol), run_orig (ORIG replica),
                  run_transfer (Kermany -> OCTDL), save. Default settings in the CFG dict.
  runner.py       First run (repeat 1), stages P1-P6, writes pickles to py/jobs/.
  analyze.py      Aggregates runner.py pickles + MATLAB reference jobs -> summary.pkl (all metrics, CIs, tests).
  export.py       summary.pkl -> /mnt/user-data/outputs/OCTGA_Python_Results.xlsx.
  runner2.py      Remaining experiments (repeats 2-5, IL/ORIG, permutations 20-99, learning curve rep 2-3);
                  one predictions CSV + one summary CSV per job, resumable, joblib parallel, zips results.
  e13_aligned.py  E13 aligned subset (AMD/DME/NO) + patient-cluster bootstrap from the 5 Kermany models.
  readme.py       Writes results/README.txt (platform, versions, progress) for the results package.
  tests/t1-t5.py  Sanity checks and timing benchmarks used while porting (folds/seed check, ReliefF,
                  single jobs, numba L1 distance, L1 logistic timing).

EXPERIMENT -> CODE
  E2/E4/E5 (configs F,A,B,C,D,E,Etuned,G,L1,FZ; PG)  run_job(...)  | runner.py P1 (rep 1), runner2.py (rep 2-5)
  E7 RP control                                       inside run_job for conf='F', PG, no lc/perm
  E3 protocols IL / ORIG                              run_job(prot='IL'), run_orig  | runner.py P2, runner2.py
  E6 selection stability, E7 complementarity,
  E9 calibration, E11 cost (classifier stages)        computed in analyze.py from job outputs
  E8 permutation                                      run_job(permIdx=p)  | runner.py P6 (1-19), runner2.py (20-99)
  E10 learning curve                                  run_job(lcFrac=f)   | runner.py P5 (rep 1), runner2.py (rep 2-3)
  E12 GA sensitivity                                  run_job(variant=...) via VARIANTS  | runner.py P4
  E13 transfer Kermany -> OCTDL                       run_transfer  | runner.py P3; aligned subset: e13_aligned.py
  Not implemented: E5-iii fine-tuned ResNet-18, E11 end-to-end CNN latency (need images / CNN weights).

HOW TO RUN (from the folder containing these files)
  python3 runner.py P1 P2 P3 P4 P5 P6      # repeat-1 stages, any subset of stage names
  python3 analyze.py && python3 export.py   # tables / Excel for runner.py outputs
  python3 runner2.py                        # remaining experiments, resumes automatically
  python3 e13_aligned.py                    # needs jobs/T2_TRANSFER_KtoO_s*.pkl from runner.py P3

INPUT / OUTPUT PATHS (edit these lines)
  octga.py     BASE = '/home/claude/w'
               load_data():    BASE/Revision_Features/features_octdl.mat, BASE/Revision_Features/inventory.mat,
                               BASE/Revision_Results/folds.mat
               load_kermany(): BASE/Revision_Features/features_kermany.mat (labels read from inventory.mat)
  runner.py    OUT = '/home/claude/w/py/jobs'
  runner2.py   RES = '/home/claude/w/results'; zip -> /mnt/user-data/outputs/results.zip
  analyze.py   reads jobs/*.pkl and MATLAB jobs in /home/claude/w/Revision_Results/jobs/
  export.py    writes /mnt/user-data/outputs/OCTGA_Python_Results.xlsx

SETTINGS USED (FAST mode)
  GA: N 20, maxIter 50, stall 15, CR 0.8, MR 0.3, k-NN k=5 (set in runner.py / runner2.py; CFG holds the
      full-config defaults N 100, maxIter 500, stall 100). ORIG: N 20, maxIter 50, targetAcc 0.95, cubic SVM C=1.
  ReliefF k=10 classification, keep W>0, range-scaled L1 distances; RReliefF (ORIG) sigma 50.
  SVM: linear/poly2/poly3/rbf x C {0.1,1,10}, KernelScale sqrt(D), inner 5-fold patient-grouped CV on BAcc,
       inner fold 1 = GA hold-out; Platt scaling (T2). L1: lambda logspace(-5,-1,9), C = 1/(lambda*n).
  E12 variants: MR 0.1, MR 0.01, MR 1/D, N 50.

VERSION NOTES
  This octga.py is the current version. Two changes were made during the session:
   (1) libsvm max_iter=300000 cap on every SVC fit (added from permutation 10 of the first run on);
   (2) run_job(permSeedMode='perm') option and extra summary fields (nPoolDark/nPoolDense, RP details,
       ORIG timings) used by runner2.py. runner.py uses the default permSeedMode='job' (seed = job seed).
  Repeat-1 results, E12, E13 models and permutations 1-9 were produced before the cap existed.
