import platform, subprocess, sys, numpy, scipy, sklearn, numba, joblib, pandas, glob, os, re, collections
cpu = [l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')][0]
ram = round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30, 1)
S = sorted(os.path.basename(f)[:-4] for f in glob.glob('/home/claude/w/results/summaries/*.csv'))
def cnt(pat): return sum(bool(re.fullmatch(pat, s)) for s in S)
prog = []
for rep in (2, 3, 4, 5):
    for t in (2, 7):
        prog.append(f'  1. PG  T{t} rep{rep}: {cnt(rf"T{t}_(F|A|B|C|D|E|Etuned|G|L1|FZ)_PG_r{rep}_k\d+")}/100 config-fold jobs, RP {cnt(rf"T{t}_RP_PG_r{rep}_k\d+")}/10')
for rep in (2, 3, 4, 5):
    for t in (2, 7):
        prog.append(f'  2. IL  T{t} rep{rep}: {cnt(rf"T{t}_F_IL_r{rep}_k\d+")}/10 ; ORIG {cnt(rf"T{t}_F_ORIG_r{rep}_k0")}/1')
perms = collections.Counter(int(m.group(1)) for s in S if (m := re.fullmatch(r'T2_F_PG_r1_k\d+_perm(\d+)', s)))
full = sorted(p for p, c in perms.items() if c == 10)
prog.append(f'  3. Permutations complete (10/10 folds): {len(full)} {full if full else ""} ; partial: {sorted(p for p,c in perms.items() if c<10)}')
for rep in (2, 3):
    prog.append(f'  4. Learning curve rep{rep}: {cnt(rf"T\d_F_PG_r{rep}_k\d+_lc\d\d")}/60')
prog.append('  5. E13 aligned subset: done (E13_transfer/)')
txt = f"""OCT-GA revision - remaining experiments (Python FAST pipeline)
==================================================================
Platform: {platform.platform()}
CPU: {cpu}; logical cores available: {os.cpu_count()}; RAM: {ram} GB
Python {sys.version.split()[0]}; numpy {numpy.__version__}; scipy {scipy.__version__}; scikit-learn {sklearn.__version__};
numba {numba.__version__}; joblib {joblib.__version__}; pandas {pandas.__version__}
Inputs (md5-identical to the files used for OCTGA_Python_Results): features_octdl.mat, features_kermany.mat,
inventory.mat, folds.mat.

Method: unchanged FAST pipeline (GA N 20, maxIter 50, stall 15, CR 0.8, MR 0.3, k-NN k=5; ReliefF k=10
classification, W>0; SVM linear/poly2/poly3/rbf x C {{0.1,1,10}}, KernelScale sqrt(D), inner 5-fold
patient-grouped CV on balanced accuracy, inner fold 1 = GA hold-out; Platt scaling for T2; ORIG replica
GA N 20, maxIter 50, targetAcc 0.95). Every job seeded from its job id (same job_seed() as MATLAB).

Files
  predictions/<job_id>.csv  task, config, protocol, repeat, fold, perm_idx, lc_frac, image_index (1-based,
                            inventory order), patient_id, y_true, y_pred, p_pos (class-1 posterior, T2 only)
  summaries/<job_id>.csv    keys + nTrain, nTest, nPool_darknet, nPool_densenet, nPool, nSel, sel_from_darknet,
                            sel_from_densenet, GA_generations, GA_stop, GA_best_holdout_acc, svm_kernel,
                            svm_C_or_lambda (lambda for L1), sec_relieff, sec_ga, sec_svm_tune_fit, sec_predict, sec_total
  RP control: written as config "RP" (job_id T*_RP_PG_r*_k*), produced inside the matching F job; its
  sec_svm_tune_fit is the RP SVM tuning time, other timing fields empty.
  ORIG jobs: fold = 0 (image-level 10-fold CV inside the job); nTrain/nTest = 2064; svm fixed poly3 C=1.
  E13_transfer/: per-seed predictions of the 5 frozen Kermany->OCTDL models on all 2064 OCTDL images (with
  y7 and aligned_subset flag) and E13_transfer_metrics_patient_bootstrap.csv (all 2064 and aligned AMD/DME/NO
  subset, 1710 images; patient-cluster bootstrap, 2000 resamples, percentile 95% CI).

Deviations / notes
  1. libsvm max_iter capped at 300000 for every SVC fit (tuning and final). Added during the earlier permutation
     run because fits on random labels did not converge; it applies to all jobs in this package. The 5 E13 models
     (and repeat-1 results in OCTGA_Python_Results) were fitted before the cap existed.
  2. Permutation label seed = permutation index (one patient-level permutation per index, shared across the 10
     folds), as requested. Permutations 1-19 in OCTGA_Python_Results used the MATLAB scheme (seed = job seed,
     i.e. a different permutation per fold); they are not in this package. Run order: 81-99, then 20-80.
  3. Inner folds use numpy's RNG (outer folds are read from folds.mat), so inner splits differ from MATLAB.
  4. Hardware has {os.cpu_count()} core(s); joblib runs n_jobs = core count, so jobs ran sequentially. ReliefF
     results are reused between configs that make the identical ReliefF call on the identical training set
     (F/A/B/D); this does not change any value.
  5. Learning-curve patient subsets are drawn with each job's own seed, stratified by the patient's 7-class label
     (as in octga_run_job.m), so T2 and T7 subsets for the same repeat/fold/fraction differ.

Progress at packaging time
""" + "\n".join(prog) + "\n"
open('/home/claude/w/results/README.txt', 'w').write(txt); print("\n".join(prog))
