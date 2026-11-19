"""Aggregate all OCT-GA revision results (FULL GA Python run) into one workbook.
Re-run any time:  python aggregate_full.py   (permutation results update automatically)."""
import os, glob, pickle, platform, numpy as np, pandas as pd, scipy.io as sio, h5py
from scipy.stats import t as tdist, binom
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
R = 'D:/MBayrakOCTGA/Revision_Results_Python_FULL'; P = f'{R}/predictions'; SM = f'{R}/summaries'
OUT = 'D:/MBayrakOCTGA/R1_Revizyon/OCTGA_R1_FULL_GA_sonuclar.xlsx'
CFGS = ['F', 'A', 'B', 'C', 'D', 'E', 'Etuned', 'G', 'L1', 'FZ', 'RP']
NAMES = {2: ['Diseased', 'Normal'], 7: ['AMD', 'DME', 'ERM', 'NO', 'RAO', 'RVO', 'VID']}
BOOT = 2000

def kind(n):
    return ('perm' if '_perm' in n else 'lc' if '_lc' in n else 'ks' if '_KS' in n else 'orig' if '_ORIG_' in n else
            'il' if '_IL_' in n else 'e12' if any(t in n for t in ('_MR', '_N50', '_FAST')) else 'rp' if '_RP_' in n else 'pg')
def load(folder):
    out = {}
    for f in glob.glob(f'{folder}/*.csv'):
        n = os.path.basename(f)[:-4]; d = pd.read_csv(f); d['job_id'] = n; out.setdefault(kind(n), []).append(d)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}
PR = load(P); SU = load(SM)
inv = sio.loadmat('D:/MBayrakOCTGA/Revision_Features/inventory.mat')
y2 = inv['y2'].ravel().astype(int); y7 = inv['y7'].ravel().astype(int); grp = inv['group'].ravel().astype(int); N = len(y2)

def cmat(y, p, K): return np.bincount((y - 1) * K + (p - 1), minlength=K * K).reshape(K, K)
def core(y, p, K):
    cm = cmat(y, p, K); tp = np.diag(cm); t = cm.sum(1); q = cm.sum(0); s = cm.sum()
    tt = np.where(t > 0, t, np.nan); f1 = np.where(t + q > 0, 2 * tp / np.maximum(t + q, 1), np.nan)
    den = np.sqrt((s * s - q @ q) * (s * s - t @ t)); pe = (t @ q) / s ** 2
    v = dict(Acc=tp.sum() / s, BAcc=np.nanmean(tp / tt), MacroF1=np.nanmean(f1), MCC=((tp.sum() * s - t @ q) / den) if den else 0.0,
             Kappa=(tp.sum() / s - pe) / (1 - pe))
    if K == 2: v.update(Sens=tp[0] / t[0], Spec=tp[1] / t[1], Prec=tp[0] / max(q[0], 1))
    return v
def prob_metrics(y, pp):
    pos = (y == 1).astype(int); fpr, tpr, _ = roc_curve(pos, pp)
    b = np.clip((pp * 10).astype(int), 0, 9); ece = sum(np.mean(b == k) * abs(pos[b == k].mean() - pp[b == k].mean()) for k in range(10) if (b == k).any())
    return dict(AUC=roc_auc_score(pos, pp), AP=average_precision_score(pos, pp), Brier=np.mean((pp - pos) ** 2), ECE=ece,
                SensAtSpec90=tpr[fpr <= 0.10].max(), SensAtSpec95=tpr[fpr <= 0.05].max())
def pooled(df, n=N):
    df = df.sort_values('image_index')
    if df.image_index.nunique() != n or len(df) != n: return None
    return df
def boot_ci(y, YP, g, K, keys, seed=11):
    ug, gi = np.unique(g, return_inverse=True); mem = [np.flatnonzero(gi == i) for i in range(len(ug))]
    rs = np.random.RandomState(seed); B = np.empty((BOOT, len(keys)))
    for b in range(BOOT):
        ii = np.concatenate([mem[i] for i in rs.randint(0, len(ug), len(ug))])
        B[b] = np.mean([[core(y[ii], YP[ii, r], K)[k] for k in keys] for r in range(YP.shape[1])], 0)
    return np.nanpercentile(B, 2.5, 0), np.nanpercentile(B, 97.5, 0)
def ctest(d, nte, ntr):
    d = np.asarray(d, float); n = len(d); v = d.var(ddof=1); rho = np.mean(np.asarray(nte) / np.asarray(ntr))
    if n < 2 or v == 0: return n, d.mean(), d.std(ddof=1) if n > 1 else np.nan, np.nan, np.nan
    t = d.mean() / np.sqrt((1 / n + rho) * v); return n, d.mean(), d.std(ddof=1), t, 2 * tdist.cdf(-abs(t), n - 1)
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); a = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o])); r = np.empty(m); r[o] = a; return r
def msd(V, keys, pct=('Acc', 'BAcc', 'MacroF1', 'Sens', 'Spec', 'Prec', 'AUC', 'AP', 'SensAtSpec90', 'SensAtSpec95')):
    row = {}
    for k in keys:
        x = np.array([v[k] for v in V], float) * (100 if k in pct else 1)
        row[f'{k}_mean'] = np.nanmean(x); row[f'{k}_sd'] = np.nanstd(x, ddof=1) if len(x) > 1 else np.nan
    return row
sheets = {}

# ---------------- main (E2/E4/E5) ----------------
main = pd.concat([PR['pg'], PR['rp']], ignore_index=True)
rows, YPs, PPs = [], {}, {}
for task in (2, 7):
    y = y2 if task == 2 else y7
    for c in CFGS:
        V, YP, PPl = [], [], []
        for r in range(1, 6):
            d = pooled(main[(main.task == task) & (main.config == c) & (main.repeat == r)])
            if d is None: continue
            v = core(d.y_true.values, d.y_pred.values, task)
            if task == 2 and d.p_pos.notna().all(): v.update(prob_metrics(d.y_true.values, d.p_pos.values)); PPl.append(d.p_pos.values)
            V.append(v); YP.append(d.y_pred.values)
        if not V: continue
        YP = np.array(YP).T; YPs[(task, c)] = YP; PPs[(task, c)] = np.array(PPl).T if PPl else None
        keys = ['Acc', 'BAcc', 'MacroF1', 'MCC', 'Kappa'] + (['Sens', 'Spec', 'Prec'] if task == 2 else [])
        bkeys = ['Acc', 'BAcc', 'MacroF1', 'MCC'] + (['Sens', 'Spec'] if task == 2 else [])
        lo, hi = boot_ci(y, YP, grp, task, bkeys)
        row = dict(Task=f'T{task}', Config=c, nReps=len(V), **msd(V, keys + (['AUC', 'AP', 'Brier', 'ECE', 'SensAtSpec90', 'SensAtSpec95'] if task == 2 and 'AUC' in V[0] else [])))
        for k, a, b in zip(bkeys, lo, hi):
            s = 1 if k == 'MCC' else 100; row[f'{k}_CI_lo'] = a * s; row[f'{k}_CI_hi'] = b * s
        rows.append(row)
sheets['E2E4E5_main'] = pd.DataFrame(rows)

# fold-level + corrected t-tests
fl = []
for (task, c, r, k), d in main.groupby(['task', 'config', 'repeat', 'fold']):
    v = core(d.y_true.values, d.y_pred.values, task); fl.append(dict(task=task, config=c, repeat=r, fold=k, nTe=len(d), nTr=N - len(d), **{m: v[m] for m in ('Acc', 'BAcc', 'MacroF1')}))
fl = pd.DataFrame(fl); sheets['Fold_level_PG'] = fl
tt = []
for task in (2, 7):
    for m in ('BAcc', 'Acc', 'MacroF1'):
        F = fl[(fl.task == task) & (fl.config == 'F')].set_index(['repeat', 'fold']); blk = []
        for c in CFGS[1:]:
            C = fl[(fl.task == task) & (fl.config == c)].set_index(['repeat', 'fold']).reindex(F.index)
            n, md, sd, t, p = ctest(100 * (F[m] - C[m]).values, F.nTe, F.nTr)
            blk.append(dict(Task=f'T{task}', Metric=m, Comparison=f'F vs {c}', nFolds=n, meanDiff_pp=md, sdDiff_pp=sd, t=t, p=p))
        blk = pd.DataFrame(blk); blk['p_Holm'] = holm(blk.p.values); tt.append(blk)
sheets['E4_corrected_ttests'] = pd.concat(tt, ignore_index=True)

# per-class (F) + confusion matrices
pc, cms = [], {}
for task in (2, 7):
    y = y2 if task == 2 else y7; YP = YPs[(task, 'F')]; K = task
    S = np.array([[np.mean(YP[y == c + 1, r] == c + 1) for c in range(K)] for r in range(YP.shape[1])])
    Pr = np.array([[np.sum((YP[:, r] == c + 1) & (y == c + 1)) / max(np.sum(YP[:, r] == c + 1), 1) for c in range(K)] for r in range(YP.shape[1])])
    ug, gi = np.unique(grp, return_inverse=True); mem = [np.flatnonzero(gi == i) for i in range(len(ug))]; rs = np.random.RandomState(12); B = []
    for b in range(BOOT):
        ii = np.concatenate([mem[i] for i in rs.randint(0, len(ug), len(ug))]); yy = y[ii]
        B.append(np.nanmean([[np.mean(YP[ii, r][yy == c + 1] == c + 1) if (yy == c + 1).any() else np.nan for c in range(K)] for r in range(YP.shape[1])], 0))
    lo, hi = np.nanpercentile(np.array(B), [2.5, 97.5], 0)
    F1 = 2 * S * Pr / np.maximum(S + Pr, 1e-12)
    for c in range(K):
        pc.append(dict(Task=f'T{task}', Class=NAMES[task][c], Support=int((y == c + 1).sum()), Sens_mean=100 * S[:, c].mean(), Sens_sd=100 * S[:, c].std(ddof=1),
                       Sens_CI_lo=100 * lo[c], Sens_CI_hi=100 * hi[c], Prec_mean=100 * Pr[:, c].mean(), F1_mean=100 * F1[:, c].mean()))
    cm_sum = sum(cmat(y, YP[:, r], K) for r in range(YP.shape[1]))
    cms[task] = pd.DataFrame(cm_sum, index=[f'true_{n}' for n in NAMES[task]], columns=[f'pred_{n}' for n in NAMES[task]])
sheets['E2_perclass_F'] = pd.DataFrame(pc)
sheets['E2_CM_T2_F_sum5reps'] = cms[2].reset_index(); sheets['E2_CM_T7_F_sum5reps'] = cms[7].reset_index()

# selection / GA / kernel choice / cost
sp = SU['pg']
sel = sp[sp.config.isin(['F', 'A', 'B', 'C', 'G'])].groupby(['task', 'config']).agg(
    nFolds=('nSel', 'size'), nPool_DarkNet=('nPool_darknet', 'mean'), nPool_DenseNet=('nPool_densenet', 'mean'), nPool=('nPool', 'mean'),
    nSel_mean=('nSel', 'mean'), nSel_sd=('nSel', 'std'), nSel_min=('nSel', 'min'), nSel_max=('nSel', 'max'), fromDarkNet=('sel_from_darknet', 'mean'),
    fromDenseNet=('sel_from_densenet', 'mean'), GA_generations_mean=('GA_generations', 'mean'), GA_generations_sd=('GA_generations', 'std'),
    GA_holdout_acc_mean=('GA_best_holdout_acc', 'mean')).reset_index()
sel['Pct_of_pool_selected'] = 100 * sel.nSel_mean / sel.nPool
sel['Pct_ReliefF_retained'] = 100 * sel.nPool / np.where(sel.config == 'A', 2024, np.where(sel.config == 'B', 2920, 4944))
sheets['E6_selection_GA'] = sel
sheets['E6_SVM_kernel_choice'] = sp.groupby(['task', 'config', 'svm_kernel', 'svm_C_or_lambda']).size().rename('nFolds').reset_index()
cost = sp.groupby(['task', 'config']).agg(ReliefF_s=('sec_relieff', 'median'), GA_s=('sec_ga', 'median'), Classifier_tune_fit_s=('sec_svm_tune_fit', 'median'),
                                          Predict_s=('sec_predict', 'median'), Total_s=('sec_total', 'median'), Total_s_IQR=('sec_total', lambda x: x.quantile(.75) - x.quantile(.25)),
                                          nTest=('nTest', 'median')).reset_index()
cost['Predict_ms_per_image'] = 1000 * cost.Predict_s / cost.nTest
sheets['E11_cost_per_outer_fold'] = cost

# protocols (E3)
prot = []
for task in (2, 7):
    y = y2 if task == 2 else y7; per = {}
    for pr, src in (('PG', main[(main.config == 'F')]), ('IL', PR['il']), ('ORIG', PR['orig'])):
        V = []
        for r in range(1, 6):
            d = pooled(src[(src.task == task) & (src.repeat == r)])
            if d is not None: V.append(core(d.y_true.values, d.y_pred.values, task))
        per[pr] = V
        prot.append(dict(Task=f'T{task}', Protocol=pr, nReps=len(V), **msd(V, ['Acc', 'BAcc', 'MacroF1', 'MCC'])))
    for pr in ('IL', 'ORIG'):
        d = [100 * (a['BAcc'] - b['BAcc']) for a, b in zip(per[pr], per['PG'])]; da = [100 * (a['Acc'] - b['Acc']) for a, b in zip(per[pr], per['PG'])]
        prot.append(dict(Task=f'T{task}', Protocol=f'{pr} minus PG (pp)', nReps=len(d), Acc_mean=np.mean(da), Acc_sd=np.std(da, ddof=1), BAcc_mean=np.mean(d), BAcc_sd=np.std(d, ddof=1)))
P3 = pd.DataFrame(prot); P3['Published_Acc'] = P3.Task.map({'T2': 96.12, 'T7': 88.86}); sheets['E3_protocols'] = P3

# complementarity (E7)
e7 = []
for task in (2, 7):
    y = y2 if task == 2 else y7; A = YPs[(task, 'A')]; Bm = YPs[(task, 'B')]; Fm = YPs[(task, 'F')]
    for r in range(A.shape[1]):
        ca, cb = A[:, r] == y, Bm[:, r] == y; n11, n10, n01, n00 = map(int, ((ca & cb).sum(), (ca & ~cb).sum(), (~ca & cb).sum(), (~ca & ~cb).sum()))
        e7.append(dict(Task=f'T{task}', Repeat=r + 1, both_correct=n11, only_DarkNet_correct=n10, only_DenseNet_correct=n01, both_wrong=n00,
                       errors_A=int((~ca).sum()), errors_B=int((~cb).sum()), errors_F=int((Fm[:, r] != y).sum()), DoubleFault_pct=100 * n00 / N,
                       Disagreement_pct=100 * (n10 + n01) / N, Oracle_acc_pct=100 * (1 - n00 / N), Yule_Q=(n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10),
                       McNemar_p=min(1, 2 * binom.cdf(min(n10, n01), n10 + n01, 0.5))))
sheets['E7_complementarity'] = pd.DataFrame(e7)

# permutation (E8)
if 'perm' in PR:
    d0 = pooled(main[(main.task == 2) & (main.config == 'F') & (main.repeat == 1)]); obs = core(d0.y_true.values, d0.y_pred.values, 2)['BAcc']
    nul = []
    for p, d in PR['perm'].groupby('perm_idx'):
        d = pooled(d)
        if d is not None: nul.append(dict(perm=int(p), BAcc=core(d.y_true.values, d.y_pred.values, 2)['BAcc']))
    nul = pd.DataFrame(nul); B = len(nul)
    sheets['E8_permutation'] = pd.DataFrame([dict(Observed_BAcc=100 * obs, B=B, Null_mean=100 * nul.BAcc.mean(), Null_sd=100 * nul.BAcc.std(ddof=1), Null_max=100 * nul.BAcc.max(),
                                                  p_value=(1 + (nul.BAcc >= obs).sum()) / (B + 1), min_attainable_p=1 / (B + 1))])
    sheets['E8_null_values'] = nul

# calibration detail (E9) - reliability bins for F (all repeats pooled)
pp = PPs[(2, 'F')]; yy = np.tile(y2, pp.shape[1]); pv = pp.T.ravel(); b = np.clip((pv * 10).astype(int), 0, 9)
sheets['E9_reliability_F'] = pd.DataFrame([dict(bin=f'{k/10:.1f}-{(k+1)/10:.1f}', n=int((b == k).sum()), mean_pred=pv[b == k].mean() if (b == k).any() else np.nan,
                                                obs_diseased=(yy[b == k] == 1).mean() if (b == k).any() else np.nan) for k in range(10)])

# learning curve (E10)
lc = []; sl = SU['lc']
for task in (2, 7):
    y = y2 if task == 2 else y7
    for f in (0.25, 0.5, 0.75, 1.0):
        V, ntr = [], []
        for r in (1, 2, 3):
            src = main[(main.config == 'F')] if f == 1.0 else PR['lc'][np.isclose(PR['lc'].lc_frac, f)]
            d = pooled(src[(src.task == task) & (src.repeat == r)])
            if d is None: continue
            V.append(core(d.y_true.values, d.y_pred.values, task))
            s_ = (sp[(sp.task == task) & (sp.config == 'F') & (sp.repeat == r)] if f == 1.0 else sl[(sl.task == task) & np.isclose(sl.lc_frac, f) & (sl.repeat == r)])
            ntr.append(s_.nTrain.mean())
        lc.append(dict(Task=f'T{task}', TrainFraction=f, nReps=len(V), TrainImages_mean=np.mean(ntr), **msd(V, ['Acc', 'BAcc', 'MacroF1'])))
sheets['E10_learning_curve'] = pd.DataFrame(lc)

# GA sensitivity (E12)
e12 = []; pe = PR['e12']; se = SU['e12']
for task in (2, 7):
    y = y2 if task == 2 else y7
    base = pooled(main[(main.task == task) & (main.config == 'F') & (main.repeat == 1)])
    tags = ['base'] + ([t for t in ('MR0p1', 'MR0p01', 'MRinvD', 'N50', 'FAST')] if task == 2 else ['FAST'])
    Fb = fl[(fl.task == task) & (fl.config == 'F') & (fl.repeat == 1)].set_index('fold')
    for tag in tags:
        if tag == 'base': d, s_ = base, sp[(sp.task == task) & (sp.config == 'F') & (sp.repeat == 1)]
        else: d, s_ = pooled(pe[(pe.task == task) & pe.job_id.str.endswith('_' + tag)]), se[(se.task == task) & se.job_id.str.endswith('_' + tag)]
        if d is None: continue
        v = core(d.y_true.values, d.y_pred.values, task); row = dict(Task=f'T{task}', Variant=tag, **{k: 100 * v[k] if k != 'MCC' else v[k] for k in ('Acc', 'BAcc', 'MacroF1', 'MCC')},
                                                                         nSel_mean=s_.nSel.mean(), GA_generations_mean=s_.GA_generations.mean(), GA_holdout_acc_mean=s_.GA_best_holdout_acc.mean(), GA_sec_median=s_.sec_ga.median())
        if tag != 'base':
            fv = [core(g.y_true.values, g.y_pred.values, task)['BAcc'] for _, g in d.groupby('fold')]
            n, md, sd, t, p = ctest(100 * (np.array(fv) - Fb.sort_index().BAcc.values), Fb.nTe, Fb.nTr); row.update(diff_vs_base_pp=md, p_vs_base=p)
        e12.append(row)
sheets['E12_GA_sensitivity'] = pd.DataFrame(e12)

# transfer (E13)
al = np.isin(y7, [1, 2, 4]); e13 = []; agg = {}
for s in range(1, 6):
    f = f'{R}/pkl/T2_TRANSFER_KtoO_s{s}.pkl'
    if not os.path.exists(f): continue
    r = pickle.load(open(f, 'rb'))
    for name, m in (('all_2064', np.ones(N, bool)), ('aligned_AMD_DME_NO', al)):
        v = core(y2[m], r['yPred'][m], 2); v.update(prob_metrics(y2[m], r['pPos'][m])); agg.setdefault(name, []).append((m, r['yPred']))
        e13.append(dict(Seed=s, Subset=name, nImages=int(m.sum()), nPatients=len(np.unique(grp[m])), nSel=r['nSel'], SVM=f"{r['svm']['kernel']} C={r['svm']['C']}",
                        **{k: 100 * v[k] for k in ('Acc', 'BAcc', 'Sens', 'Spec', 'AUC')}, MCC=v['MCC']))
E13 = pd.DataFrame(e13); sumr = []
for name, lst in agg.items():
    m = lst[0][0]; YP = np.array([yp[m] for _, yp in lst]).T; lo, hi = boot_ci(y2[m], YP, grp[m], 2, ['BAcc', 'Sens', 'Spec'], seed=13)
    q = E13[E13.Subset == name]
    sumr.append(dict(Seed='mean+-sd (5 seeds)', Subset=name, nImages=int(m.sum()), Acc=q.Acc.mean(), Acc_sd=q.Acc.std(ddof=1), BAcc=q.BAcc.mean(), BAcc_sd=q.BAcc.std(ddof=1),
                     BAcc_CI_lo=100 * lo[0], BAcc_CI_hi=100 * hi[0], Sens=q.Sens.mean(), Spec=q.Spec.mean(), AUC=q.AUC.mean(), AUC_sd=q.AUC.std(ddof=1)))
sheets['E13_transfer'] = pd.concat([E13, pd.DataFrame(sumr)], ignore_index=True)

# Kermany repeated subsets (E14)
ks = PR['ks']; ksr = []; kfl = []
for s in range(1, 6):
    for c in ('F', 'Etuned'):
        d = ks[ks.job_id.str.endswith(f'_KS{s}') & (ks.config == c)]
        if d.image_index.nunique() != 2000: continue
        v = core(d.y_true.values, d.y_pred.values, 2); ksr.append(dict(Subset=s, Config=c, **{k: 100 * v[k] for k in ('Acc', 'BAcc', 'Sens', 'Spec')}, MCC=v['MCC']))
        for k, g in d.groupby('fold'): kfl.append(dict(Subset=s, Config=c, fold=k, BAcc=core(g.y_true.values, g.y_pred.values, 2)['BAcc'], nTe=len(g)))
K14 = pd.DataFrame(ksr); kfl = pd.DataFrame(kfl)
summ14 = K14.groupby('Config')[['Acc', 'BAcc', 'Sens', 'Spec', 'MCC']].agg(['mean', 'std']); summ14.columns = [f'{a}_{b}' for a, b in summ14.columns]
Fk = kfl[kfl.Config == 'F'].set_index(['Subset', 'fold']); Ek = kfl[kfl.Config == 'Etuned'].set_index(['Subset', 'fold']).reindex(Fk.index)
n, md, sd, t, p = ctest(100 * (Fk.BAcc - Ek.BAcc).values, Fk.nTe, 2000 - Fk.nTe)
sheets['E14_Kermany_subsets'] = pd.concat([K14, summ14.reset_index().assign(Subset='mean/sd over 5 subsets'),
                                           pd.DataFrame([dict(Subset='F vs Etuned (50 folds), corrected t-test', Config='F minus Etuned', diff_BAcc_pp=md, sd_diff_pp=sd, t=t, p_value=p)])], ignore_index=True)

# inventory + README
L = pd.read_csv('D:/MBayrakOCTGA/OCT-GA-MAKALE-MERVEBAYRAK/Datasets/OCTDL_full/OCTDL_labels.csv')
L['eyeKnown'] = L.eye.astype(str).isin(['OD', 'OS']); L['pe'] = L.patient_id.astype(str) + '_' + L.eye.astype(str)
inv7 = [dict(Class=c, Images=int((L.disease == c).sum()), Patients=L.patient_id[L.disease == c].nunique(), EyesKnown=L.pe[(L.disease == c) & L.eyeKnown].nunique()) for c in NAMES[7]]
inv7.append(dict(Class='Total', Images=len(L), Patients=L.patient_id.nunique(), EyesKnown=L.pe[L.eyeKnown].nunique()))
sheets['E0_inventory'] = pd.DataFrame(inv7)
import sklearn, scipy, numba
nperm = sheets.get('E8_permutation', pd.DataFrame([dict(B=0)])).B.iloc[0]
readme = pd.DataFrame([
    ('Created', pd.Timestamp.now().strftime('%d.%m.%Y %H:%M')),
    ('Data', 'OCTDL full release: 2064 images, 820 unique patient_id; frozen DarkNet53 (conv53+avg1) + DenseNet201 (fc1000+avg_pool) features, 4944-D, extracted once in MATLAB R2025a'),
    ('Outer CV', 'Patient-grouped stratified 10-fold x 5 repeats (folds.mat, seeds 1-5); IL = image-level folds; ORIG = replica of the original code'),
    ('Inner protocol', 'z-score, ReliefF (k=10, classification, W>0), GA, SVM kernel {linear, poly2, poly3, rbf} x C {0.1, 1, 10} (inner 5-fold patient-grouped CV, balanced accuracy) and Platt scaling fitted on outer-training data only; inner fold 1 = GA hold-out'),
    ('GA', 'N 100, maxIter 500, stall 100, CR 0.8, MR 0.3, k-NN k=5 (as stated in the manuscript); ORIG replica: N 50, targetAcc 0.95, RReliefF, cubic SVM C=1'),
    ('Tests', 'Corrected resampled t-test (Nadeau-Bengio) on per-fold differences, Holm within task x metric; CIs = patient-cluster bootstrap (2000), metric averaged over repeats'),
    ('Permutation', f'Patient-level label permutation, full pipeline, B = {nperm} completed so far (target 49; groups 50-99 deliberately skipped, see README)'),
    ('Platform', f'{platform.processor()} | Intel Core i9-14900K, 24 cores / 32 threads, 64 GB RAM | Python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}, scikit-learn {sklearn.__version__}, numba {numba.__version__}, pandas {pd.__version__}'),
    ('Units', 'Percent for Acc/BAcc/MacroF1/Sens/Spec/Prec/AUC/AP; MCC, Kappa, Brier, ECE as fractions; *_pp = percentage points'),
    ('Not included', 'Fine-tuned ResNet-18 (E5-iii) and end-to-end CNN latency (E11) - pending'),
], columns=['Item', 'Value'])
order = ['README', 'E0_inventory', 'E2E4E5_main', 'E4_corrected_ttests', 'E2_perclass_F', 'E2_CM_T2_F_sum5reps', 'E2_CM_T7_F_sum5reps', 'E3_protocols',
         'E6_selection_GA', 'E6_SVM_kernel_choice', 'E7_complementarity', 'E8_permutation', 'E8_null_values', 'E9_reliability_F', 'E10_learning_curve',
         'E11_cost_per_outer_fold', 'E12_GA_sensitivity', 'E13_transfer', 'E14_Kermany_subsets', 'Fold_level_PG']
sheets['README'] = readme
from openpyxl.styles import Font, PatternFill
with pd.ExcelWriter(OUT, engine='openpyxl') as w:
    for nm in order:
        if nm in sheets: sheets[nm].to_excel(w, sheet_name=nm, index=False)
    for ws in w.book.worksheets:
        for c in ws[1]: c.font = Font(bold=True); c.fill = PatternFill('solid', fgColor='DDEBF7')
        ws.freeze_panes = 'A2'
pickle.dump(dict(sheets=sheets, YPs=YPs, PPs=PPs), open(f'{R}/aggregate_cache.pkl', 'wb'))
print('written', OUT, 'perm B =', nperm)
