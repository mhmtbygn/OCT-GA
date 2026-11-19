import glob, pickle, numpy as np, pandas as pd, scipy.io as sio, scipy.stats as st, os
from sklearn.metrics import roc_auc_score, roc_curve
from octga import load_data, load_kermany
D = load_data(); N = len(D['y2']); grp = D['group']
J = {}
for f in glob.glob('jobs/*.pkl'):
    r = pickle.load(open(f, 'rb')); J[r['id']] = r

def metrics(yt, yp, n, p=None):
    cm = np.zeros((n, n), int); np.add.at(cm, (yt - 1, yp - 1), 1)
    s = cm.sum(); tp = np.diag(cm); t = cm.sum(1); pp = cm.sum(0)
    fn = t - tp; fp = pp - tp; tn = s - tp - fn - fp
    with np.errstate(all='ignore'):
        sens = tp / t; spec = tn / (tn + fp); prec = tp / pp; f1 = 2 * tp / (2 * tp + fp + fn)
        den = np.sqrt((s**2 - pp @ pp) * (s**2 - t @ t))
    M = dict(acc=np.trace(cm) / s, bacc=np.nanmean(sens[t > 0]), macroF1=np.nanmean(f1[(t > 0) | (pp > 0)]),
             mcc=0 if den == 0 else (np.trace(cm) * s - t @ pp) / den)
    pe = (t @ pp) / s**2; M['kappa'] = (M['acc'] - pe) / (1 - pe)
    M.update(cm=cm, sensC=sens, specC=spec, precC=prec, f1C=f1, support=t)
    if n == 2:
        M.update(sensitivity=sens[0], specificity=sens[1], precision=prec[0], f1=f1[0])
        if p is not None and not np.isnan(p).any():
            pos = (yt == 1).astype(int); M['auc'] = roc_auc_score(pos, p)
            fpr, tpr, _ = roc_curve(pos, p)
            M['sens@spec90'] = tpr[fpr <= 0.10].max(); M['sens@spec95'] = tpr[fpr <= 0.05].max()
            o = np.argsort(-p); ps = pos[o]; prec_at = np.cumsum(ps) / np.arange(1, len(ps) + 1)
            M['ap'] = prec_at[ps == 1].sum() / max(1, ps.sum()); M['brier'] = np.mean((p - pos) ** 2)
            b = np.clip(np.digitize(p, np.linspace(0, 1, 11)[1:-1]), 0, 9); ece = 0
            for k in range(10):
                m = b == k
                if m.any(): ece += m.mean() * abs(pos[m].mean() - p[m].mean())
            M['ece'] = ece
    return M

def pooled(prefix, n_folds=10, rp=False):
    yp = np.zeros(N, int); pp = np.full(N, np.nan); rows = []
    for k in range(1, n_folds + 1):
        r = J.get(f'{prefix}_k{k}' if '{k}' not in prefix else prefix.format(k=k))
        if r is None: return None
        y = r['rp']['yPred'] if rp else r['yPred']; yp[r['teIdx']] = y
        if r.get('pPos') is not None and not rp: pp[r['teIdx']] = r['pPos']
        rows.append(r)
    return yp, (None if np.isnan(pp).any() else pp), rows

def boot_ci(yt, yp, n, B=2000, seed=1):
    ug, gi = np.unique(grp, return_inverse=True); mem = [np.flatnonzero(gi == i) for i in range(len(ug))]
    rs = np.random.RandomState(seed); v = []
    for _ in range(B):
        idx = np.concatenate([mem[i] for i in rs.randint(0, len(ug), len(ug))])
        t = yt[idx]; p = yp[idx]
        v.append(np.mean([np.mean(p[t == c] == c) for c in range(1, n + 1) if (t == c).any()]))
    return np.percentile(v, [2.5, 97.5])

def fold_bacc(rows, rp=False):
    out = []
    for r in rows:
        yp = r['rp']['yPred'] if rp else r['yPred']; yt = r['yTrue']; n = r['task']
        out.append(np.mean([np.mean(yp[yt == c] == c) for c in range(1, n + 1) if (yt == c).any()]))
    return np.array(out)

def corrected_t(d, nte, ntr):
    n = len(d); rho = np.mean(nte / ntr); v = np.var(d, ddof=1)
    if v == 0: return np.nan, np.nan
    t = d.mean() / np.sqrt((1 / n + rho) * v); return t, 2 * st.t.cdf(-abs(t), n - 1)

def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o): run = max(run, min(1, (m - i) * p[j])); adj[j] = run
    return adj

S = {}; REPORT = []
say = lambda s: REPORT.append(s)
CONF = ['F', 'A', 'B', 'C', 'D', 'E', 'Etuned', 'G', 'L1', 'FZ']
DESC = dict(F='Proposed: per-backbone ReliefF -> concat -> GA -> tuned SVM', A='DarkNet53 only (ReliefF+GA)',
            B='DenseNet201 only (ReliefF+GA)', C='Concat, no ReliefF, GA', D='Per-backbone ReliefF, no GA',
            E='All 4944, fixed cubic SVM', Etuned='All 4944, tuned SVM', G='Concat -> single ReliefF -> GA',
            L1='L1 logistic, all 4944', FZ='Frozen DenseNet201 GAP + linear SVM')
main = []; ttests = []
for task in (2, 7):
    yt = D['y2'] if task == 2 else D['y7']
    base = pooled(f'T{task}_F_PG_r1')
    for c in CONF + ['RP']:
        P = pooled(f'T{task}_F_PG_r1', rp=True) if c == 'RP' else pooled(f'T{task}_{c}_PG_r1')
        if P is None: continue
        yp, pp, rows = P; M = metrics(yt, yp, task, pp); fb = fold_bacc(rows, rp=(c == 'RP'))
        ci = boot_ci(yt, yp, task)
        row = dict(Task=f'T{task}', Config=c, Description=DESC.get(c, 'Random subset, same size/pool as GA (E7)'),
                   Acc=M['acc'], BAcc=M['bacc'], BAcc_CI_low=ci[0], BAcc_CI_high=ci[1], MacroF1=M['macroF1'],
                   MCC=M['mcc'], Kappa=M['kappa'], FoldBAcc_mean=fb.mean(), FoldBAcc_sd=fb.std(ddof=1),
                   nSel_mean=np.mean([r['rp']['nSel'] if c == 'RP' else r['nSel'] for r in rows]),
                   sec_per_fold=np.mean([r['T']['total'] for r in rows]) if c != 'RP' else np.nan)
        if task == 2: row.update(Sens=M['sensitivity'], Spec=M['specificity'], AUC=M.get('auc', np.nan))
        main.append(row)
        if c != 'F' and base is not None:
            d = fold_bacc(base[2]) - fb
            t, p = corrected_t(d, np.array([r['nTest'] for r in rows]), np.array([r['nTrain'] for r in rows]))
            ttests.append(dict(Task=f'T{task}', Comparison=f'F vs {c}', meanDiff_BAcc=d.mean(), t=t, p=p))
S['E2_E4_E5_main'] = pd.DataFrame(main)
tt = pd.DataFrame(ttests)
if len(tt):
    tt['p_Holm'] = np.nan
    for task in tt.Task.unique():
        m = tt.Task == task; tt.loc[m, 'p_Holm'] = holm(tt.loc[m, 'p'].values)
S['E4_corrected_ttests'] = tt

# per-class F
pc = []
for task in (2, 7):
    P = pooled(f'T{task}_F_PG_r1')
    if P is None: continue
    yt = D['y2'] if task == 2 else D['y7']; M = metrics(yt, P[0], task)
    names = ['Diseased', 'Normal'] if task == 2 else ['AMD', 'DME', 'ERM', 'NO', 'RAO', 'RVO', 'VID']
    for i, nm in enumerate(names):
        pc.append(dict(Task=f'T{task}', Class=nm, Support=M['support'][i], Sensitivity=M['sensC'][i],
                       Specificity=M['specC'][i], Precision=M['precC'][i], F1=M['f1C'][i]))
    S[f'CM_T{task}_F'] = pd.DataFrame(M['cm'], index=[f'true_{n}' for n in names], columns=[f'pred_{n}' for n in names])
S['perclass_F'] = pd.DataFrame(pc)

# E3 protocols
e3 = []
for task in (2, 7):
    yt = D['y2'] if task == 2 else D['y7']
    for prot in ('PG', 'IL'):
        P = pooled(f'T{task}_F_{prot}_r1')
        if P: M = metrics(yt, P[0], task); e3.append(dict(Task=f'T{task}', Protocol=prot, Acc=M['acc'], BAcc=M['bacc'], MacroF1=M['macroF1'], MCC=M['mcc']))
    r = J.get(f'T{task}_F_ORIG_r1_k0')
    if r: M = metrics(yt, r['yPred'], task); e3.append(dict(Task=f'T{task}', Protocol='ORIG', Acc=M['acc'], BAcc=M['bacc'], MacroF1=M['macroF1'], MCC=M['mcc'], nSel=r['nSel'], GA_stop=r['ga']['stop']))
S['E3_protocols'] = pd.DataFrame(e3)

# E6 selection stability
e6 = []
for task in (2, 7):
    for c in ('F', 'A', 'B', 'C', 'G'):
        P = pooled(f'T{task}_{c}_PG_r1')
        if not P: continue
        sels = [set(r['sel'].tolist()) for r in P[2]]
        jac = [len(a & b) / len(a | b) for i, a in enumerate(sels) for b in sels[i + 1:]]
        e6.append(dict(Task=f'T{task}', Config=c, nPool_mean=np.mean([r['nPool'] for r in P[2]]),
                       nSel_mean=np.mean([r['nSel'] for r in P[2]]), selDarkNet_mean=np.mean([r['selDark'] for r in P[2]]),
                       selDenseNet_mean=np.mean([r['selDense'] for r in P[2]]), Jaccard_mean=np.mean(jac), Jaccard_min=np.min(jac),
                       GA_nGen_mean=np.mean([r['ga']['nGen'] for r in P[2]]), GA_bestHoldoutAcc_mean=np.mean([1 - r['ga']['bestFit'] for r in P[2]])))
S['E6_selection'] = pd.DataFrame(e6)

# E7 complementarity A vs B
e7 = []
for task in (2, 7):
    yt = D['y2'] if task == 2 else D['y7']; PA = pooled(f'T{task}_A_PG_r1'); PB = pooled(f'T{task}_B_PG_r1'); PF = pooled(f'T{task}_F_PG_r1')
    if not (PA and PB): continue
    ea = PA[0] != yt; eb = PB[0] != yt
    n11 = np.sum(~ea & ~eb); n00 = np.sum(ea & eb); n10 = np.sum(~ea & eb); n01 = np.sum(ea & ~eb)
    Q = (n11 * n00 - n10 * n01) / (n11 * n00 + n10 * n01)
    pm = st.binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if n10 + n01 > 0 else 1
    row = dict(Task=f'T{task}', err_A_DarkNet=ea.sum(), err_B_DenseNet=eb.sum(), both_wrong=n00, onlyA_wrong=n01, onlyB_wrong=n10,
               oracle_acc=1 - n00 / N, Yule_Q=Q, McNemar_exact_p=pm)
    if PF: row['err_F'] = int((PF[0] != yt).sum())
    e7.append(row)
S['E7_complementarity'] = pd.DataFrame(e7)

# E8 permutation
obs = pooled('T2_F_PG_r1'); null = []
for p in range(1, 101):
    P = pooled(f'T2_F_PG_r1_k{{k}}_perm{p:03d}')
    if P is None: continue
    yperm = np.zeros(N, int)
    for r in P[2]: yperm[r['teIdx']] = r['yTrue']
    null.append(metrics(yperm, P[0], 2)['bacc'])
if obs and null:
    ob = metrics(D['y2'], obs[0], 2)['bacc']; null = np.array(null)
    S['E8_permutation'] = pd.DataFrame([dict(observed_BAcc=ob, null_mean=null.mean(), null_sd=null.std(ddof=1), null_max=null.max(),
                                             B=len(null), p_value=(1 + np.sum(null >= ob)) / (1 + len(null)))])
    S['E8_null_values'] = pd.DataFrame(dict(perm=np.arange(1, len(null) + 1), BAcc=null))

# E9 calibration (F, D, L1, Etuned T2)
e9 = []
for c in ('F', 'A', 'B', 'D', 'Etuned', 'G', 'L1', 'FZ'):
    P = pooled(f'T2_{c}_PG_r1')
    if P and P[1] is not None:
        M = metrics(D['y2'], P[0], 2, P[1])
        e9.append(dict(Config=c, AUC=M['auc'], AP=M['ap'], Brier=M['brier'], ECE=M['ece'], Sens_at_Spec90=M['sens@spec90'], Sens_at_Spec95=M['sens@spec95']))
S['E9_calibration_T2'] = pd.DataFrame(e9)

# E10 learning curve
e10 = []
for task in (2, 7):
    yt = D['y2'] if task == 2 else D['y7']
    for f in (25, 50, 75, 100):
        P = pooled(f'T{task}_F_PG_r1') if f == 100 else pooled(f'T{task}_F_PG_r1_k{{k}}_lc{f:02d}')
        if P: M = metrics(yt, P[0], task); e10.append(dict(Task=f'T{task}', TrainPatientsPct=f, nTrain_mean=np.mean([r['nTrain'] for r in P[2]]), Acc=M['acc'], BAcc=M['bacc'], MacroF1=M['macroF1']))
S['E10_learning_curve'] = pd.DataFrame(e10)

# E11 cost (classifier stages, Python) + MATLAB extraction info
e11 = []
for task in (2, 7):
    P = pooled(f'T{task}_F_PG_r1')
    if P:
        for k in ('relieff', 'ga', 'fit', 'predict', 'total'):
            e11.append(dict(Task=f'T{task}', Stage=k, mean_sec=np.mean([r['T'].get(k, np.nan) for r in P[2]])))
        e11.append(dict(Task=f'T{task}', Stage='predict_ms_per_image', mean_sec=1000 * np.mean([r['T']['predict'] / r['nTest'] for r in P[2]])))
S['E11_cost_python'] = pd.DataFrame(e11)

# E12 GA sensitivity
e12 = []
for tag in ('base', 'MR0p1', 'MR0p01', 'MRinvD', 'N50'):
    P = pooled('T2_F_PG_r1') if tag == 'base' else pooled(f'T2_F_PG_r1_k{{k}}_{tag}')
    if P:
        M = metrics(D['y2'], P[0], 2)
        e12.append(dict(Variant=tag, BAcc=M['bacc'], Acc=M['acc'], MCC=M['mcc'], nSel_mean=np.mean([r['nSel'] for r in P[2]]),
                        GA_nGen_mean=np.mean([r['ga']['nGen'] for r in P[2]]), GA_holdoutAcc_mean=np.mean([1 - r['ga']['bestFit'] for r in P[2]])))
S['E12_GA_sensitivity'] = pd.DataFrame(e12)

# E13 transfer
e13 = []
for s in range(1, 6):
    r = J.get(f'T2_TRANSFER_KtoO_s{s}')
    if r:
        M = metrics(D['y2'], r['yPred'], 2, r['pPos'])
        e13.append(dict(Seed=s, Acc=M['acc'], BAcc=M['bacc'], Sens=M['sensitivity'], Spec=M['specificity'], MCC=M['mcc'], AUC=M.get('auc'), nSel=r['nSel'], SVM=f"{r['svm']['kernel']} C={r['svm']['C']}"))
S['E13_transfer_Kermany_to_OCTDL'] = pd.DataFrame(e13)

# MATLAB reference (full GA) from the uploaded job files
mrows = []
for task in (2, 7):
    yt = D['y2'] if task == 2 else D['y7']
    for c in CONF:
        fs = [f'/home/claude/w/Revision_Results/jobs/T{task}_{c}_PG_r1_k{k}.mat' for k in range(1, 11)]
        if not all(os.path.exists(f) for f in fs): continue
        yp = np.zeros(N, int)
        for f in fs:
            r = sio.loadmat(f, squeeze_me=True, struct_as_record=False)['res']; yp[np.atleast_1d(r.teIdx) - 1] = np.atleast_1d(r.yPred)
        M = metrics(yt, yp, task)
        mrows.append(dict(Task=f'T{task}', Config=c, Acc=M['acc'], BAcc=M['bacc'], MacroF1=M['macroF1'], MCC=M['mcc']))
S['MATLAB_full_GA_reference'] = pd.DataFrame(mrows)
pickle.dump(S, open('summary.pkl', 'wb'))
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
for k, v in S.items(): print('\n###', k); print(v.round(4).to_string(index=False if not k.startswith('CM') else True))
