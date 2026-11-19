import pickle, numpy as np, pandas as pd, os
from sklearn.metrics import roc_auc_score
from octga import load_data
D = load_data(); RES = '/home/claude/w/results/E13_transfer'; os.makedirs(RES, exist_ok=True)
y2, y7, g = D['y2'], D['y7'], D['group']; al = np.isin(y7, [1, 2, 4])
def mets(t, p, s):
    sens = np.mean(p[t == 1] == 1); spec = np.mean(p[t == 2] == 2)
    return dict(Acc=np.mean(t == p), BAcc=(sens + spec) / 2, Sens=sens, Spec=spec, AUC=roc_auc_score((t == 1).astype(int), s))
rows = []
for s in range(1, 6):
    r = pickle.load(open(f'jobs/T2_TRANSFER_KtoO_s{s}.pkl', 'rb'))
    idx = np.arange(len(y2))
    pd.DataFrame(dict(seed=s, image_index=idx + 1, patient_id=g, y7=y7, aligned_subset=al.astype(int), y_true=y2,
                      y_pred=r['yPred'], p_pos=r['pPos'])).to_csv(f'{RES}/T2_TRANSFER_KtoO_s{s}_predictions.csv', index=False)
    for subset, m in (('all_2064', np.ones(len(y2), bool)), ('aligned_AMD_DME_NO', al)):
        t, p, sc, gg = y2[m], r['yPred'][m], r['pPos'][m], g[m]
        M = mets(t, p, sc); ug, gi = np.unique(gg, return_inverse=True); mem = [np.flatnonzero(gi == i) for i in range(len(ug))]
        rs = np.random.RandomState(s); B = []
        for _ in range(2000):
            ii = np.concatenate([mem[i] for i in rs.randint(0, len(ug), len(ug))])
            B.append(list(mets(t[ii], p[ii], sc[ii]).values()))
        lo, hi = np.percentile(np.array(B), [2.5, 97.5], axis=0)
        row = dict(seed=s, subset=subset, nImages=int(m.sum()), nPatients=len(ug), nSel=r['nSel'],
                   svm_kernel=r['svm']['kernel'], svm_C=r['svm']['C'])
        for k, v, a, b in zip(M.keys(), M.values(), lo, hi): row.update({k: v, f'{k}_CI_low': a, f'{k}_CI_high': b})
        rows.append(row)
pd.DataFrame(rows).to_csv(f'{RES}/E13_transfer_metrics_patient_bootstrap.csv', index=False)
print(pd.DataFrame(rows)[['seed', 'subset', 'nImages', 'BAcc', 'BAcc_CI_low', 'BAcc_CI_high', 'AUC']].round(4).to_string(index=False))
