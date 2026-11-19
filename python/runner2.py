"""Remaining OCT-GA revision jobs; one predictions CSV + one summary CSV per job; resumable."""
import os, sys, time, copy, glob, zipfile, shutil, numpy as np, pandas as pd, traceback
os.environ.setdefault('OMP_NUM_THREADS', '1')
from joblib import Parallel, delayed
import octga
from octga import *
RES = 'D:/MBayrakOCTGA/Revision_Results_Python'; PRED = f'{RES}/predictions'; SUMM = f'{RES}/summaries'
for d in (PRED, SUMM): os.makedirs(d, exist_ok=True)
cfg = copy.deepcopy(CFG); cfg['ga'].update(N=20, maxIter=50, stall=15)
ORIG_GA = dict(N=20, maxIter=50, CR=0.8, MR=0.3, stall=np.inf, knnK=5, targetAcc=0.95)
CONFIGS = ['F', 'A', 'B', 'C', 'D', 'E', 'Etuned', 'G', 'L1', 'FZ']
_cache = {}; _orig_rp = octga.relieff_pos
def cached_rp(X, y, k, m):  # identical ReliefF call on identical training data -> reuse (F/A/B/D share it)
    key = (X.shape, float(X[:5, :5].sum()), float(X[-3:, -3:].sum()), hash(y.tobytes()), m)
    if key not in _cache:
        if len(_cache) > 6: _cache.clear()
        _cache[key] = _orig_rp(X, y, k, m)
    return _cache[key]
octga.relieff_pos = cached_rp

def jid(j):
    if j['kind'] == 'orig': return f"T{j['task']}_F_ORIG_r{j['rep']}_k0"
    suf = (f"_lc{int(round(100*j['lc'])):02d}" if 'lc' in j else '') + (f"_perm{j['perm']:03d}" if 'perm' in j else '')
    return f"T{j['task']}_{j['conf']}_{j['prot']}_r{j['rep']}_k{j['fold']}{suf}"

def write(r, D, conf_override=None, rp=False):
    conf = conf_override or r['config']; jidx = r['id'] if not rp else r['id'].replace('_F_', '_RP_', 1)
    te = np.asarray(r['teIdx']); yp = r['rp']['yPred'] if rp else r['yPred']
    pp = (r['rp']['pPos'] if rp else r.get('pPos'))
    keys = dict(task=r['task'], config=conf, protocol=r['protocol'], repeat=r['rep'], fold=r['fold'],
                perm_idx=r.get('permIdx', 0), lc_frac=r.get('lcFrac', 1.0))
    P = pd.DataFrame(dict(**{k: [v] * len(te) for k, v in keys.items()}, image_index=te + 1,
                          patient_id=D['group'][te], y_true=r['yTrue'], y_pred=yp,
                          p_pos=pp if (pp is not None and r['task'] == 2) else np.nan))
    ga = r.get('ga') or {}; svm = r['rp']['svm'] if rp else (r.get('svm') or {}); T = r['T']
    S = dict(**keys, job_id=jidx, nTrain=r['nTrain'], nTest=r['nTest'], nPool_darknet=r['nPoolDark'],
             nPool_densenet=r['nPoolDense'], nPool=r['nPool'], nSel=r['rp']['nSel'] if rp else r['nSel'],
             sel_from_darknet=r['rp']['selDark'] if rp else r['selDark'], sel_from_densenet=r['rp']['selDense'] if rp else r['selDense'],
             GA_generations=np.nan if rp else ga.get('nGen', np.nan), GA_stop='' if rp else ga.get('stop', ''),
             GA_best_holdout_acc=np.nan if (rp or 'bestFit' not in ga) else 1 - ga['bestFit'],
             svm_kernel=svm.get('kernel', ''), svm_C_or_lambda=svm.get('C', np.nan),
             sec_relieff=np.nan if rp else T.get('relieff', np.nan), sec_ga=np.nan if rp else T.get('ga', np.nan),
             sec_svm_tune_fit=r['rp']['T']['svm_tune_fit'] if rp else T.get('fit', np.nan),
             sec_predict=np.nan if rp else T.get('predict', np.nan), sec_total=np.nan if rp else T.get('total', np.nan))
    P.to_csv(f'{PRED}/{jidx}.csv.tmp', index=False); os.replace(f'{PRED}/{jidx}.csv.tmp', f'{PRED}/{jidx}.csv')
    pd.DataFrame([S]).to_csv(f'{SUMM}/{jidx}.csv.tmp', index=False); os.replace(f'{SUMM}/{jidx}.csv.tmp', f'{SUMM}/{jidx}.csv')

def done(j): return os.path.exists(f'{SUMM}/{jid(j)}.csv') and (j.get('conf') != 'F' or j.get('prot') != 'PG' or 'lc' in j or 'perm' in j
                                                               or os.path.exists(f"{SUMM}/{jid(j).replace('_F_', '_RP_', 1)}.csv"))
_D = None
def run_group(group):
    global _D
    if _D is None: _D = load_data()
    out = []
    for j in group:
        if done(j): continue
        t = time.time()
        try:
            if j['kind'] == 'orig':
                r = run_orig(_D, j['task'], j['rep'], ORIG_GA); r['nTrain'] = r['nTest'] = len(r['yTrue']); r['lcFrac'] = 1.0
            else:
                i = jid(j); suf = i[len(f"T{j['task']}_{j['conf']}_{j['prot']}_r{j['rep']}_k{j['fold']}"):]
                r = run_job(_D, j['task'], j['conf'], j['prot'], j['rep'], j['fold'], cfg, lcFrac=j.get('lc', 1.0),
                            permIdx=j.get('perm', 0), suffix=suf, permSeedMode='perm')
            if r.get('rp') is not None: write(r, _D, 'RP', rp=True)
            write(r, _D)
            out.append(f"{r['id']} {time.time()-t:.1f}s")
        except Exception as e:
            out.append(f'ERROR {jid(j)} {e!r}'); traceback.print_exc()
    return out

def groups():
    G = []
    for rep in (2, 3, 4, 5):                                   # 1. PG repeats 2-5, all configs
        for task in (2, 7):
            for k in range(1, 11): G.append([dict(kind='job', task=task, conf=c, prot='PG', rep=rep, fold=k) for c in CONFIGS])
    for rep in (2, 3, 4, 5):                                   # 2. IL + ORIG repeats 2-5
        for task in (2, 7):
            G.append([dict(kind='orig', task=task, rep=rep)])
            for k in range(1, 11): G.append([dict(kind='job', task=task, conf='F', prot='IL', rep=rep, fold=k)])
    for p in list(range(81, 100)) + list(range(20, 81)):       # 3. permutations (seed = perm index)
        for k in range(1, 11): G.append([dict(kind='job', task=2, conf='F', prot='PG', rep=1, fold=k, perm=p)])
    for rep in (2, 3):                                         # 4. learning curve
        for k in range(1, 11):
            for f in (0.25, 0.5, 0.75): G.append([dict(kind='job', task=t, conf='F', prot='PG', rep=rep, fold=k, lc=f) for t in (2, 7)])
    return G

def zip_results():
    tmp = '/home/claude/w/results.zip.tmp'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(glob.glob(f'{RES}/**/*', recursive=True)):
            if os.path.isfile(f) and not f.endswith('.tmp'): z.write(f, os.path.relpath(f, os.path.dirname(RES)))
    os.makedirs('/mnt/user-data/outputs', exist_ok=True); shutil.move(tmp, '/mnt/user-data/outputs/results.zip')

if __name__ == '__main__':
    todo = [g for g in groups() if not all(done(j) for j in g)]
    print(f'{len(todo)} groups to run', flush=True); last = time.time()
    nj = max(1, os.cpu_count())
    for s in range(0, len(todo), nj):
        for res in Parallel(n_jobs=nj)(delayed(run_group)(g) for g in todo[s:s + nj]):
            for line in res: print(line, flush=True)
        if time.time() - last > 1800: zip_results(); last = time.time(); print('zipped', flush=True)
    zip_results(); print('ALL DONE', flush=True)
