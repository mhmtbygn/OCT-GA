import sys, os, time, copy, pickle, numpy as np, traceback
from octga import *
OUT = '/home/claude/w/py/jobs'; os.makedirs(OUT, exist_ok=True)
FAST_GA = dict(N=20, maxIter=50, stall=15)        # full: N 100, maxIter 500, stall 100
cfg = copy.deepcopy(CFG); cfg['ga'].update(FAST_GA)
ORIG_GA = dict(N=20, maxIter=50, CR=0.8, MR=0.3, stall=np.inf, knnK=5, targetAcc=0.95)
D = load_data()
# cache ReliefF per training set (identical for F/A/B/D and for G): wrap relieff_pos
import octga
_cache = {}
_orig_rp = octga.relieff_pos
def cached_rp(X, y, k, m):
    key = (X.shape, float(X[:5, :5].sum()), float(X[-3:, -3:].sum()), y.tobytes().__hash__(), m)
    if key not in _cache:
        if len(_cache) > 6: _cache.clear()
        _cache[key] = _orig_rp(X, y, k, m)
    return _cache[key]
octga.relieff_pos = cached_rp
CONFIGS = ['F', 'A', 'B', 'C', 'D', 'E', 'Etuned', 'G', 'L1', 'FZ']
VARIANTS = {'MR0p1': dict(MR=0.1), 'MR0p01': dict(MR=0.01), 'MRinvD': dict(MR=-1), 'N50': dict(N=50)}
def joblist(stage):
    J = []
    if stage == 'P1':
        for task in (2, 7):
            for k in range(1, 11):
                for c in CONFIGS: J.append(dict(kind='job', task=task, conf=c, prot='PG', rep=1, fold=k))
    if stage == 'P2':
        for task in (2, 7):
            J.append(dict(kind='orig', task=task))
            for k in range(1, 11): J.append(dict(kind='job', task=task, conf='F', prot='IL', rep=1, fold=k))
    if stage == 'P3':
        for s in range(1, 6): J.append(dict(kind='transfer', seed=s))
    if stage == 'P4':
        for k in range(1, 11):
            for tag in VARIANTS: J.append(dict(kind='job', task=2, conf='F', prot='PG', rep=1, fold=k, variant=tag))
    if stage == 'P5':
        for task in (2, 7):
            for k in range(1, 11):
                for f in (0.25, 0.5, 0.75): J.append(dict(kind='job', task=task, conf='F', prot='PG', rep=1, fold=k, lc=f))
    if stage == 'P6':
        for p in range(1, 20):
            for k in range(1, 11): J.append(dict(kind='job', task=2, conf='F', prot='PG', rep=1, fold=k, perm=p))
    return J
def jid(j):
    if j['kind'] == 'orig': return f"T{j['task']}_F_ORIG_r1_k0"
    if j['kind'] == 'transfer': return f"T2_TRANSFER_KtoO_s{j['seed']}"
    suf = ''
    if 'variant' in j: suf = '_' + j['variant']
    if 'lc' in j: suf = f"_lc{int(round(100*j['lc'])):02d}"
    if 'perm' in j: suf = f"_perm{j['perm']:03d}"
    return f"T{j['task']}_{j['conf']}_{j['prot']}_r{j['rep']}_k{j['fold']}{suf}"
Xk = yk = None
for stage in sys.argv[1:]:
    J = joblist(stage); t0 = time.time()
    for n, j in enumerate(J):
        i = jid(j)
        if os.path.exists(f'{OUT}/{i}.pkl'): continue
        t = time.time()
        try:
            if j['kind'] == 'orig': r = run_orig(D, j['task'], 1, ORIG_GA)
            elif j['kind'] == 'transfer':
                if Xk is None: Xk, yk = load_kermany()
                r = run_transfer(Xk, yk, D['X'], D['bb'], j['seed'], cfg)
            else:
                suf = i[len(f"T{j['task']}_{j['conf']}_{j['prot']}_r1_k{j['fold']}"):]
                r = run_job(D, j['task'], j['conf'], j['prot'], j['rep'], j['fold'], cfg, lcFrac=j.get('lc', 1.0),
                            permIdx=j.get('perm', 0), variant=VARIANTS.get(j.get('variant')), suffix=suf)
            save(r, OUT)
            yt = r.get('yTrue', D['y2']); ba = bacc(yt, r['yPred'], int(yt.max()))
            print(f"[{stage} {n+1}/{len(J)}] {i:30s} {time.time()-t:6.1f}s nSel {r['nSel']:5d} BAcc {ba:.4f}", flush=True)
        except Exception as e:
            print('ERROR', i, e, flush=True); traceback.print_exc()
    print(f'== {stage} done in {(time.time()-t0)/60:.1f} min', flush=True)
