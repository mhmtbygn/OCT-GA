"""FULL-config local driver for the OCT-GA revision (manuscript GA settings).
Independent worker processes claim job groups via lock files (no multiprocessing needed).
Usage: python run_full.py worker <id>   |   python run_full.py count"""
import os, sys, time, copy, traceback, pickle, numpy as np
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'): os.environ.setdefault(v, '1')
import runner2 as R2
import octga
from octga import CFG, run_job, run_orig, run_transfer, load_kermany, load_data, strat_group_kfold
RES = 'D:/MBayrakOCTGA/Revision_Results_Python_FULL'
R2.RES = RES; R2.PRED = f'{RES}/predictions'; R2.SUMM = f'{RES}/summaries'
LOCK = f"{RES}/locks_{os.environ.get('OCTGA_RUNTAG', 'shared')}"; PKL = f'{RES}/pkl'; LOGS = f'{RES}/logs'
for d in (R2.PRED, R2.SUMM, LOCK, PKL, LOGS): os.makedirs(d, exist_ok=True)
R2.cfg = copy.deepcopy(CFG)                     # FULL GA: N 100, maxIter 500, stall 100, CR 0.8, MR 0.3, k 5
R2.ORIG_GA = dict(N=50, maxIter=500, CR=0.8, MR=0.3, stall=np.inf, knnK=5, targetAcc=0.95)   # as original code
CONFIGS = R2.CONFIGS
VARIANTS = {'MR0p1': dict(MR=0.1), 'MR0p01': dict(MR=0.01), 'MRinvD': dict(MR=-1), 'N50': dict(N=50),
            'FAST': dict(N=20, maxIter=50, stall=15)}
KS_SEEDS = (1, 2, 3, 4, 5)                      # E14 Kermany balanced 2000-image subsets

def groups():
    G = []
    for rep in (1, 2, 3, 4, 5):                                       # E2/E4/E5/E6/E7/E9 main
        for task in (7, 2):
            for k in range(1, 11): G.append(('pg', [dict(kind='job', task=task, conf=c, prot='PG', rep=rep, fold=k) for c in CONFIGS]))
    for rep in (1, 2, 3, 4, 5):                                       # E3 protocols
        for task in (7, 2):
            G.append(('orig', [dict(kind='orig', task=task, rep=rep)]))
            for k in range(1, 11): G.append(('il', [dict(kind='job', task=task, conf='F', prot='IL', rep=rep, fold=k)]))
    for s in range(1, 6): G.append(('transfer', [dict(kind='transfer', seed=s)]))          # E13
    for rep in (1, 2, 3):                                             # E10
        for k in range(1, 11):
            for f in (0.25, 0.5, 0.75): G.append(('lc', [dict(kind='job', task=t, conf='F', prot='PG', rep=rep, fold=k, lc=f) for t in (7, 2)]))
    for k in range(1, 11):                                            # E12 (+ FAST budget for both tasks)
        for tag in VARIANTS:
            for t in ((2, 7) if tag == 'FAST' else (2,)): G.append(('e12', [dict(kind='job', task=t, conf='F', prot='PG', rep=1, fold=k, variant=tag)]))
    for s in KS_SEEDS:                                                # E14 Kermany repeated subsets
        for k in range(1, 11): G.append(('ks', [dict(kind='ks', seed=s, conf=c, fold=k) for c in ('F', 'Etuned')]))
    for p in range(1, 100):                                           # E8 permutations (seed = perm index)
        for k in range(1, 11): G.append(('perm', [dict(kind='job', task=2, conf='F', prot='PG', rep=1, fold=k, perm=p)]))
    return G

def gid(i, g): return f'g{i:05d}_{g[0]}'
def jid(j):
    if j['kind'] == 'transfer': return f"T2_TRANSFER_KtoO_s{j['seed']}"
    if j['kind'] == 'ks': return f"T2_{j['conf']}_KS{j['seed']}_r1_k{j['fold']}"
    if 'variant' in j: return f"T{j['task']}_F_PG_r1_k{j['fold']}_{j['variant']}"
    return R2.jid(j)
def jdone(j):
    if j['kind'] == 'transfer': return os.path.exists(f"{PKL}/{jid(j)}.pkl")
    if j['kind'] == 'ks' or 'variant' in j: return os.path.exists(f"{R2.SUMM}/{jid(j)}.csv")
    return R2.done(j)

_KS = {}
def ks_data(seed):
    """Balanced 1000 Normal + 1000 Diseased images drawn from the local Kermany pool (image-level folds)."""
    if seed not in _KS:
        if 'K' not in _KS: _KS['K'] = load_kermany()
        Xk, yk = _KS['K']; rs = np.random.RandomState(1000 + seed)
        idx = np.sort(np.concatenate([rs.choice(np.flatnonzero(yk == c), 1000, replace=False) for c in (1, 2)]))
        y = yk[idx]; n = len(y); folds = strat_group_kfold(y, np.arange(n), 10, 2000 + seed)
        bb = _D['bb']; lay = _D['lay']
        _KS[seed] = dict(X=np.ascontiguousarray(Xk[idx]), bb=bb, lay=lay, y7=y, y2=y, group=np.arange(n) + 1,
                         foldsPG=folds[:, None], foldsIL=folds[:, None], kerm_index=idx + 1)
        np.savetxt(f'{RES}/E14_KS{seed}_kermany_indices.csv', idx + 1, fmt='%d')
    return _KS[seed]

def run_one(j, D):
    if j['kind'] == 'orig':
        r = run_orig(D, j['task'], j['rep'], R2.ORIG_GA); r['nTrain'] = r['nTest'] = len(r['yTrue']); r['lcFrac'] = 1.0
        R2.write(r, D); return r
    if j['kind'] == 'transfer':
        if 'K' not in _KS: _KS['K'] = load_kermany()
        r = run_transfer(_KS['K'][0], _KS['K'][1], D['X'], D['bb'], j['seed'], R2.cfg)
        pickle.dump(r, open(f"{PKL}/{jid(j)}.pkl", 'wb')); return r
    if j['kind'] == 'ks':
        Dk = ks_data(j['seed'])
        r = run_job(Dk, 2, j['conf'], 'PG', 1, j['fold'], R2.cfg, suffix=f"_KS{j['seed']}", permSeedMode='perm')
        r['rp'] = None; R2.write(r, Dk); return r
    i = jid(j); base = f"T{j['task']}_{j['conf']}_{j['prot']}_r{j['rep']}_k{j['fold']}"
    r = run_job(D, j['task'], j['conf'], j['prot'], j['rep'], j['fold'], R2.cfg, lcFrac=j.get('lc', 1.0),
                permIdx=j.get('perm', 0), variant=VARIANTS.get(j.get('variant')), suffix=i[len(base):], permSeedMode='perm')
    if r.get('rp') is not None: R2.write(r, D, 'RP', rp=True)
    R2.write(r, D); return r

def _alive(pid):
    try:
        import psutil; return psutil.pid_exists(pid)
    except Exception:
        return True
def claim(path):
    """Claim a job group; a lock whose owner process is gone is taken over (stale lock)."""
    me = str(os.getpid())
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, me.encode()); os.close(fd); return True
    except FileExistsError:
        try: owner = int(open(path).read().strip() or 0)
        except Exception: return False
        if owner > 100 and _alive(owner): return False
        if owner <= 100:      # old-format lock (worker number, not PID): treat as stale only if no recent output
            if time.time() - os.path.getmtime(path) < 4 * 3600: return False
        tmp = f'{path}.{me}.tmp'; open(tmp, 'w').write(me); os.replace(tmp, path); time.sleep(0.2)
        return open(path).read().strip() == me

_D = None
def worker(wid):
    global _D
    _D = load_data()
    log = open(f'{LOGS}/worker{wid:02d}.log', 'a', buffering=1)
    log.write(f"{time.strftime('%d.%m %H:%M:%S')} START pid {os.getpid()}\n")
    for i, g in enumerate(groups()):
        if all(jdone(j) for j in g[1]): continue
        if not claim(f'{LOCK}/{gid(i, g)}.lock'): continue
        for j in g[1]:
            if jdone(j): continue
            t = time.time()
            try:
                r = run_one(j, _D)
                log.write(f"{time.strftime('%d.%m %H:%M:%S')} OK {jid(j)} {time.time()-t:.1f}s nSel {r.get('nSel')}\n")
            except Exception as e:
                log.write(f"{time.strftime('%d.%m %H:%M:%S')} ERROR {jid(j)} {e!r}\n{traceback.format_exc()}\n")
    log.write(f"{time.strftime('%d.%m %H:%M:%S')} WORKER DONE\n")

if __name__ == '__main__':
    if sys.argv[1] == 'worker': worker(int(sys.argv[2]))
    elif sys.argv[1] == 'count':
        G = groups()
        print(len(G), 'groups,', sum(len(g[1]) for g in G), 'jobs,', sum(all(jdone(j) for j in g[1]) for g in G), 'groups done')
