"""Local driver: independent worker processes claim jobs via lock files (no multiprocessing)."""
import os, sys, time, traceback, pickle, numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import runner2 as R2
from octga import run_job, run_orig, run_transfer, load_kermany, save
RES = R2.RES; LOCK = f'{RES}/locks'; PKL = f'{RES}/pkl'
for d in (LOCK, PKL): os.makedirs(d, exist_ok=True)
CONFIGS = R2.CONFIGS
VARIANTS = {'MR0p1': dict(MR=0.1), 'MR0p01': dict(MR=0.01), 'MRinvD': dict(MR=-1), 'N50': dict(N=50)}

def groups():
    G = []
    for rep in (1, 2, 3, 4, 5):                       # PG main, T7 first (heavier)
        for task in (7, 2):
            for k in range(1, 11): G.append(('pg', [dict(kind='job', task=task, conf=c, prot='PG', rep=rep, fold=k) for c in CONFIGS]))
    for rep in (1, 2, 3, 4, 5):                       # protocols
        for task in (7, 2):
            G.append(('orig', [dict(kind='orig', task=task, rep=rep)]))
            for k in range(1, 11): G.append(('il', [dict(kind='job', task=task, conf='F', prot='IL', rep=rep, fold=k)]))
    for s in range(1, 6): G.append(('transfer', [dict(kind='transfer', seed=s)]))
    for p in range(1, 100):                           # permutations (seed = perm index)
        for k in range(1, 11): G.append(('perm', [dict(kind='job', task=2, conf='F', prot='PG', rep=1, fold=k, perm=p)]))
    for rep in (1, 2, 3):                             # learning curve
        for k in range(1, 11):
            for f in (0.25, 0.5, 0.75): G.append(('lc', [dict(kind='job', task=t, conf='F', prot='PG', rep=rep, fold=k, lc=f) for t in (7, 2)]))
    for k in range(1, 11):                            # GA sensitivity (E12)
        for tag in VARIANTS: G.append(('e12', [dict(kind='job', task=2, conf='F', prot='PG', rep=1, fold=k, variant=tag)]))
    return G

def gid(i, g): return f'g{i:05d}_{g[0]}'
def jid(j):
    if j['kind'] == 'transfer': return f"T2_TRANSFER_KtoO_s{j['seed']}"
    if 'variant' in j: return f"T2_F_PG_r1_k{j['fold']}_{j['variant']}"
    return R2.jid(j)
def jdone(j):
    if j['kind'] == 'transfer': return os.path.exists(f"{PKL}/{jid(j)}.pkl")
    if 'variant' in j: return os.path.exists(f"{R2.SUMM}/{jid(j)}.csv")
    return R2.done(j)

def run_one(j, D, Xk):
    if j['kind'] == 'orig':
        r = R2.run_orig(D, j['task'], j['rep'], R2.ORIG_GA); r['nTrain'] = r['nTest'] = len(r['yTrue']); r['lcFrac'] = 1.0
        R2.write(r, D); return r
    if j['kind'] == 'transfer':
        r = run_transfer(Xk[0], Xk[1], D['X'], D['bb'], j['seed'], R2.cfg)
        pickle.dump(r, open(f"{PKL}/{jid(j)}.pkl", 'wb')); return r
    i = jid(j); base = f"T{j['task']}_{j['conf']}_{j['prot']}_r{j['rep']}_k{j['fold']}"
    r = run_job(D, j['task'], j['conf'], j['prot'], j['rep'], j['fold'], R2.cfg, lcFrac=j.get('lc', 1.0),
                permIdx=j.get('perm', 0), variant=VARIANTS.get(j.get('variant')), suffix=i[len(base):], permSeedMode='perm')
    if r.get('rp') is not None: R2.write(r, D, 'RP', rp=True)
    R2.write(r, D); return r

def worker(wid):
    from octga import load_data
    D = load_data(); Xk = None
    log = open(f'{RES}/logs/worker{wid:02d}.log', 'a', buffering=1)
    for i, g in enumerate(groups()):
        if all(jdone(j) for j in g[1]): continue
        try: fd = os.open(f'{LOCK}/{gid(i, g)}.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(wid).encode()); os.close(fd)
        except FileExistsError: continue
        for j in g[1]:
            if jdone(j): continue
            t = time.time()
            try:
                if j['kind'] == 'transfer' and Xk is None: Xk = load_kermany()
                r = run_one(j, D, Xk)
                log.write(f"{time.strftime('%H:%M:%S')} OK {jid(j)} {time.time()-t:.1f}s nSel {r.get('nSel')}\n")
            except Exception as e:
                log.write(f"{time.strftime('%H:%M:%S')} ERROR {jid(j)} {e!r}\n{traceback.format_exc()}\n")
    log.write('WORKER DONE\n')

if __name__ == '__main__':
    os.makedirs(f'{RES}/logs', exist_ok=True)
    if sys.argv[1] == 'worker': worker(int(sys.argv[2]))
    elif sys.argv[1] == 'count':
        G = groups(); print(len(G), 'groups,', sum(len(g[1]) for g in G), 'jobs,', sum(all(jdone(j) for j in g[1]) for g in G), 'groups done')
