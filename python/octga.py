"""Python port of the OCT-GA R1 revision pipeline (Kod/OCTGA_Revision).
Same nested protocol as octga_run_job.m; GA budget is configurable (FAST mode)."""
import numpy as np, scipy.io as sio, h5py, struct, time, os, pickle, warnings
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
warnings.filterwarnings('ignore')
BASE = 'D:/MBayrakOCTGA'

# ---------------------------------------------------------------- data
def load_data():
    h = h5py.File(f'{BASE}/Revision_Features/features_octdl.mat')
    X = np.ascontiguousarray(h['X'][:].T, dtype=np.float32)
    bb = h['bb'][:].ravel().astype(int); lay = h['lay'][:].ravel().astype(int)
    inv = sio.loadmat(f'{BASE}/Revision_Features/inventory.mat')
    fo = sio.loadmat(f'{BASE}/Revision_Results/folds.mat')
    return dict(X=X, bb=bb, lay=lay, y7=inv['y7'].ravel().astype(int), y2=inv['y2'].ravel().astype(int),
                group=inv['group'].ravel().astype(int), foldsPG=fo['foldsPG'].astype(int),
                foldsIL=fo['foldsIL'].astype(int))

def load_kermany():
    h = h5py.File(f'{BASE}/Revision_Features/features_kermany.mat')
    X = np.ascontiguousarray(h['X'][:].T, dtype=np.float32)
    inv = sio.loadmat(f'{BASE}/Revision_Features/inventory.mat')
    fw = inv['__function_workspace__'].tobytes()
    a = np.frombuffer(fw[:len(fw) // 8 * 8], dtype='<f8')
    m = ((a == 1) | (a == 2)).astype(int)
    idx = np.flatnonzero(np.diff(np.concatenate([[0], m, [0]])))
    runs = [(s, e) for s, e in zip(idx[::2], idx[1::2]) if e - s == X.shape[0]]
    # kerm.y2 (1 Diseased, 2 Normal) = 3 - labelOrig; labelOrig is first in file
    lab = a[runs[0][0]:runs[0][1]].astype(int); y2 = 3 - lab
    assert np.array_equal(y2, a[runs[1][0]:runs[1][1]].astype(int))
    return X, y2

def job_seed(jid):
    v = np.array([ord(c) for c in jid], dtype=np.float64)
    return int(np.mod(np.sum(v * np.arange(1, len(v) + 1)) * 7919 + len(v), 2**31 - 2) + 1)

def strat_group_kfold(y, g, K, seed):
    y = np.asarray(y); g = np.asarray(g)
    ug, gi = np.unique(g, return_inverse=True); nG = len(ug); C = y.max()
    cnt = np.zeros((nG, C)); np.add.at(cnt, (gi, y - 1), 1)
    rs = np.random.RandomState(seed); perm = rs.permutation(nG)
    s = cnt[perm].std(1, ddof=1); order = perm[np.argsort(-s, kind='stable')]
    tot = cnt.sum(0); tot[tot == 0] = 1
    fc = np.zeros((K, C)); gF = np.zeros(nG, int)
    for gid in order:
        best, bk, bn = np.inf, 0, np.inf
        for k in range(K):
            f = fc.copy(); f[k] += cnt[gid]
            ev = np.mean(np.std(f / tot, axis=0, ddof=1)); nk = fc[k].sum()
            if ev < best - 1e-12 or (abs(ev - best) <= 1e-12 and nk < bn):
                best, bk, bn = ev, k, nk
        gF[gid] = bk + 1; fc[bk] += cnt[gid]
    return gF[gi]

# ---------------------------------------------------------------- ReliefF (numba)
import numba
def _scaled(X):
    X = X.astype(np.float32); mn = X.min(0); rng = X.max(0) - mn; rng[rng == 0] = 1
    return np.ascontiguousarray((X - mn) / rng, dtype=np.float32)

@numba.njit(fastmath=True, cache=True)
def _l1(X):
    n, d = X.shape; D = np.empty((n, n), np.float32)
    for i in range(n):
        D[i, i] = np.inf
        for j in range(i + 1, n):
            s = np.float32(0)
            for f in range(d): s += abs(X[i, f] - X[j, f])
            D[i, j] = s; D[j, i] = s
    return D

@numba.njit(cache=True)
def _relief_cls(Xs, Dm, yi, nc, k):
    n, d = Xs.shape; W = np.zeros(d)
    pri = np.zeros(nc)
    for i in range(n): pri[yi[i]] += 1
    pri /= n
    for i in range(n):
        ci = yi[i]
        for c in range(nc):
            ids = np.where(yi == c)[0]
            if c == ci: ids = ids[ids != i]
            kk = min(k, ids.size)
            if kk <= 0: continue
            nb = ids[np.argsort(Dm[i, ids], kind='mergesort')[:kk]]
            fac = -1.0 / kk if c == ci else pri[c] / (1 - pri[ci]) / kk
            for j in nb:
                for f in range(d): W[f] += fac * abs(Xs[j, f] - Xs[i, f])
    return W / n

@numba.njit(cache=True)
def _relief_reg(Xs, Dm, yr, k, w):
    n, d = Xs.shape; NdC = 0.0; NdA = np.zeros(d); NdCdA = np.zeros(d)
    for i in range(n):
        nb = np.argsort(Dm[i], kind='mergesort')[:k]
        for r in range(k):
            j = nb[r]; dy = abs(yr[j] - yr[i]); NdC += dy * w[r]
            for f in range(d):
                df = abs(Xs[j, f] - Xs[i, f]); NdA[f] += w[r] * df; NdCdA[f] += dy * w[r] * df
    return NdCdA / NdC - (NdA - NdCdA) / (n - NdC)

def relieff(X, y, k=10, method='classification', sigma=None):
    """Kononenko ReliefF (classification, equal neighbour weights) / RReliefF (regression,
    rank weights exp(-(rank/sigma)^2), sigma 50); range-scaled L1 distances; all instances."""
    Xs = _scaled(X); Dm = _l1(Xs)
    if method == 'classification':
        cls, yi = np.unique(y, return_inverse=True)
        return _relief_cls(Xs, Dm, yi.astype(np.int64), len(cls), k)
    sigma = 50 if sigma is None else sigma
    y = np.asarray(y, float); yr = (y - y.min()) / max(1e-12, y.max() - y.min())
    w = np.exp(-(np.arange(1, k + 1) / sigma) ** 2); w /= w.sum()
    return _relief_reg(Xs, Dm, yr, k, w)

def relieff_pos(X, y, k, method):
    w = relieff(X, y, k, method); keep = np.flatnonzero(w > 0)
    if keep.size == 0: keep = np.array([np.argmax(w)])
    return keep, w.astype(np.float32)

# ---------------------------------------------------------------- GA
def knn_err(Xt, yt, Xv, yv, mask, k, nCls, rs, nt2, nv2):
    if not mask.any():
        mask = np.zeros_like(mask); mask[rs.randint(mask.size)] = True
    A = Xt[:, mask]; B = Xv[:, mask]
    D2 = nv2[:, mask].sum(1)[:, None] + nt2[:, mask].sum(1)[None, :] - 2 * (B @ A.T)
    nn = np.argpartition(D2, k - 1, axis=1)[:, :k]
    lab = yt[nn]; cnt = np.stack([(lab == c).sum(1) for c in range(1, nCls + 1)], 1)
    return np.mean(cnt.argmax(1) + 1 != yv)

def ga_select(Xt, yt, Xv, yv, g, rs):
    t0 = time.time(); D = Xt.shape[1]; N = g['N']
    MR = g['MR'] if g['MR'] >= 0 else 1.0 / D
    nCls = int(max(yt.max(), yv.max()))
    nt2 = Xt ** 2; nv2 = Xv ** 2
    fitFn = lambda m: knn_err(Xt, yt, Xv, yv, m, g['knnK'], nCls, rs, nt2, nv2)
    if D < 2:
        return dict(mask=np.ones(D, bool), bestFit=fitFn(np.ones(D, bool)), curve=[], nGen=0, nEval=1, stop='D<2', time=0)
    pop = rs.rand(N, D) > 0.5
    fit = np.array([fitFn(p) for p in pop]); bi = fit.argmin(); best = fit[bi]; bestSol = pop[bi].copy()
    curve = []; stall = 0; nEval = N; stop = 'maxIter'; gen = 0
    for gen in range(1, g['maxIter'] + 1):
        w = 1 - fit
        if w.sum() <= 0: w = np.ones(N)
        cumP = np.cumsum(w / w.sum()); cumP[-1] = 1
        ch = []
        while len(ch) < N:
            if rs.rand() < g['CR']:
                p1 = pop[np.searchsorted(cumP, rs.rand())]; p2 = pop[np.searchsorted(cumP, rs.rand())]
                cp = rs.randint(1, D)
                ch.append(np.r_[p1[:cp], p2[cp:]]); ch.append(np.r_[p2[:cp], p1[cp:]])
        ch = np.array(ch[:N]); mut = rs.rand(N, D) < MR; ch[mut] = ~ch[mut]
        cf = np.empty(N); imp = False
        for i in range(N):
            cf[i] = fitFn(ch[i])
            if cf[i] < best: best = cf[i]; bestSol = ch[i].copy(); imp = True
        nEval += N
        allP = np.vstack([pop, ch]); allF = np.r_[fit, cf]; o = np.argsort(allF, kind='stable')
        pop = allP[o[:N]]; fit = allF[o[:N]]; curve.append(best)
        stall = 0 if imp else stall + 1
        if not np.isnan(g.get('targetAcc', np.nan)) and 1 - best >= g['targetAcc']: stop = 'targetAcc'; break
        if stall >= g['stall']: stop = 'stall'; break
    return dict(mask=bestSol, bestFit=best, curve=curve, nGen=gen, nEval=nEval, stop=stop, time=time.time() - t0)

# ---------------------------------------------------------------- SVM (precomputed kernels)
def _kern(G, ntr_te, ntr_tr, D, kernel):
    """G = Xa @ Xb.T ; KernelScale = sqrt(D) (z-scored data)."""
    if kernel == 'linear': return G / D
    if kernel == 'poly2': return (1 + G / D) ** 2
    if kernel == 'poly3': return (1 + G / D) ** 3
    if kernel == 'rbf': return np.exp(-(ntr_te[:, None] + ntr_tr[None, :] - 2 * G) / D)

class KSVM:
    def __init__(self, Xtr, y, kernel, C, prob=False):
        self.Xtr = Xtr.astype(np.float64); self.kernel = kernel; self.D = Xtr.shape[1]
        self.n2 = (self.Xtr ** 2).sum(1)
        K = _kern(self.Xtr @ self.Xtr.T, self.n2, self.n2, self.D, kernel)
        self.m = SVC(kernel='precomputed', C=C, probability=prob, random_state=0,
                     decision_function_shape='ovo', cache_size=1000, max_iter=300000).fit(K, y)
    def _K(self, X):
        X = X.astype(np.float64); return _kern(X @ self.Xtr.T, (X ** 2).sum(1), self.n2, self.D, self.kernel)
    def predict(self, X): return self.m.predict(self._K(X))
    def proba(self, X): return self.m.predict_proba(self._K(X))

def bacc(y, p, nCls):
    return np.mean([np.mean(p[y == c] == c) for c in range(1, nCls + 1) if (y == c).any()])

def svm_tune_fit(X, y, innerFold, kernels, Cs, nCls, prob=False):
    X64 = X.astype(np.float64); G = X64 @ X64.T; n2 = np.diag(G).copy(); D = X.shape[1]
    rows = []; best = None; bs = -np.inf
    Kc = {k: _kern(G, n2, n2, D, k) for k in kernels}
    for k in kernels:
        K = Kc[k]
        for C in Cs:
            pred = np.zeros_like(y)
            for f in np.unique(innerFold):
                tr = innerFold != f; va = ~tr
                m = SVC(kernel='precomputed', C=C, cache_size=1000, max_iter=300000).fit(K[np.ix_(tr, tr)], y[tr])
                pred[va] = m.predict(K[np.ix_(va, tr)])
            s = bacc(y, pred, nCls); rows.append((k, C, s))
            if s > bs: bs = s; best = dict(kernel=k, C=C)
    return KSVM(X, y, best['kernel'], best['C'], prob), best, rows

def fit_l1(X, y, lam, nCls):
    C = 1.0 / (lam * len(y))
    base = LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=200, tol=1e-3)
    return (base if nCls == 2 else OneVsRestClassifier(base)).fit(X, y)

def lin_tune_fit(X, y, innerFold, lambdas, nCls):
    sc = []
    for lam in lambdas:
        pred = np.zeros_like(y)
        for f in np.unique(innerFold):
            tr = innerFold != f; va = ~tr
            pred[va] = fit_l1(X[tr], y[tr], lam, nCls).predict(X[va])
        sc.append(bacc(y, pred, nCls))
    lam = lambdas[int(np.argmax(sc))]
    return fit_l1(X, y, lam, nCls), lam, sc

# ---------------------------------------------------------------- one outer fold
CFG = dict(innerK=5, relieffK=10, kernels=['linear', 'poly2', 'poly3', 'rbf'], C=[0.1, 1, 10],
           lam=list(np.logspace(-5, -1, 9)),
           ga=dict(N=100, maxIter=500, CR=0.8, MR=0.3, stall=100, knnK=5, targetAcc=np.nan),
           gaConfigs=['F', 'A', 'B', 'C', 'G'])

def permute_group_labels(y, group, seed):
    ug, gi = np.unique(group, return_inverse=True)
    gl = np.array([np.bincount(y[gi == i]).argmax() for i in range(len(ug))])
    rs = np.random.RandomState(seed); return gl[rs.permutation(len(gl))][gi]

def run_job(D, task, conf, prot, rep, fold, cfg, lcFrac=1.0, permIdx=0, variant=None, suffix='', permSeedMode='job'):
    jid = f'T{task}_{conf}_{prot}_r{rep}_k{fold}{suffix}'
    t0 = time.time(); seed = job_seed(jid); rs = np.random.RandomState(seed % (2**32))
    y = (D['y2'] if task == 2 else D['y7']).copy(); nCls = task
    if permIdx > 0: y = permute_group_labels(y, D['group'], permIdx if permSeedMode == 'perm' else seed)
    if prot == 'PG': fv = D['foldsPG'][:, rep - 1]; grp = D['group']
    else: fv = D['foldsIL'][:, rep - 1]; grp = np.arange(len(y))
    te = np.flatnonzero(fv == fold); tr = np.flatnonzero(fv != fold)
    if lcFrac < 1:
        ug, gi = np.unique(grp[tr], return_inverse=True)
        gLab = np.array([np.bincount(D['y7'][tr][gi == i]).argmax() for i in range(len(ug))])
        keep = []
        for c in np.unique(gLab):
            gc = ug[gLab == c]; nk = max(1, round(lcFrac * len(gc)))
            keep += list(gc[rs.permutation(len(gc))[:nk]])
        tr = tr[np.isin(grp[tr], keep)]
    ytr, yte = y[tr], y[te]
    Xtr = D['X'][tr]; Xte = D['X'][te]
    mu = Xtr.mean(0); sd = Xtr.std(0, ddof=1); sd[sd == 0] = 1
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    inner = strat_group_kfold(D['y7'][tr], grp[tr], cfg['innerK'], seed + 1)
    gaVal = inner == 1; gaTr = ~gaVal
    c1 = np.flatnonzero(D['bb'] == 1); c2 = np.flatnonzero(D['bb'] == 2)
    T = {}; t = time.time(); W = {}
    if conf in ('F', 'D'):
        k1, W['w1'] = relieff_pos(Xtr[:, c1], ytr, cfg['relieffK'], 'classification')
        k2, W['w2'] = relieff_pos(Xtr[:, c2], ytr, cfg['relieffK'], 'classification')
        pool = np.r_[c1[k1], c2[k2]]
    elif conf == 'A':
        k1, W['w1'] = relieff_pos(Xtr[:, c1], ytr, cfg['relieffK'], 'classification'); pool = c1[k1]
    elif conf == 'B':
        k2, W['w2'] = relieff_pos(Xtr[:, c2], ytr, cfg['relieffK'], 'classification'); pool = c2[k2]
    elif conf == 'G':
        kG, W['wG'] = relieff_pos(Xtr, ytr, cfg['relieffK'], 'classification'); pool = kG
    elif conf in ('C', 'E', 'Etuned', 'L1'): pool = np.arange(D['X'].shape[1])
    elif conf == 'FZ': pool = np.flatnonzero((D['bb'] == 2) & (D['lay'] == 2))
    T['relieff'] = time.time() - t
    ga = None
    if conf in cfg['gaConfigs']:
        g = dict(cfg['ga'])
        if variant: g.update(variant)
        t = time.time(); ga = ga_select(Xtr[gaTr][:, pool], ytr[gaTr], Xtr[gaVal][:, pool], ytr[gaVal], g, rs)
        T['ga'] = time.time() - t; sel = pool[ga['mask']]
    else: sel = pool
    t = time.time(); prob = (nCls == 2)
    if conf == 'E':
        mdl = KSVM(Xtr[:, sel], ytr, 'poly3', 1, prob); best = dict(kernel='poly3', C=1)
    elif conf == 'L1':
        mdl, lam, _ = lin_tune_fit(Xtr[:, sel], ytr, inner, cfg['lam'], nCls); best = dict(kernel='L1', C=lam)
    elif conf == 'FZ':
        mdl, best, _ = svm_tune_fit(Xtr[:, sel], ytr, inner, ['linear'], cfg['C'], nCls, prob)
    else:
        mdl, best, _ = svm_tune_fit(Xtr[:, sel], ytr, inner, cfg['kernels'], cfg['C'], nCls, prob)
    T['fit'] = time.time() - t
    t = time.time(); yPred = mdl.predict(Xte[:, sel]); T['predict'] = time.time() - t
    pPos = None
    if nCls == 2:
        try: pPos = (mdl.predict_proba(Xte[:, sel]) if conf == 'L1' else mdl.proba(Xte[:, sel]))[:, 0]
        except Exception: pPos = None
    T['rp_t0'] = time.time(); rp = None
    if conf == 'F' and prot == 'PG' and lcFrac == 1 and permIdx == 0 and variant is None:
        rpIdx = pool[rs.permutation(len(pool))[:len(sel)]]
        mR, bR, _ = svm_tune_fit(Xtr[:, rpIdx], ytr, inner, cfg['kernels'], cfg['C'], nCls)
        rp = dict(yPred=mR.predict(Xte[:, rpIdx]), pPos=None, svm=bR, nSel=len(rpIdx), selDark=int((D['bb'][rpIdx] == 1).sum()),
                  selDense=int((D['bb'][rpIdx] == 2).sum()), T=dict(svm_tune_fit=time.time() - T['rp_t0']))
    T.pop('rp_t0')
    T['total'] = time.time() - t0
    return dict(id=jid, task=task, config=conf, protocol=prot, rep=rep, fold=fold, lcFrac=lcFrac,
                permIdx=permIdx, teIdx=te, yTrue=yte, yPred=yPred, pPos=pPos, nTrain=len(tr), nTest=len(te),
                nPool=len(pool), nPoolDark=int((D['bb'][pool] == 1).sum()), nPoolDense=int((D['bb'][pool] == 2).sum()), nSel=len(sel), sel=sel.astype(np.uint16),
                selDark=int((D['bb'][sel] == 1).sum()), selDense=int((D['bb'][sel] == 2).sum()),
                ga=None if ga is None else {k: v for k, v in ga.items() if k != 'mask'},
                svm=best, rp=rp, T=T, relieffW=W if rep == 1 and prot == 'PG' and lcFrac == 1 and permIdx == 0 else None)

def run_orig(D, task, rep, cfg_orig_ga):
    """Replica of the original code: global z-score, RReliefF on ALL images, GA hold-out = IL fold 1,
    cubic SVM C=1, image-level 10-fold CV."""
    jid = f'T{task}_F_ORIG_r{rep}_k0'; seed = job_seed(jid); rs = np.random.RandomState(seed % (2**32))
    t0 = time.time(); y = D['y2'] if task == 2 else D['y7']; fv = D['foldsIL'][:, rep - 1]
    Xz = (D['X'] - D['X'].mean(0)) / np.where(D['X'].std(0, ddof=1) == 0, 1, D['X'].std(0, ddof=1))
    c1 = np.flatnonzero(D['bb'] == 1); c2 = np.flatnonzero(D['bb'] == 2)
    tR = time.time(); k1, _ = relieff_pos(Xz[:, c1], y.astype(float), 10, 'regression')
    k2, _ = relieff_pos(Xz[:, c2], y.astype(float), 10, 'regression')
    pool = np.r_[c1[k1], c2[k2]]; v = fv == 1; tR = time.time() - tR
    ga = ga_select(Xz[~v][:, pool], y[~v], Xz[v][:, pool], y[v], cfg_orig_ga, rs); sel = pool[ga['mask']]
    yPred = np.zeros_like(y)
    for k in np.unique(fv):
        te = fv == k; m = KSVM(Xz[~te][:, sel], y[~te], 'poly3', 1); yPred[te] = m.predict(Xz[te][:, sel])
    return dict(id=jid, task=task, config='F', protocol='ORIG', rep=rep, fold=0, teIdx=np.arange(len(y)), nTrain=len(y), nTest=len(y),
                yTrue=y, yPred=yPred, pPos=None, nPool=len(pool), nPoolDark=len(k1), nPoolDense=len(k2), nSel=len(sel),
                selDark=int((D['bb'][sel] == 1).sum()), selDense=int((D['bb'][sel] == 2).sum()), svm=dict(kernel='poly3', C=1), sel=sel.astype(np.uint16),
                ga={k: v for k, v in ga.items() if k != 'mask'}, T=dict(relieff=tR, ga=ga['time'], total=time.time() - t0))

def save(res, d):
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/{res['id']}.pkl", 'wb') as f: pickle.dump(res, f)

def run_transfer(Xk, yk, Xo, bb, seed, cfg):
    """E13: pipeline fitted on Kermany only (image-level inner CV), frozen, applied to all OCTDL."""
    t0 = time.time(); rs = np.random.RandomState(seed)
    mu = Xk.mean(0); sd = Xk.std(0, ddof=1); sd[sd == 0] = 1; Xz = (Xk - mu) / sd
    inner = strat_group_kfold(yk, np.arange(len(yk)), cfg['innerK'], seed)
    c1 = np.flatnonzero(bb == 1); c2 = np.flatnonzero(bb == 2)
    k1, _ = relieff_pos(Xz[:, c1], yk, cfg['relieffK'], 'classification')
    k2, _ = relieff_pos(Xz[:, c2], yk, cfg['relieffK'], 'classification')
    pool = np.r_[c1[k1], c2[k2]]; v = inner == 1
    ga = ga_select(Xz[~v][:, pool], yk[~v], Xz[v][:, pool], yk[v], cfg['ga'], rs); sel = pool[ga['mask']]
    mdl, best, _ = svm_tune_fit(Xz[:, sel], yk, inner, cfg['kernels'], cfg['C'], 2, prob=True)
    Xoz = (Xo - mu) / sd
    return dict(id=f'T2_TRANSFER_KtoO_s{seed}', yPred=mdl.predict(Xoz[:, sel]), pPos=mdl.proba(Xoz[:, sel])[:, 0],
                nPool=len(pool), nPoolDark=len(k1), nPoolDense=len(k2), nSel=len(sel), svm=best, ga={k: v for k, v in ga.items() if k != 'mask'},
                T=dict(total=time.time() - t0))
