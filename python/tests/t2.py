import time, numpy as np, copy, sys
from octga import *
D=load_data()
cfg=copy.deepcopy(CFG); cfg['ga'].update(N=20,maxIter=30,stall=10)
for conf in ['F','E','L1']:
    r=run_job(D,2,conf,'PG',1,1,cfg)
    print(conf, {k:round(v,1) for k,v in r['T'].items()}, 'nPool',r['nPool'],'nSel',r['nSel'],'acc',np.mean(r['yTrue']==r['yPred']),'bacc',bacc(r['yTrue'],r['yPred'],2), r['svm'], r['ga'] and (r['ga']['nGen'],r['ga']['nEval'],r['ga']['time']))
    sys.stdout.flush()
