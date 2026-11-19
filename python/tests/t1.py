import time, numpy as np
from octga import *
D=load_data(); print(D['X'].shape, np.bincount(D['y2']), np.bincount(D['y7']))
Xk,yk=load_kermany(); print(Xk.shape,np.bincount(yk), yk[:3], yk[-3:])
# fold check vs MATLAB
print('PG fold check', np.array_equal(strat_group_kfold(D['y7'],D['group'],10,1), D['foldsPG'][:,0]))
print('seed check', job_seed('T2_F_PG_r1_k1'))
t=time.time()
X=(D['X'][:1600,:2000]-0); y=D['y2'][:1600]
w=relieff(X,y,10); print('relieff 1600x2000', time.time()-t)
