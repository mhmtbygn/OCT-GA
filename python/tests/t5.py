import time,numpy as np
from octga import *
D=load_data(); fv=D['foldsPG'][:,0]; tr=fv!=1
X=D['X'][tr]; X=(X-X.mean(0))/X.std(0); y=D['y7'][tr]
for lam in [1e-5,1e-3,1e-1]:
    t=time.time(); m=fit_l1(X[:1500],y[:1500],lam,7); print(lam,time.time()-t,flush=True)
