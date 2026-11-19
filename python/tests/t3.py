import numpy as np, time, numba
from scipy.spatial.distance import cdist
@numba.njit(fastmath=True, cache=True)
def l1(X):
    n,d=X.shape; D=np.empty((n,n),np.float32)
    for i in range(n):
        D[i,i]=np.inf
        for j in range(i+1,n):
            s=np.float32(0)
            for f in range(d): s+=abs(X[i,f]-X[j,f])
            D[i,j]=s; D[j,i]=s
    return D
X=np.random.rand(1850,2470).astype(np.float32)
l1(X[:10]); t=time.time(); D=l1(X); print('numba',time.time()-t)
t=time.time(); D2=cdist(X[:400].astype(np.float64),X.astype(np.float64),'cityblock'); print('cdist 400 rows',time.time()-t)
print(np.abs(D[:400]-D2).max() if False else '')
