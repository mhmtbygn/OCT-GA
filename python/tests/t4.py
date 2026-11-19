import time, numpy as np
from octga import *
D=load_data(); X=D['X'][:1850]; y=D['y2'][:1850]; y7=D['y7'][:1850]
relieff(X[:50,:20],y[:50]); relieff(X[:50,:20],y[:50].astype(float),method='regression')
for yy,nm in ((y,'T2'),(y7,'T7')):
    t=time.time(); w=relieff(X,yy,10); print(nm,'relieff full 4944',time.time()-t, (w>0).sum())
t=time.time(); w=relieff(X,y.astype(float),10,'regression'); print('rrelieff',time.time()-t,(w>0).sum())
