import json
import numpy as np
from search import *
records=[]
for n in [4,5,6,7,8,9]:
 for d in range(3,n,2):
  for g in [.5,.65,.7,.8,.9,.95]:
   P=succ(np.r_[np.arange(1,n),0],g);H,S,B=svdfit(P,d)
   bad=[]
   for delta in range(1,n):
    for k in range(n):
     gap=P[0,k]-P[delta,k];hgap=H[0,k]-H[delta,k]
     if gap*hgap < -1e-9:bad.append({'delta':delta,'goal':k,'true_gap':float(gap),'fb_gap':float(hgap),'bound':bound(P,0,delta,k)})
   if not bad:continue
   case=max(bad,key=lambda b:b['bound'])
   fit=softmaxfit(P,d,restarts=5,iters=1000)
   r={'n':n,'d':d,'g':g,'case':case,'kl':fit['kl'],'ratio':fit['kl']/case['bound'],'density_gap':float(fit['Q'][0,case['goal']]-fit['Q'][case['delta'],case['goal']]),'Q':fit['Q'].tolist(),'F':fit['F'].tolist(),'B':fit['B'].tolist(),'P':P.tolist(),'H':H.tolist(),'S':S.tolist()}
   records.append(r)
   print({k:r[k] for k in ['n','d','g','case','kl','ratio','density_gap']},flush=True)
   (OUT/'cycle_fits.json').write_text(json.dumps(records,indent=2))
