import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp,xlogy
OUT=Path(__file__).parent

def succ(nexts,g):
 n=len(nexts); T=np.eye(n)[nexts]; return (1-g)*np.linalg.inv(np.eye(n)-g*T)
def svdfit(P,d,w=None,rho=None):
 m,n=P.shape
 if w is None:w=np.full(m,1/m)
 if rho is None:rho=np.full(n,1/n)
 Z=np.sqrt(w)[:,None]*P/np.sqrt(rho)
 U,S,Vt=np.linalg.svd(Z,full_matrices=False)
 H=((U[:,:d]*S[:d])@Vt[:d])/np.sqrt(w)[:,None]*np.sqrt(rho)
 B=Vt[:d].T/np.sqrt(rho)[:,None]
 return H,S,B

def kl(P,Q,w=None):
 if w is None:w=np.full(len(P),1/len(P))
 return float(np.sum(w[:,None]*(xlogy(P,P)-xlogy(P,Q))))
def bkl(p,q):return xlogy(p,p/q)+xlogy(1-p,(1-p)/(1-q))
def bound(P,i,j,k,w=None):
 if w is None:w=np.full(len(P),1/len(P))
 p,q=P[i,k],P[j,k];m=(w[i]*p+w[j]*q)/(w[i]+w[j])
 if m==0 or m==1:return 0.
 return float(w[i]*bkl(p,m)+w[j]*bkl(q,m))
def softmaxfit(P,d,restarts=6,w=None,rho=None,iters=1000):
 m,n=P.shape
 if w is None:w=np.full(m,1/m)
 if rho is None:rho=np.full(n,1/n)
 baseline=np.log(rho)
 best=None
 rng=np.random.default_rng(192)
 for rep in range(restarts):
  logP=np.log(np.maximum(P,1e-4));logP-=logP.mean(1,keepdims=True)
  if rep==0:
   U,S,Vt=np.linalg.svd(logP,full_matrices=False); F=U[:,:d]*np.sqrt(S[:d]);B=Vt[:d].T*np.sqrt(S[:d])
  elif rep==1 and d>=2:
   theta=2*np.pi*np.arange(n)/n;B=np.zeros((n,d)); B[:,0]=np.cos(theta);B[:,1]=np.sin(theta);F=np.zeros((m,d));F[:,:2]=4*B[np.argmax(P,axis=1),:2]
  else:F=rng.normal(size=(m,d));B=rng.normal(size=(n,d))
  def fun(x):
   F=x[:m*d].reshape(m,d);B=x[m*d:].reshape(n,d);L=F@B.T+baseline;L-=logsumexp(L,axis=1,keepdims=True);Q=np.exp(L); E=w[:,None]*(Q-P)
   return -np.sum(w[:,None]*P*L),np.r_[(E@B).ravel(),(E.T@F).ravel()]
  opt=minimize(fun,np.r_[F.ravel(),B.ravel()],method='L-BFGS-B',jac=True,options={'maxiter':iters,'ftol':1e-13,'gtol':1e-9})
  F=opt.x[:m*d].reshape(m,d);B=opt.x[m*d:].reshape(n,d);L=F@B.T+baseline;Q=np.exp(L-logsumexp(L,axis=1,keepdims=True));loss=kl(P,Q,w)
  if best is None or loss<best['kl']: best={'kl':loss,'Q':Q,'F':F,'B':B,'niter':opt.nit,'success':bool(opt.success)}
 return best

def explore():
 records=[]
 for n in range(5,10):
  families={'chain':np.r_[np.arange(1,n),n-1],'cycle':np.r_[np.arange(1,n),0],'fork':np.r_[np.arange(1,n-1),n-2,n-2]}
  for name,nexts in families.items():
   for g in [.01,.03,.1,.2,.3,.5,.7,.9]:
    P=succ(nexts,g)
    for d in [2,3]:
     H,S,B=svdfit(P,d)
     wrong=[]
     for k in range(n):
      for i in range(n):
       for j in range(i+1,n):
        gap=P[i,k]-P[j,k];hgap=H[i,k]-H[j,k]
        if gap*hgap < -1e-8:wrong.append({'i':i,'j':j,'k':k,'true':float(gap),'fb':float(hgap),'bound':bound(P,i,j,k)})
     if not wrong:continue
     best=max(wrong,key=lambda x:x['bound']);rec={'n':n,'family':name,'nexts':nexts.tolist(),'gamma':g,'d':d,'sv':S.tolist(),'spherical_error':float(np.ptp(np.sum(B**2,axis=1))),'wrong':best}
     records.append(rec)
 records.sort(key=lambda r:r['wrong']['bound'],reverse=True)
 (OUT/'svd_search.json').write_text(json.dumps(records,indent=2))
 for r in records[:16]:
  P=succ(r['nexts'],r['gamma']);fit=softmaxfit(P,r['d'],restarts=4);r['density_kl']=fit['kl'];r['certificate_ratio']=fit['kl']/r['wrong']['bound'];i,j,k=[r['wrong'][key] for key in ['i','j','k']];r['density_gap']=float(fit['Q'][i,k]-fit['Q'][j,k]);r['P']=P.tolist();r['density_Q']=fit['Q'].tolist();r['density_F']=fit['F'].tolist();r['density_B']=fit['B'].tolist();print({key:r[key] for key in ['n','family','gamma','d','wrong','density_kl','certificate_ratio','density_gap']},flush=True)
 (OUT/'fitted_candidates.json').write_text(json.dumps(records[:16],indent=2))
if __name__=='__main__':explore()
