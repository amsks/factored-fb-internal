import importlib.util, os, sys, pathlib
os.environ.setdefault("JAX_PLATFORMS","cpu")
import numpy as np
REPO=pathlib.Path('.').resolve(); HERE=REPO/'analysis/single_bottleneck'
spec=importlib.util.spec_from_file_location("sb_run",HERE/"run.py"); sb=importlib.util.module_from_spec(spec)
argv=sys.argv; sys.argv=[argv[0]]; spec.loader.exec_module(sb); sys.argv=argv
fam,bf=sb.fam,sb.bf; G=0.8
mdp=sb.fixed_processes()[1]; n=mdp['n']; b=mdp['decision']; rho=mdp['rho']; r=mdp['r']
T=mdp['t_exact']                      # rows of (1-gamma) M
Qtrue=T@r/(1-G); gap=Qtrue[2*b]-Qtrue[2*b+1]; R=r.max()-r.min()
eps=(1-G)**2*gap**2/(2*R**2)
print(f"n={n} b={b} Q(a+)={Qtrue[2*b]:.3f} Q(a-)={Qtrue[2*b+1]:.3f} gap={gap:.3f} eps={eps:.4f}")
X=n*mdp['m_exact']; dm=X[2*b]-X[2*b+1]; W=np.sqrt(rho)[:,None]
def kl(p,q): m=p>0; return float(np.sum(p[m]*np.log(p[m]/q[m])))
for d in (2,3):
    cfg,pred,F,B=fam.train_group(mdp,d,'fb')
    for i in range(len(cfg)):
        P=bf.readout(pred[i],'fb',G,n); q=P@r/(1-G)
        Bi=np.asarray(B[i],float); mom=Bi.T@(rho*dm)       # lemma condition
        proj=Bi@np.linalg.lstsq(W*Bi,np.sqrt(rho)*dm,rcond=None)[0]
        ghat=float(np.sum(rho*proj*r))/(1-G)*(1-G)          # refit value gap
        print(f"FB d={d} {cfg[i]} Q+={q[2*b]:.2f} Q-={q[2*b+1]:.2f} |B-moment diff|={np.linalg.norm(mom):.2e} refit gap={ghat:.3f}")
    cfg,pred,F,B=fam.train_group(mdp,d,'density')
    for i in range(len(cfg)):
        P=np.asarray(pred[i],float); q=P@r/(1-G)
        k1,k2=kl(T[2*b],P[2*b]),kl(T[2*b+1],P[2*b+1])
        print(f"DFB d={d} {cfg[i]} Q+={q[2*b]:.2f} Q-={q[2*b+1]:.2f} KL+={k1:.4f} KL-={k2:.4f} below eps: {max(k1,k2)<eps}")
