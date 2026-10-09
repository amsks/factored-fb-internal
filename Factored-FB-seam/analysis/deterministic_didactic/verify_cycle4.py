"""Exact deterministic-cycle baseline; explicit verification, no optimization.
Four states. Actions stay and advance clockwise. Continuation always advances.
Uniform state-action rows and uniform next-state/stationary reference.
Squared fit is an exact-successor rank-three projection, not a TD loss claim.
Density rank-three realizes the full target exactly in this baseline.
"""
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp


def verify(gamma=.5):
 n=4;d=3;C=np.roll(np.eye(n),1,axis=1);rho=np.full(n,1/n)
 D=(1-gamma)*np.linalg.inv(np.eye(n)-gamma*C)
 successor=np.stack((np.arange(n),(np.arange(n)+1)%n),axis=1).ravel()
 P=D[successor];w=np.full(2*n,1/(2*n))
 Z=np.sqrt(w)[:,None]*P/np.sqrt(rho)
 U,S,Vt=np.linalg.svd(Z,full_matrices=False)
 H=(U[:,:d]*S[:d])@Vt[:d]/np.sqrt(w)[:,None]*np.sqrt(rho)
 theta=np.arange(n)*np.pi/2
 Bfb=np.c_[np.ones(n),np.sqrt(2)*np.cos(theta),np.sqrt(2)*np.sin(theta)]
 Ffb=H@Bfb
 np.testing.assert_allclose(Ffb@Bfb.T*rho,H,atol=1e-13)
 np.testing.assert_allclose(Bfb.T@(rho[:,None]*Bfb),np.eye(d),atol=1e-13)
 np.testing.assert_allclose(np.sum(Bfb**2,axis=1),d,atol=1e-13)
 Bctrl=np.c_[np.sqrt(2)*np.cos(theta),np.sqrt(2)*np.sin(theta),(-1.)**np.arange(n)]
 Fctrl=P@Bctrl
 Hctrl=Fctrl@Bctrl.T*rho
 np.testing.assert_allclose(Hctrl,P-rho,atol=1e-13)
 np.testing.assert_allclose(Hctrl[1::2]-Hctrl[::2],P[1::2]-P[::2],atol=1e-13)
 np.testing.assert_allclose(Bctrl.T@(rho[:,None]*Bctrl),np.eye(d),atol=1e-13)
 np.testing.assert_allclose(np.sum(Bctrl**2,axis=1),d,atol=1e-13)
 logp=np.log(P);logp-=logp.mean(axis=1,keepdims=True)
 Fdensity=logp@Bctrl/n
 logits=Fdensity@Bctrl.T
 Q=np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
 np.testing.assert_allclose(Q,P,atol=1e-13)
 truegap=P[3,0]-P[2,0];fbgap=H[3,0]-H[2,0]
 np.testing.assert_allclose(truegap,(1-gamma)*gamma**2/((1+gamma)*(1+gamma**2)),atol=1e-13)
 np.testing.assert_allclose(fbgap,-(1-gamma)**2/(2*(1+gamma**2)),atol=1e-13)
 assert truegap>0>fbgap
 assert S[2]>S[3]
 if gamma==.5:np.testing.assert_allclose(H[[0,2,4,6]],[[.45,.35,.05,.15],[.15,.45,.35,.05],[.05,.15,.45,.35],[.35,.05,.15,.45]],atol=1e-13)
 return {'states':list(range(n)),'actions':['stay','advance'],'transition_successors':successor.reshape(n,2).tolist(),'continuation_policy':'advance at every state','gamma':gamma,'rank':d,'row_weights':w.tolist(),'reference':rho.tolist(),'reward':[1,0,0,0],'diagnostic_start':1,'normalized_successors':P.tolist(),'FB_fit':H.tolist(),'FB_forward':Ffb.tolist(),'FB_backward':Bfb.tolist(),'singular_values':S.tolist(),'FB_normalized_gap_advance_minus_stay':float(fbgap),'true_normalized_gap_advance_minus_stay':float(truegap),'FB_raw_gap_advance_minus_stay':float(fbgap/(1-gamma)),'true_raw_gap_advance_minus_stay':float(truegap/(1-gamma)),'control_fit':Hctrl.tolist(),'control_forward':Fctrl.tolist(),'control_backward':Bctrl.tolist(),'density_logits':logits.tolist(),'density_forward':Fdensity.tolist(),'density_backward':Bctrl.tolist(),'density_fit':Q.tolist(),'target_probability_rank':int(np.linalg.matrix_rank(P)),'target_centered_log_rank':int(np.linalg.matrix_rank(logp)),'true_greedy_actions':np.argmax(P[:,0].reshape(n,2),axis=1).tolist(),'FB_greedy_actions':np.argmax(H[:,0].reshape(n,2),axis=1).tolist(),'density_greedy_actions':np.argmax(Q[:,0].reshape(n,2),axis=1).tolist(),'control_fit_can_be_negative':True,'all_assertions_passed':True}

if __name__=='__main__':
 for g in [.01,.1,.25,.5,.75,.9,.99]:verify(g)
 result=verify()
 Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))
