"""Small deterministic checks; no training and no parameter search."""
import json
import numpy as np
from scipy.optimize import minimize

np.set_printoptions(precision=12, suppress=True)
X = np.array([1., 1., -1., -1.])
Y = np.array([1., -1., 1., -1.])
C = np.array([.49, .49, .01, .01])

def bkl(p, q):
    return p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))

def kl(P, Q):
    return np.mean(np.sum(P*np.log(P/Q), axis=1))

def log_singular(Q):
    L=np.log(Q)
    return np.linalg.svd(L-L.mean(axis=1, keepdims=True), compute_uv=False)

def weighted_fit(P, rho):
    T=.5*P/np.sqrt(rho)
    left,s,right=np.linalg.svd(T, full_matrices=False)
    Q=2*((left[:,:2]*s[:2])@right[:2])*np.sqrt(rho)
    return {
        'rho':rho.tolist(), 'weighted_singular_values':s.tolist(), 'Qls':Q.tolist(),
        'goal_gaps':(Q[[0,2],2]-Q[[1,3],2]).tolist(),
        'weighted_fb_error':float(np.mean(np.sum((P-Q)**2/rho,axis=1))),
        'KL_ls':float(kl(P,Q)),
        'max_row_sum_error':float(np.max(np.abs(Q.sum(axis=1)-1))),
        'minimum_probability':float(Q.min()),
    }

def report(name, u, v):
    P=C+X[:,None]*u+Y[:,None]*v
    Qgood=C+Y[:,None]*v
    left,s,right=np.linalg.svd(P, full_matrices=False)
    Qls=(left[:,:2]*s[:2])@right[:2]
    bounds=[]
    checked=[]
    for rows in [(0,1),(2,3)]:
        gp,gm=P[list(rows),2]
        bar=(gp+gm)/2
        exact=(bkl(gp,bar)+bkl(gm,bar))/4
        res=minimize(lambda q:(bkl(gp,q[0])+bkl(gm,q[1]))/4,
                     np.array([bar,bar]), method='SLSQP',
                     bounds=[(1e-10,1-1e-10)]*2,
                     constraints={'type':'ineq','fun':lambda q:q[1]-q[0]},
                     options={'ftol':1e-14,'maxiter':50})
        bounds.append(float(exact))
        checked.append({'value':float(res.fun),'q':res.x.tolist(),'success':bool(res.success)})
    # The zero-sum paired basis is [(1,-1,0,0),(0,0,1,-1)]/sqrt(2).
    A,B=np.sqrt(2)*u[[0,2]]
    D,E=np.sqrt(2)*v[[0,2]]
    H=np.array([[A*A+D*D,A*B+D*E],[A*B+D*E,B*B+E*E]])
    vals,vec=np.linalg.eigh(H)
    w=vec[:,-1]
    if w[0]<0: w=-w
    result={
        'name':name, 'P':P.tolist(), 'Qls':Qls.tolist(),'Qgood':Qgood.tolist(),
        'P_singular_values':s.tolist(),
        'ls_weighted_fb_error':float(np.sum((P-Qls)**2)),
        'good_weighted_fb_error':float(np.sum((P-Qgood)**2)),
        'ls_mean_probability_squared_error':float(np.mean(np.sum((P-Qls)**2,axis=1))),
        'good_mean_probability_squared_error':float(np.mean(np.sum((P-Qgood)**2,axis=1))),
        'KL_good':float(kl(P,Qgood)), 'KL_ls':float(kl(P,Qls)),
        'wrong_order_KL_bounds':bounds,
        'constrained_bernoulli_check':checked,
        'true_goal_gaps':(P[[0,2],2]-P[[1,3],2]).tolist(),
        'ls_goal_gaps':(Qls[[0,2],2]-Qls[[1,3],2]).tolist(),
        'good_goal_gaps':(Qgood[[0,2],2]-Qgood[[1,3],2]).tolist(),
        'true_centered_log_singular':log_singular(P).tolist(),
        'good_centered_log_singular':log_singular(Qgood).tolist(),
        'good_probability_singular':np.linalg.svd(Qgood,compute_uv=False).tolist(),
        'column_covariance_2d':H.tolist(),
        'column_top_eigenvector_2d':w.tolist(),
        'action_projection_top':float(np.array([D,E])@w),
        'sum_error_ls':float(np.max(np.abs(Qls.sum(axis=1)-1))),
        'minimum_probability_ls':float(Qls.min()),
    }
    result['natural_reference_fit']=weighted_fit(P,C)
    for rho_name,rho in [('uniform',np.full(4,.25)),('natural',C)]:
        # Row features encode the two actions. Column features are unrestricted.
        F=np.column_stack((Y==1,Y==-1)).astype(float)
        B_probability=(Qgood[[0,1]]/rho).T
        B_log=np.log(Qgood[[0,1]]/rho).T
        density_product=F@B_probability.T
        probabilities=rho*np.exp(F@B_log.T)
        probabilities/=probabilities.sum(axis=1,keepdims=True)
        assert np.max(np.abs(density_product*rho-Qgood))<1e-12
        assert np.max(np.abs(probabilities-Qgood))<1e-12
        result[rho_name+'_true_centered_log_density_singular']=log_singular(P/rho).tolist()
        result[rho_name+'_good_log_density_rank']=int(np.linalg.matrix_rank(F@B_log.T))
        result[rho_name+'_good_weighted_fb_error']=float(np.mean(np.sum((P-Qgood)**2/rho,axis=1)))
    # Exact-decimal bounds certify the retained column direction has negative
    # goal coordinate but a positive projection of the action direction.
    if name=='strict':
        raw_H=H/2
        kappa_upper=abs(raw_H[0,1])/(raw_H[0,0]-raw_H[1,1])
        result['kappa_upper_bound']=float(kappa_upper)
        result['action_projection_positive_if_kappa_below']=float(v[0]/v[2])
        assert 0<kappa_upper<v[0]/v[2]
    assert np.all(P>0) and np.all(Qgood>0) and np.all(Qls>0)
    assert np.max(np.abs(P.sum(axis=1)-1))<1e-12
    assert np.max(np.abs(Qls.sum(axis=1)-1))<1e-12
    assert min(bounds)>kl(P,Qgood)
    assert log_singular(P)[2]>1e-6
    assert log_singular(Qgood)[2]<1e-12
    assert np.linalg.svd(Qgood, compute_uv=False)[2]<1e-12
    assert s[1]>s[2]
    if name=='strict':
        assert np.all(Qls[[0,2],2]-Qls[[1,3],2]<0)
    else:
        assert np.max(np.abs(Qls[[0,2],2]-Qls[[1,3],2]))<1e-12
    return result

reports=[
    report('baseline', np.array([.0196,-.0196,0.,0.]),np.array([0.,0.,.009,-.009])),
    report('strict', np.array([.0196,-.0196,-.0008,.0008]),np.array([.001,-.001,.009,-.009])),
]
print(json.dumps(reports,indent=2))
