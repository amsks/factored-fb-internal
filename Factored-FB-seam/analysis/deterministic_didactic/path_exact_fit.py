from search import *
p=np.eye(4)[np.stack((np.maximum(0,np.arange(4)-1),np.minimum(3,np.arange(4)+1)),axis=1)];T=p.mean(1);records=[]
for g in [.5,.8,.9,.99]:
 P=(1-g)*p.reshape(-1,4)@np.linalg.inv(np.eye(4)-g*T);H,S,B=svdfit(P,2);fit=softmaxfit(P,2,restarts=8,iters=2000)
 r={'gamma':g,'P':P.tolist(),'H':H.tolist(),'density_Q':fit['Q'].tolist(),'density_F':fit['F'].tolist(),'density_B':fit['B'].tolist(),'density_KL':fit['kl'],'density_raw_action_gaps':((fit['Q'][:,1].reshape(4,2)[:,1]-fit['Q'][:,1].reshape(4,2)[:,0])/(1-g)).tolist(),'true_raw_action_gaps':((P[:,1].reshape(4,2)[:,1]-P[:,1].reshape(4,2)[:,0])/(1-g)).tolist(),'FB_raw_action_gaps':((H[:,1].reshape(4,2)[:,1]-H[:,1].reshape(4,2)[:,0])/(1-g)).tolist(),'S':S.tolist(),'backward_norms':np.linalg.norm(B,axis=1).tolist(),'wrong_state0_kl_bound':bound(P,0,1,1),'wrong_state2_kl_bound':bound(P,4,5,1)}
 records.append(r);print({k:r[k] for k in ['gamma','density_KL','density_raw_action_gaps','true_raw_action_gaps','FB_raw_action_gaps','wrong_state0_kl_bound']},flush=True)
 (OUT/'path_exact_fit.json').write_text(json.dumps(records,indent=2))
