"""Bounded full-expectation spherical FB TD search using the archived runner.

One forward head. Orthogonality coefficient 500 matches the relative objective
weight of production's two identical forward heads with coefficient 1000.
Only backward rows are normalized, to sqrt(d); forward outputs are unrestricted.
"""
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
sys.path.insert(0,str(Path('scripts/didactic').resolve()))
import deterministic_td as td


def graph(family,n,d,goal,swap_prob=.4):
 s=np.arange(n)
 if family=='path':nexts=np.stack((np.maximum(0,s-1),np.minimum(n-1,s+1)),axis=1)
 elif family=='cycle':nexts=np.stack(((s-1)%n,(s+1)%n),axis=1)
 elif family=='prism':
  if n!=6:raise ValueError('prism needs six states')
  nexts=np.stack((3*(s//3)+(s+1)%3,(s+3)%6),axis=1)
  return np.eye(n)[nexts],np.tile([1-swap_prob,swap_prob],(n,1)),d,goal
 else:raise ValueError(family)
 return np.eye(n)[nexts],np.full((n,2),.5),d,goal


def greedy_return(prob,p,gamma,goal,start):
 n=p.shape[0];pi=np.eye(2)[np.argmax(prob[:,:,goal],axis=1)];T=np.sum(pi[:,:,None]*p,axis=1);r=T[:,goal];v=np.linalg.solve(np.eye(n)-gamma*T,r)
 opt=np.zeros(n)
 for _ in range(10000):
  new=np.max(p[:,:,goal]+gamma*np.einsum('san,n->sa',p,opt),axis=1)
  if np.max(abs(new-opt))<1e-13:break
  opt=new
 return {'greedy_actions':np.argmax(pi,axis=1).tolist(),'greedy_return':float(v[start]),'optimal_return':float(opt[start]),'return_ratio':float(v[start]/opt[start])}


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--n',type=int,default=4);ap.add_argument('--d',type=int,default=2);ap.add_argument('--family',choices=['path','cycle','prism'],default='path');ap.add_argument('--swap-prob',type=float,default=.4);ap.add_argument('--goal',type=int,default=1);ap.add_argument('--start',type=int,default=0);ap.add_argument('--gamma',nargs='+',type=float,default=[.5,.8,.9]);ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);ap.add_argument('--rates',nargs='+',type=float,default=[.001,.01]);ap.add_argument('--ortho',type=float,default=500.);ap.add_argument('--steps',type=int,default=100000);ap.add_argument('--interval',type=int,default=10000);ap.add_argument('--methods',nargs='+',choices=['fb','density'],default=['fb']);args=ap.parse_args()
 if args.steps%args.interval:ap.error('steps must be divisible by interval')
 args.out.mkdir(parents=True,exist_ok=True)
 if (args.out/'search_config.json').exists():raise FileExistsError(args.out)
 meta={**vars(args),'out':str(args.out),'target_rate':.005,'sphere_fb':True,'sphere_density':False,'one_forward_head':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(td.__file__).read_bytes()).hexdigest(),'normalization':'sqrt(d)*B/||B||; F unrestricted','float64':True}
 (args.out/'search_config.json').write_text(json.dumps(meta,indent=2))
 p,pi,d,goal=graph(args.family,args.n,args.d,args.goal,args.swap_prob)
 td.problem=lambda name:(p,pi,d,goal)
 base_loss=td.loss_fn
 def loss(params,target,kernel,policy,gamma,method,sphere=False):
  ans=base_loss(params,target,kernel,policy,gamma,method,sphere)
  if method=='fb':
   b=td.backward(params,sphere);ortho=.5*jnp.mean((b@b.T)**2)-jnp.mean(jnp.sum(b*b,axis=-1));ans+=(args.ortho-1)*ortho
  return ans
 td.loss_fn=loss
 train_args=SimpleNamespace(out=args.out,problem=args.family+str(args.n),gamma=args.gamma,seeds=args.seeds,rates=args.rates,steps=args.steps,interval=args.interval,tau=.005,sphere=True)
 for method in args.methods:
  td.train(train_args,method)
  a=np.load(args.out/f'{method}.npz');cfg=json.loads((args.out/f'{method}_configs.json').read_text());latest=json.loads((args.out/f'{method}_latest.json').read_text());summary=[]
  for i,c in enumerate(cfg):
   raw=a['predictions'][-1,i];prob=raw if method=='density' else raw*(1-c['gamma'])/args.n;result=greedy_return(prob.reshape(args.n,2,args.n),p,c['gamma'],args.goal,args.start)
   b=a['final_b'][i]
   if method=='fb':b=b*np.sqrt(args.d)/np.linalg.norm(b,axis=1,keepdims=True)
   cov=b.T@b/args.n;E=args.n*np.mean(np.sum((prob-a['exact'][i])**2,axis=1));orth_excess=.5*np.sum((cov-np.eye(args.d))**2)
   result.update(**latest[i],exact_target_squared=float(E),orthogonality_excess=float(orth_excess),reference=(p.mean(axis=(0,1))).tolist(),**{'configured_ortho':args.ortho,'n':args.n,'d':args.d,'family':args.family,'goal':args.goal,'start':args.start})
   result['start_gap']=result['all_state_gaps'][args.start]
   result['gap_at_goal']=result['all_state_gaps'][args.goal]
   summary.append(result)
  (args.out/f'{method}_control_summary.json').write_text(json.dumps(summary,indent=2))
  print(json.dumps({'FINAL_CONTROL':[{k:r[k] for k in ['gamma','seed','lr','start_gap','greedy_actions','return_ratio','gradient_norm']} for r in summary],'method':method}),flush=True)
if __name__=='__main__':main()
