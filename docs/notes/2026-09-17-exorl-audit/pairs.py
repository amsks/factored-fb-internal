import numpy as np, glob, os, sys
ROOT='/home/adityamohan/exorl_data/exorl/datasets'
CACHE='/home/adityamohan/exorl_data/relabel_cache'
SPEC={'walker':(0,24,15.9),'cheetah':(0,17,7.5),'quadruped':(0,78,21.087),'point_mass_maze':(0,2,0.03)}
TASKS={'walker':['stand','walk','run','spin','flip'],'cheetah':['run','run_backward','walk','walk_backward'],
       'quadruped':['stand','walk','run','jump'],'point_mass_maze':['reach_top_left','reach_top_right','reach_bottom_left','reach_bottom_right']}
rng=np.random.default_rng(0)
NEP=200
for dom,(a,b,eps) in SPEC.items():
    files=sorted(glob.glob(f'{ROOT}/{dom}/rnd/buffer/episode_*.npz'))
    nfiles=len(files)
    obs_eps=[np.load(f)['observation'].astype(np.float64) for f in files[:NEP]]
    obs=np.concatenate(obs_eps)  # includes obs[0..T]
    D=obs.shape[1]
    blk=obs[:,a:b]
    # random pairs
    i,j=rng.integers(0,len(blk),400000),rng.integers(0,len(blk),400000); k=i!=j
    d=np.linalg.norm(blk[i[k]]-blk[j[k]],axis=-1)
    # consecutive pairs
    dc=np.concatenate([np.linalg.norm(e[1:,a:b]-e[:-1,a:b],axis=-1) for e in obs_eps])
    # 10-step-apart pairs
    d10=np.concatenate([np.linalg.norm(e[10:,a:b]-e[:-10,a:b],axis=-1) for e in obs_eps])
    d50=np.concatenate([np.linalg.norm(e[50:,a:b]-e[:-50,a:b],axis=-1) for e in obs_eps])
    # per-dim std to see which dims dominate the distance
    sd=blk.std(0)
    print(f"=== {dom}: files={nfiles} obs_dim={D} block {a}:{b} eps={eps}")
    print(f"  random-pair L2: median {np.median(d):.2f} mean {d.mean():.2f}; frac<=eps {(d<=eps).mean():.4f}")
    print(f"  consecutive-step L2: median {np.median(dc):.2f}; frac<=eps {(dc<=eps).mean():.4f}; 10-step frac<=eps {(d10<=eps).mean():.4f} (median {np.median(d10):.2f}); 50-step frac<=eps {(d50<=eps).mean():.4f} (median {np.median(d50):.2f})")
    top=np.argsort(-sd)[:6]
    print(f"  per-dim std: top dims {list(zip(top.tolist(), np.round(sd[top],2).tolist()))}; sum var top6/all = {(sd[top]**2).sum()/(sd**2).sum():.2f}")
    # Gaussian-softmax training reward concentration over a 1024 batch: r_j = softmax(-0.5||u_j-g||^2/eps^2)
    idx=rng.integers(0,len(blk),1024); U=blk[idx]
    gidx=rng.integers(0,len(blk),64); G=blk[gidx]
    lg=-0.5*(((G[:,None,:]-U[None,:,:])/eps)**2).sum(-1)
    lg=lg-lg.max(1,keepdims=True); r=np.exp(lg); r/=r.sum(1,keepdims=True)
    ess=1/(r**2).sum(1); top1=r.max(1)
    print(f"  train-task reward on 1024 candidates: ESS median {np.median(ess):.1f} (min {ess.min():.1f} max {ess.max():.1f}); top-1 mass median {np.median(top1):.3f}")
    # reward linearity: R^2 of ridge(reward ~ obs) and ridge(reward ~ [obs, obs^2]) using cache if it matches 5000 eps
    for t in TASKS[dom]:
        cands=glob.glob(f'{CACHE}/{dom}_rnd_{t}_*.npy')
        if not cands: print(f"  task {t}: no cache"); continue
        rw=None
        for c in cands:
            arr=np.load(c)
            if arr.shape[0] in (5000*1000, 5000*1001, nfiles*1000, nfiles*1001, NEP*1000):
                rw=arr; break
        if rw is None: print(f"  task {t}: cache shapes {[np.load(c).shape for c in cands]} do not match"); continue
        # rewards aligned to next_observations: reward at s_{t+1}; obs_eps each has T+1 rows; loader emits obs[1:]
        n=min(NEP*1000, rw.shape[0])
        nobs=np.concatenate([e[1:,:] for e in obs_eps])[:n]; y=rw[:n].astype(np.float64)
        X=np.concatenate([nobs, np.ones((n,1))],1)
        w=np.linalg.lstsq(X,y,rcond=None)[0]; r2=1-((X@w-y)**2).sum()/((y-y.mean())**2).sum()
        X2=np.concatenate([nobs, nobs**2, np.ones((n,1))],1)
        w2=np.linalg.lstsq(X2,y,rcond=None)[0]; r2b=1-((X2@w2-y)**2).sum()/((y-y.mean())**2).sum()
        print(f"  task {t}: reward mean {y.mean():.3f} std {y.std():.3f} frac>0 {(y>0).mean():.3f} frac>0.5 {(y>0.5).mean():.3f}; R2 linear-in-obs {r2:.3f}, quadratic {r2b:.3f} (n={n}, cache {os.path.basename(c)} shape {arr.shape})")
