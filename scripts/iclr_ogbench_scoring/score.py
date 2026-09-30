import pandas as pd, numpy as np, os, glob
H='/tmp/claude-1001/-home-adityamohan-git-Factored-FB-seam/6f977769-662c-4feb-8ac6-21bbfa3a994d/scratchpad/hub/exp'
def score(p):
    if not os.path.exists(p): return None
    d=pd.read_csv(p); g=d.groupby('step')['success'].mean().sort_index()
    return 100*g.iloc[-3:].mean(), int(g.index[-1]), len(d[d.step==g.index[-1]])
envs=['cube-single','cube-double','scene']
def src(env,crit,seed,kind,root=None):
    r=f'{env}-play-v0_{crit}_flowbc_seed_{seed}'
    c=[f'{H}/{r}',f'{H}/tdjtab/{r}',f'{H}/ortho1/{r}',f'exp/tdjtab/{r}',f'exp/{r}'] if root is None else [f'{x}/{r}' for x in root]
    for x in c:
        p=f'{x}/{kind}.csv'
        if os.path.exists(p): return p
arms={'fb_tdj':('fb_tdj','eval_relabel',None),'hilp_tdjc':('hilp_tdjc','eval_relabel',None),
 'rldp_o1':('rldp','eval_relabel',['exp/ortho1',f'{H}/ortho1']),
 'dfb_z':('density_fb_z','eval_relabel',None),'dfb_goal':('density_fb','eval',None)}
for a,(crit,kind,root) in arms.items():
    for env in envs:
        rs=[]
        for s in range(5):
            p=src(env,crit,s,kind,root); v=score(p) if p else None
            rs.append(v)
        vals=[v[0] for v in rs if v]
        m=np.mean(vals) if vals else float('nan'); sd=np.std(vals,ddof=1) if len(vals)>1 else float('nan')
        print(f'{a:10s} {env:12s} n={len(vals)} mean={m:5.1f} sd={sd:4.1f} se={sd/np.sqrt(len(vals)) if vals else 0:4.1f} per-seed',
              [None if v is None else (round(v[0],1),v[1]) for v in rs])
