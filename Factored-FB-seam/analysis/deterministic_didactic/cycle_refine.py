from search import *
records=[]
for n,d in [(5,3),(7,5),(9,7)]:
 for g in [.02,.05,.1,.2,.3,.35,.4,.45,.475,.5,.525,.55,.575,.6]:
  P=succ(np.r_[np.arange(1,n),0],g);H,S,B=svdfit(P,d)
  bad=[]
  for delta in range(1,n):
   for k in range(n):
    tg=P[0,k]-P[delta,k];fg=H[0,k]-H[delta,k]
    if tg*fg < -1e-8:bad.append({'delta':delta,'goal':k,'true_gap':float(tg),'fb_gap':float(fg),'bound':bound(P,0,delta,k)})
  if not bad:continue
  case=max(bad,key=lambda b:b['bound']);fit=softmaxfit(P,d,restarts=4,iters=1500)
  r={'n':n,'d':d,'g':g,'case':case,'kl':fit['kl'],'ratio':fit['kl']/case['bound'],'density_gap':float(fit['Q'][0,case['goal']]-fit['Q'][case['delta'],case['goal']]),'Q':fit['Q'].tolist(),'F':fit['F'].tolist(),'B':fit['B'].tolist(),'P':P.tolist(),'H':H.tolist(),'S':S.tolist()}
  records.append(r);print({k:r[k] for k in ['n','d','g','case','kl','ratio','density_gap']},flush=True)
  (OUT/'cycle_refine.json').write_text(json.dumps(records,indent=2))
