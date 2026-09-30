import re, math
src = open('content/experiments.tex').read()
cell = re.compile(r'(?:\\textbf\{)?([\d.]+)\}?\\pmse\{([\d.]+)\}\}?')
groups = {'OGBench':['cube-single','cube-double','scene'],
          'MetaWorld':['push','bin-picking','box-close'],
          'ManiSkill':['push-cube','pick-cube','stack-cube','poke-cube','lift-peg'],
          'ExoRL':['walker','quadruped','point-mass','cheetah']}
rows={}
for line in src.splitlines():
    m=re.match(r'\s*\\texttt\{([\w-]+)\}\s*&(.*)\\\\',line)
    if not m: continue
    cells=[c.strip() for c in m.group(2).split('&')][:7]
    vals=[]
    for c in cells:
        if c=='--': vals.append(None); continue
        mm=cell.search(c); vals.append((float(mm.group(1)),float(mm.group(2))))
    rows[m.group(1)]=vals
def delta(v):
    best=max(x[0] for x in v[:6] if x); return 100*(v[6][0]-best)/best
for g,ts in groups.items():
    agg=[]
    for j in range(7):
        if rows[ts[0]][j] is None: agg.append(None); continue
        mu=sum(rows[t][j][0] for t in ts)/len(ts)
        se=math.sqrt(sum(rows[t][j][1]**2 for t in ts))/len(ts)
        agg.append((mu,se))
    print(g, [(round(a[0],1),round(a[1],1)) if a else None for a in agg], 'delta %.1f'%delta(agg))
    for t in ts: print('   ',t,'delta %.1f'%delta(rows[t]))
