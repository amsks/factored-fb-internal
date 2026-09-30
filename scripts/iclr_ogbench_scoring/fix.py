import re, math
exec(open('/tmp/claude-1001/-home-adityamohan-git-Factored-FB-seam/6f977769-662c-4feb-8ac6-21bbfa3a994d/scratchpad/agg.py').read().split("for g,ts in groups.items():")[0])
RED,NEU,GRN=(210,20,20),(100,100,100),(0,150,40); CAP=25.0
def col(d):
    t=max(-1,min(1,d/CAP)); a,b=(NEU,GRN) if t>=0 else (NEU,RED); t=abs(t)
    return ','.join(str(round(a[i]+(b[i]-a[i])*t)) for i in range(3))
def dcell(d):
    s='$+$' if d>=0 else '$-$'
    return r'\textcolor[RGB]{%s}{(%s%.1f\%%)}'%(col(d),s,abs(d))
def fmt(v,best,bse,dec=1):
    if v is None: return '--'
    m='%.1f'%v[0]; b=v[0]>=best-bse-1e-9
    return (r'\textbf{%s}'%m if b else m)+r'\pmse{%.1f}'%v[1]
def row(label,v):
    best_i=max((j for j in range(7) if v[j]),key=lambda j:v[j][0])
    best,bse=v[best_i]
    return '    %-38s & '%label+' & '.join(fmt(x,best,bse) for x in v)+' & '+dcell(delta(v))+r' \\'
names={'OGBench':r'\textit{OGBench (success rate, \%)}','MetaWorld':r'\textit{MetaWorld (success rate, \%)}',
       'ManiSkill':r'\textit{ManiSkill (success rate, \%)}','ExoRL':r'\textit{ExoRL (DMC returns)}'}
out=[]
for g,ts in groups.items():
    agg=[None if rows[ts[0]][j] is None else (sum(rows[t][j][0] for t in ts)/len(ts),
         math.sqrt(sum(rows[t][j][1]**2 for t in ts))/len(ts)) for j in range(7)]
    agg=[None if a is None else (round(a[0],1),round(a[1],1)) for a in agg]
    out+=[r'    \midrule',row(names[g],agg),r'    \midrule']+[row(r'\texttt{%s}'%t,rows[t]) for t in ts]
body='\n'.join(out[1:])
lines=src.split('\n')
s=next(i for i,l in enumerate(lines) if l.strip().startswith(r'\textit{OGBench'))
e=next(i for i,l in enumerate(lines) if l.strip()==r'\bottomrule')
lines[s:e]=body.split('\n')
open('content/experiments.tex','w').write('\n'.join(lines))
print(body)
