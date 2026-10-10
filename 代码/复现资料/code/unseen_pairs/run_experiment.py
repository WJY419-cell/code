"""Predeclared unseen-pair experiment; original manuscript and solver are read-only."""
from pathlib import Path
import os,sys,json,csv,hashlib,importlib.util,time
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
from scipy.sparse.csgraph import connected_components

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('MPC_UNSEEN_OUT',str(ROOT/'supplement/unseen_pairs'))).expanduser();OUT.mkdir(parents=True,exist_ok=True)
SOURCE=ROOT
os.environ['MPC_FIT_AUDIT']=str(OUT/'optimizer_checks.json')
os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/matplotlib-cache')
spec=importlib.util.spec_from_file_location('mpc_unseen',SOURCE/'code/metapath_coupling_experiment.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
SEEDS=list(range(12));FRACTIONS=[.1,.2,.3];MODES=['within_layer','global']
GRID=[0,.01,.05,.2,.8]
def dump(name,rows):
 with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def select(c,mask):return tuple(x[mask] for x in c)
def encoded(c,n):return c[0]*n+c[1]
def membership(c,mask,n):return mask[c[2],encoded(c,n)]
def masks(n,K,mode,fraction,seed,excluded=None):
 rng=np.random.RandomState(seed);pairs=np.array([i*n+j for i in range(n) for j in range(i+1,n)])
 mask=np.zeros((K,n*n),bool)
 if mode=='global':
  available=pairs if excluded is None else pairs[~excluded[0,pairs]]
  take=rng.permutation(available)[:max(1,int(round(fraction*len(available))))];mask[:,take]=True
 else:
  for k in range(K):
   available=pairs if excluded is None else pairs[~excluded[k,pairs]]
   take=rng.permutation(available)[:max(1,int(round(fraction*len(available))))];mask[k,take]=True
 return mask
def graph(ds,records,excluded):
 if not ds.name.startswith('MovieLens'):return ds.similarity
 n=ds.n_items;K=len(ds.layer_names);coverage=np.zeros((K,n*n))
 for r in records:
  for a in range(len(r.items)):
   for b in range(a+1,len(r.items)):
    i,j=sorted([int(r.items[a]),int(r.items[b])]);idx=i*n+j
    if not excluded[r.layer,idx]:coverage[r.layer,idx]=1
 norm=np.linalg.norm(coverage,axis=1)
 return coverage@coverage.T/(norm[:,None]*norm[None,:]+1e-12)
def components(c,n,K):
 vals=[]
 for k in range(K):
  A=np.zeros((n,n));sel=c[2]==k;A[c[0][sel],c[1][sel]]=1;A+=A.T
  vals.append(connected_components(A,directed=False,return_labels=False))
 return max(vals),sum(v>1 for v in vals)
def leakage_checks(full,fit,val,te,outer,inner,n,mode):
 assert not membership(full,outer,n).any()
 assert not membership(fit,outer|inner,n).any()
 assert membership(val,inner,n).all() and not membership(val,outer,n).any()
 assert membership(te,outer,n).all()
 a=set(zip(full[0],full[1],full[2]));b=set(zip(te[0],te[1],te[2]));assert not a&b
 if mode=='global':assert not set(zip(full[0],full[1]))&set(zip(te[0],te[1]))
 return 0
def main():
 protected=[SOURCE/'code/metapath_coupling_experiment.py']
 protected.extend(path for path in [SOURCE/'main.tex',SOURCE/'main.pdf'] if path.is_file())
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
 protocol={'date':'2026-10-05','seeds':SEEDS,'primary_mask_fraction':.2,'sensitivity_mask_fractions':[.1,.3],'modes':MODES,'lambda_s_candidates':GRID,'ridge_u':.03,'ridge_b':.005,'outer_split':'original user/ranking split, original per-layer caps','outer_masks':'nested random subsets of all fixed-catalog unordered pairs; independent of directions and performance','inner_validation':'separate user records and additional same-mode pair mask among outer-unmasked pairs; same masking fraction','final_fit':'all outer training records after outer mask only','MovieLens_graph':'training-record item co-occurrence only, including ties without reading their direction; masked pairs removed; inner graph rebuilt from inner training records','catalog':'original fixed catalog and layers; no new-item or purely inductive-catalog claim','selection':'mean NLL; ties choose first candidate, including zero','metric':'comparison-count-weighted NLL primary, accuracy secondary','methods':['Pooled-BT','Layerwise-BT','Uncoupled-Joint','MPC-Rank','RankCentrality','Borda'],'inference':'paired split summaries; overlapping resplits not independent populations; Holm correction across all 90 NLL comparisons','protected_hashes_before':hashes}
 (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
 rows=[];audits=[];tuning=[];start=time.time();saved={}
 for ds in [m.breakfast_dataset(),m.sushi_dataset(),m.movielens_dataset()]:
  n=ds.n_items;K=len(ds.layer_names)
  for seed in SEEDS:
   train,test=m.split_records(ds,seed,ds.test_fraction)
   if ds.train_cap_per_layer:train=m.cap_training_records(train,K,ds.train_cap_per_layer,seed)
   assert not {r.unit for r in train}&{r.unit for r in test}
   raw=m.counts_from_records(train,n,K);rawtest=m.counts_from_records(test,n,K)
   innerds=m.Dataset(ds.name,train,n,ds.layer_names,ds.similarity,ds.aligned_units,.2,None)
   itr,iva=m.split_records(innerds,seed+10007,.2)
   assert not {r.unit for r in itr}&{r.unit for r in iva}
   cit=m.counts_from_records(itr,n,K);civ=m.counts_from_records(iva,n,K)
   for mode in MODES:
    for frac in FRACTIONS:
     outer=masks(n,K,mode,frac,seed+310019)
     inner=masks(n,K,mode,frac,seed+720031,outer)
     tr=select(raw,~membership(raw,outer,n));te=select(rawtest,membership(rawtest,outer,n))
     fit=select(cit,~membership(cit,outer|inner,n));val=select(civ,membership(civ,inner,n)&~membership(civ,outer,n))
     assert all(len(c[0]) and (c[3]+c[4]).sum()>0 for c in [tr,te,fit,val])
     leakage_checks(tr,fit,val,te,outer,inner,n,mode)
     S=graph(ds,train,outer);Si=graph(ds,itr,outer|inner)
     key=f'{ds.name}_{seed}_{mode}_{frac}'
     saved[key+'_outer']=outer;saved[key+'_inner']=inner
     best=(float('inf'),None)
     for lam in GRID:
      u,b=m.fit_joint(fit,n,K,Si,.03,.005,lam,steps=400)
      loss=m.pair_metrics(val,u,b)[1]
      tuning.append({'dataset':ds.name,'seed':seed,'mode':mode,'fraction':frac,'lambda_s':lam,'validation_nll':loss})
      if loss<best[0]:best=(loss,lam)
     methods={'Pooled-BT':(m.fit_pooled(tr,n,K,steps=650),None),
      'Layerwise-BT':m.fit_layerwise(tr,n,K,steps=600),
      'Uncoupled-Joint':m.fit_joint(tr,n,K,S,.03,.005,0,steps=750),
      'MPC-Rank':m.fit_joint(tr,n,K,S,.03,.005,best[1],steps=750),
      'RankCentrality':m.calibrate_scale(tr,m.rank_centrality(tr,n)),
      'Borda':m.calibrate_scale(tr,m.borda_scores(tr,n))}
     cnt=float((te[3]+te[4]).sum());known=np.unique(encoded(tr,n));donor=np.isin(encoded(te,n),known)
     maxcomp,disconnected=components(tr,n,K)
     audits.append({'dataset':ds.name,'seed':seed,'mode':mode,'fraction':frac,'train_comparisons':float((tr[3]+tr[4]).sum()),'test_comparisons':cnt,'test_layer_pairs':len(te[0]),'test_distinct_pairs':len(np.unique(encoded(te,n))),'donor_coverage_fraction':float(np.sum((te[3]+te[4])[donor])/cnt),'inner_validation_comparisons':float((val[3]+val[4]).sum()),'max_item_components_per_layer':maxcomp,'disconnected_layers':disconnected,'leakage_violations':0})
     for method,(u,b) in methods.items():
      acc,loss=m.pair_metrics(te,u,b)
      rows.append({'dataset':ds.name,'seed':seed,'mode':mode,'fraction':frac,'method':method,'test_nll':loss,'accuracy':acc,'test_comparisons':cnt,'lambda_s':best[1] if method=='MPC-Rank' else ''})
   print(ds.name,'seed',seed,'elapsed',round(time.time()-start,1),flush=True)
   dump('per_seed.csv',rows);dump('split_audit.csv',audits);dump('tuning.csv',tuning)
 np.savez_compressed(OUT/'pair_masks.npz',**saved)
 after={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected};assert hashes==after
 (OUT/'completion.json').write_text(json.dumps({'seconds':time.time()-start,'rows':len(rows),'scenarios':len(audits),'protected_files_unchanged':True,'hashes_after':after},indent=2),encoding='utf-8')
 print('COMPLETE',len(rows),len(audits),flush=True)
if __name__=='__main__':main()
