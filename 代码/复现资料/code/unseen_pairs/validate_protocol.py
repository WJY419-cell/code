"""Small independent invariance tests for pair masking and label-free graphs."""
import os,json,hashlib
from pathlib import Path
import numpy as np
import run_experiment as e
os.environ.pop('MPC_FIT_AUDIT',None)  # Do not overwrite the completed fit audit on exit.
n=5;K=3
for mode in e.MODES:
 a=e.masks(n,K,mode,.2,73);b=e.masks(n,K,mode,.3,73)
 assert np.all(~a|b)
 v=e.masks(n,K,mode,.2,97,a);assert not np.any(a&v)
 i,j=np.triu_indices(n,1);c=(np.tile(i,K),np.tile(j,K),np.repeat(np.arange(K),len(i)),np.ones(K*len(i)),np.ones(K*len(i)))
 tr=e.select(c,~e.membership(c,a,n));te=e.select(c,e.membership(c,a,n))
 fit=e.select(c,~e.membership(c,a|v,n));val=e.select(c,e.membership(c,v,n))
 e.leakage_checks(tr,fit,val,te,a,v,n,mode)
 # Deliberate insertion of a forbidden pair must be rejected.
 poisoned=tuple(np.r_[x,y[:1]] for x,y in zip(tr,te))
 try:e.leakage_checks(poisoned,fit,val,te,a,v,n,mode)
 except AssertionError:pass
 else:raise AssertionError('Contamination was not detected')
records=[e.m.Record(1,0,np.array([0,1,2]),np.array([1.,2.,2.])),e.m.Record(2,1,np.array([0,1,3]),np.array([4.,3.,2.]))]
ds=e.m.Dataset('MovieLens-test',records,n,['a','b','c'],np.eye(K))
mask=e.masks(n,K,'global',.2,73)
g=e.graph(ds,records,mask)
changed=[e.m.Record(r.unit,r.layer,r.items,np.zeros(len(r.items))) for r in records]
assert np.array_equal(g,e.graph(ds,changed,mask))
before=json.loads((e.OUT/'protocol.json').read_text())['protected_hashes_before']
assert all(hashlib.sha256((e.SOURCE/'code/metapath_coupling_experiment.py').read_bytes()).hexdigest()==sha for path,sha in before.items() if path.endswith('metapath_coupling_experiment.py'))
result={'nested_masks':True,'inner_outer_masks_disjoint':True,'contamination_rejected':True,'graph_invariant_to_rating_values_and_ties':True,'original_solver_unchanged':True}
(e.OUT/'protocol_validation.json').write_text(json.dumps(result,indent=2));print(result)
