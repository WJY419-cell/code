from pathlib import Path
import sys,importlib.util,json
import numpy as np
from scipy.optimize import minimize
p=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('mpc',p/'code/metapath_coupling_experiment.py')
m=importlib.util.module_from_spec(spec);sys.modules['mpc']=m;spec.loader.exec_module(m)
rng=np.random.default_rng(73);n=5;K=3
i,j=np.triu_indices(n,1);i=np.tile(i,K);j=np.tile(j,K);k=np.repeat(np.arange(K),10)
c=(i,j,k,rng.integers(0,9,len(i)).astype(float),rng.integers(1,9,len(i)).astype(float))
S=np.array([[0,1,.2],[1,0,.6],[.2,.6,0.]])
actual=m._solve;captured={}
def capture(obj,size,steps,meta):
 captured['obj']=obj
 return actual(obj,size,steps,meta)
m._solve=capture
u,b=m.fit_joint(c,n,K,S,lambda_u=.03,lambda_b=.005,lambda_s=.2)
obj=captured['obj'];x=rng.normal(size=n*K);f,g=obj(x);eps=1e-6
numeric=np.array([(obj(x+np.eye(n*K)[t]*eps)[0]-obj(x-np.eye(n*K)[t]*eps)[0])/(2*eps) for t in range(n*K)])
err=float(np.max(abs(g-numeric)))
# Independent direct constrained minimization, with one redundant constraint removed.
def value(z):
 a=z[:n];B=z[n:].reshape(n,K);d=a[i]-a[j]+B[i,k]-B[j,k]
 return np.sum((c[3]+c[4])*np.logaddexp(0,d)-c[3]*d)/sum(c[3]+c[4])+.03/2*a@a+.005/2*np.sum(B*B)+.2/(2*K)*np.sum((B@m.laplacian(S))*B)
def constraints(z):
 B=z[n:].reshape(n,K)
 return np.r_[z[:n].sum(),B.sum(axis=0),B.sum(axis=1)[:-1]]
ref=minimize(value,np.zeros(n+n*K),method='SLSQP',constraints={'type':'eq','fun':constraints},options={'ftol':1e-13,'maxiter':1000})
fit=np.r_[u,b.ravel()];gap=abs(value(fit)-ref.fun);constraint=float(np.max(abs(constraints(fit))))
assert err<1e-8 and ref.success and gap<1e-10 and constraint<1e-12
result={'finite_difference_max_error':err,'independent_SLSQP_objective_gap':gap,'constraint_max_residual':constraint,'reference_success':bool(ref.success),'fit_max_gradient':m.FIT_AUDIT[-1]['gradient_inf']}
(p/'audit/independent_solver_validation.json').write_text(json.dumps(result,indent=2));print(result)
