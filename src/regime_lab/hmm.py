"""Diagonal Gaussian HMM fitted by numerically stable Baum-Welch EM."""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp

@dataclass
class GaussianHMM:
    n_states: int = 2
    n_init: int = 4
    max_iter: int = 150
    tol: float = 1e-5
    min_var: float = 1e-4
    random_state: int = 42

    def _check(self, X):
        X=np.asarray(X,dtype=float)
        if X.ndim!=2 or len(X)<self.n_states*5 or not np.isfinite(X).all():
            raise ValueError("X must be finite 2D with at least 5*n_states rows")
        return X

    def _emissions(self,X):
        d=X[:,None,:]-self.means_[None,:,:]
        return -0.5*(np.log(2*np.pi*self.vars_).sum(axis=1)[None,:]+(d*d/self.vars_[None,:,:]).sum(axis=2))

    def _forward(self,logb):
        T,K=logb.shape
        a=np.empty((T,K))
        a[0]=np.log(self.start_)+logb[0]
        logtrans=np.log(self.trans_)
        for t in range(1,T):
            a[t]=logb[t]+logsumexp(a[t-1,:,None]+logtrans,axis=0)
        return a,float(logsumexp(a[-1]))

    def _backward(self,logb):
        T,K=logb.shape
        b=np.zeros((T,K))
        logtrans=np.log(self.trans_)
        for t in range(T-2,-1,-1):
            b[t]=logsumexp(logtrans+logb[t+1][None,:]+b[t+1][None,:],axis=1)
        return b

    def fit(self,X):
        X=self._check(X)
        T,D=X.shape
        rng=np.random.default_rng(self.random_state)
        best=-np.inf
        for _ in range(self.n_init):
            centers=X[rng.choice(T,self.n_states,replace=False)].copy()
            self.means_=centers
            self.vars_=np.maximum(np.var(X,axis=0)[None,:]*np.ones((self.n_states,1)),self.min_var)
            self.start_=np.full(self.n_states,1/self.n_states)
            self.trans_=np.eye(self.n_states)*.85+np.full((self.n_states,self.n_states),.15/self.n_states)
            history=[]
            for i in range(self.max_iter):
                logb=self._emissions(X)
                a,ll=self._forward(logb)
                b=self._backward(logb)
                gamma=np.exp(a+b-ll)
                logxi=(a[:-1,:,None]+np.log(self.trans_)[None,:,:]+logb[1:,None,:]+b[1:,None,:]-ll)
                xi=np.exp(logxi).sum(axis=0)
                w=gamma.sum(axis=0)+1e-12
                self.start_=np.maximum(gamma[0],1e-8)
                self.start_/=self.start_.sum()
                self.trans_=(xi+1e-3)/(xi.sum(axis=1,keepdims=True)+self.n_states*1e-3)
                self.means_=gamma.T@X/w[:,None]
                self.vars_=np.maximum((gamma.T@(X*X))/w[:,None]-self.means_**2,self.min_var)
                history.append(ll)
                if len(history)>1 and abs(history[-1]-history[-2])<self.tol*(1+abs(ll)):
                    break
            _,final_ll=self._forward(self._emissions(X))
            if final_ll>best:
                best=final_ll
                params=tuple(z.copy() for z in (self.start_,self.trans_,self.means_,self.vars_))
                self.history_=history
        self.start_,self.trans_,self.means_,self.vars_=params
        self.log_likelihood_=best
        self.n_features_in_=D
        return self

    def filter(self,X,initial=None):
        """Causal P(S_t|X_0,...,X_t), never future smoothed states."""
        X=np.asarray(X,float)
        if X.ndim!=2 or X.shape[1]!=self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("Invalid feature matrix")
        if len(X)==0:
            return np.empty((0,self.n_states))
        logb=self._emissions(X)
        probs=np.empty((len(X),self.n_states))
        prior=self.start_ if initial is None else np.asarray(initial)@self.trans_
        if initial is not None and (np.asarray(initial).shape!=(self.n_states,) or not np.isclose(np.sum(initial),1)):
            raise ValueError("Invalid initial distribution")
        for t in range(len(X)):
            lp=np.log(np.maximum(prior,1e-300))+logb[t]
            probs[t]=np.exp(lp-logsumexp(lp))
            prior=probs[t]@self.trans_
        return probs

    def score(self,X):
        return self._forward(self._emissions(np.asarray(X,float)))[1]