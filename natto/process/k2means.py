
from natto.input import hungarian as h 
from sklearn.mixture import _gaussian_mixture as _gm
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as ed
from sklearn.mixture import GaussianMixture as gmm 


#############
# KMEANS 
############
def assign(x1,x2,c1,c2):
    r = ed(x1,c1)
    r2 = ed(x2,c2)
    r3=np.concatenate((r,r2), axis=1)
    z = np.argmin(r3, axis = 1)
    res = np.array( [zz if zz < c1.shape[0] else zz-c1.shape[0]  for zz in z] )

    # collecting somestats
    z2 = np.argmin(r, axis = 1) -  np.argmin(r2, axis = 1)
    print("unmatching:", sum(z2==0))

    return res
    


def centers(y,X):
    cents = []
    for i in np.unique(y):
        cents.append(X[y==i].mean(axis=0))
    return np.array(cents)


def opti_kmeans(X1,X2,y): 
    c1, c2 = centers(y,X1), centers(y,X2)
    c2 = hung.hungsort(c1,c2)
    y = assign(X1,X2,c1,c2)
    return y










###################
#  OPTIMIZE GMM 
##################

def hot1(y): 
    # be sure to start with label 0 ... 
    uy =  y.max()+1
    r= np.zeros((len(y), uy)) 
    r[range(len(y)), y] =1 
    return r 




def get_means_resp(X,log_resp, cov):

    _, means_, covariances_ = _gm._estimate_gaussian_parameters(X, np.exp(log_resp), 1e-6, cov)
    precisions_cholesky_    = _gm._compute_precision_cholesky( covariances_, cov)
    log_resp                = _gm._estimate_log_gaussian_prob( X, means_, precisions_cholesky_,cov)

    return means_, log_resp


def optimize(X1,X2,y, cov='tied'): 

    log_resp = hot1(y)  # init 

    m1, l1 = get_means_resp(X1,log_resp,cov)
    m2, l2 = get_means_resp(X2,log_resp,cov)

    # now we should check if the labels are ok ... 
    # i can just run the hung and see if its just 1 2 3 4.. 
    # should rarely fail.. 

    (a,b),_ = h.hungarian(m1,m2) 
    print(b)
    
    log_resp = l1+l2
    return log_resp.argmax(axis=1)







##################3
#  MAIN FUNCTION IS HERE;; SIMUCLUST 
######################


def init(X,clusts=10):
    return gmm(n_components=clusts, n_init=30).fit_predict(X) 


def optistep(X1,X2,y,method):
    if method == 'kmeans':
        return optimize_kmeans(X1,X2,y)
    else:
        return optimize(X1,X2,y, cov=method)

def simulclust(X1,X2,y, method = 'tied', n_iter=100, debug = False):
    for asd in range(n_iter):
        yold = y.copy()
        y =  optistep(X1,X2,y,method) 
        chang = sum(y!=yold)
        if debug: 
            print(f"simuclust: score: {score}  changes in iter:  {chang}")
        if chang == 0:
            break

    return y



    




'''
so we want to use as much as possible from sklearn.. so we read the code..: 

we assume init is given.. 

reg_covar:  float, defaults to 1e-6.

base mixture: 

    fit_predict: 
            # EXPECT
            log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
            #log_prob_norm =  np.mean(log_prob_norm)
            
            # MAXIMIZE 
            self._m_step(X, log_resp)

            return log_resp.argmax(axis=1)

    gauss mix: 

        def _estimate_log_prob(self, X):
                return _estimate_log_gaussian_prob(
                            X, self.means_, self.precisions_cholesky_, self.covariance_type)

        def _m_step(self, X, log_resp):
            """M step.
            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
            log_resp : array-like, shape (n_samples, n_components)
                Logarithm of the posterior probabilities (or responsibilities) of
                the point of each sample in X.
            """
            n_samples, _ = X.shape
            self.weights_, self.means_, self.covariances_ = (
                _estimate_gaussian_parameters(X, np.exp(log_resp), self.reg_covar,
                                              self.covariance_type))
            self.weights_ /= n_samples
            self.precisions_cholesky_ = _compute_precision_cholesky(
                self.covariances_, self.covariance_type)
'''



