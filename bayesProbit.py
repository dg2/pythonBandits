'''
Bayesian learning for Probit models

Dario Garcia
'''

import numpy as np
import scipy.stats as stats
try:
    import statsmodels.api as sm
except:
    import scikits.statsmodels.api as m

def asGLM(X,y):
    '''
    Frequentist Probit model

    Inputs:
    - X: Feature matrix (DxN)
    - y: Observations (binary vector of length N)
    '''
    clf = sm.Probit(y,X.T)
    clf_ti = clf.fit()
    print 'Coefficients: '
    print clf_ti.params
    print 'CI: '
    print clf_ti.conf_int()
    return clf_ti

# TODO: 
# - Use Cholesky decomposition of the covariance matix to directly generate multivariate
# normal samples
# - Rewrite in Cython
def gibbsSampling(X,y,mu_0=None,sigma_0=None,NUM_SAMPLES=300,initialization='zeros'):
    '''
    Albert and Chib's algorithm for sampling from the posterior of a probit
    regression model with a MV normal prior
   
    Inputs:
    - X: Feature matrix (DxN)
    - y: Observations (binary vector)
    - mu_0: Prior mean
    - sigma_0: Prior covariance matrix (DxD)
    - NUM_SAMPLES [500]
    - initialization ['zeros']: 'zeros', 'normal'
    '''
    # Initialization
    D,N = X.shape
    if (mu_0 is None):
        mu_0 = np.zeros(D)
    if (sigma_0 is None):
        sigma_0 = np.eye(D)

    # Initialize the coefficient vector
    # Since the prior may be quite wide, instead of sampling from it
    # we sample from a narrower distribution (standard MV gaussian)
    # or we may even fix it to 0
    if (initialization=='zeros'):
        coef = np.zeros(D)
    elif (initialization=='normal'):
        coef = np.random.randn(D)

    # Covariance matrix for the samples
    i_sigma_0 = np.linalg.inv(sigma_0)
    sample_cov = np.dot(X,X.T)
    i_sigma = i_sigma_0 + sample_cov
    sigma = np.linalg.inv(i_sigma)

    out = np.zeros((D,NUM_SAMPLES))
    
    # Main loop
    for it in range(NUM_SAMPLES):
        # TODO: Progress report

        # Data augmentation step
        # For each input point, we obtain a extra variable z
        # sampled from a truncated gaussian with the same sign
        # as the corresponding observation
        z = np.zeros(N)        
        # TODO: Rewrite without the loops
        for i in range(N):            
            theta = np.dot(coef,X[:,i])
            if (y[i]==1):
                z[i] = stats.truncnorm.rvs(-theta,1e3)+theta
            else:
                z[i] = stats.truncnorm.rvs(-1e3,-theta)+theta

        # Sampling
        coef = sigma.dot(X.dot(z)+i_sigma_0.dot(mu_0))
        out[:,it] = np.random.multivariate_normal(coef, sigma)

    return out
