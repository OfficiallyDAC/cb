import jax
import jax.numpy as jnp
from jax.numpy.linalg import pinv,qr
from jaxopt import ScipyBoundedMinimize
from jaxopt.objective import least_squares
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_box

#analytical solution
#(X.T @ X)^{-1} @ X.T @ y 
def OLS_estimator(y, X, method='pinv'):
    """
    y: jnp.array(float), shape(T:length of ts,), endogenous variable 
    X: jnp.array(float), shape(K': # of regressors,T: length of ts), exogenous variables
    method: str, 'pinv' for pseudoinverse solution, 'qr' for QR factorization
    """

    if method=='pinv':
        beta = pinv(X.T)@y
    elif method=='qr':
        Q, R = qr(X.T)
        effects = Q.T@y
        beta = jnp.linalg.solve(R, effects)
    
    return beta

def OLS_BoxConstrained(y, X, features, jit=False):
    """
    y: jnp.array(float), shape(T:length of ts,), endogenous variable 
    X: jnp.array(float), shape(K: covariates,T: length of ts), exogenous variables
    features_idx: jnp.array(int/float), shape(K,), features to use are associated with nonzero entries   
    """

    K=X.shape[0]
    betas_init=jnp.zeros(K)
    lbfgsb = ScipyBoundedMinimize(fun=least_squares, method="l-bfgs-b", jit=jit)
    upper_bounds = features*jnp.ones_like(betas_init) * jnp.nan_to_num(jnp.inf)
    lower_bounds = -1.*upper_bounds
    bounds = (lower_bounds, upper_bounds)
    #Our data is transposed w.r.t jaxopt
    betas = lbfgsb.run(betas_init, bounds=bounds, data=(X.T, y)).params

    return betas

def OLS_BOXProjection(y,X,features):
    K=X.shape[0]
    betas_init=jnp.zeros(K)
    upper_bounds = features*jnp.ones_like(betas_init) * jnp.nan_to_num(jnp.inf) +1.e-10
    lower_bounds = -1.*upper_bounds - 1.e-10
    bounds = (lower_bounds, upper_bounds)
    pg = ProjectedGradient(fun=least_squares, projection=projection_box)
    betas = pg.run(betas_init, hyperparams_proj=bounds, data=(X.T, y)).params

    return betas

def loglike(y, X, betas, scale=None):
    """
    The likelihood function for the OLS model.

    Parameters
    ----------
    params : array_like
        The coefficients with which to estimate the log-likelihood.
    scale : float or None
        If None, return the profile (concentrated) log likelihood
        (profiled over the scale parameter), else return the
        log-likelihood using the given scale value.

    Returns
    -------
    float
        The likelihood function evaluated at params.
    """
    nobs = jnp.float32(y.shape[-1])
    nobs2 = nobs / 2.0
    resid = y - X.T@betas
    ssr = jnp.sum(resid**2)
    if scale is None:
        # profile log likelihood
        llf = -nobs2*jnp.log(2*jnp.pi) - nobs2*jnp.log(ssr / nobs) - nobs2
    else:
        # log-likelihood
        llf = -nobs2 * jnp.log(2 * jnp.pi * scale) - ssr / (2*scale)
    return llf

#this function must vmapped along subjects
#data are the set of time series for subject s at scale j, dim=(N,T)

def delta_score(candidate_parent, child, child_column, data, ll0, method='pinv', parallelized=False):
    """
    Function to assign the score to a certain edge.
    
    INPUT
    =====
    candidate_parent: int, candidate parent index (first pos in the edge);
    child: int, child node index (second pos in the edge);
    child_column: array, column indexed at child in  A'=P_{s'kp}+A, where A(KxK) is the adjacency of the solution and 
                P_{s'kp}(KxK) is the idiosyncratic matrix for subject s';
    X: jnp.array (float), shape (K: # of ts; T: length of ts);
    ll0: float, likelihood attribute of child node in the solution

    OUTPUT
    ======
    delta: float, edge score.
    """
    if ll0 is None: ll0=0.
    
    K,T=data.shape
    y=data[child]
    
    if not parallelized:
        #here X has shape (K,T), whereas betas (K,)
        nnz = jnp.flatnonzero(child_column.at[candidate_parent].set(1))
        X=data[nnz]
        # print(y.shape,X.shape)
        betas = OLS_estimator(y,X,method=method)
        betas_filled=jnp.zeros(K).at[nnz].set(betas)
        llj=loglike(y,X,betas)
    elif parallelized:
        child_column=child_column.at[candidate_parent].set(1)
        betas_filled = child_column*OLS_estimator(y,child_column.reshape(-1,1)*data,method=method)
        llj=loglike(y,data,betas_filled)
        
    #delta in BIC
    delta = 2*llj-2*ll0-2*jnp.log(T)
    # print("delta:{}, llj:{}, ll0:{}".format(delta, llj, ll0))
    return delta, llj, betas_filled

def vdelta_score(candidate_parent, child, child_columns, data, ll0, method, parallelized):
    return jax.vmap(delta_score, (None,None,0,0,0, None, None), (0,0,0))(candidate_parent, child, child_columns, data, ll0, method, parallelized)

# @jax.jit
def vloglike(X_sit, X_skt, A_ski):
    return jax.vmap(loglike, (0,0,0), (0))(X_sit, X_skt, A_ski)