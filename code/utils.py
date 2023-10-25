import numpy as np
import jax.numpy as jnp
import pickle

def save_obj(obj, name, data_dir):
    with open(data_dir+name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name, data_dir):
    with open(data_dir+name+'.pkl', 'rb') as f:
        return pickle.load(f)

def A_statsig(A, C, alpha, numpy=False):
    lb=alpha
    ub=1-alpha

    vlb, v50, vub = jnp.percentile(C,jnp.array([lb,50,ub]), axis=0)
    sign_lb = jnp.sign(A*vlb)
    sign_ub = jnp.sign(A*vub)
    median_ = A*v50
    A_ss = jnp.where((sign_lb*sign_ub)>0,1.,0.)

    if numpy:
        return np.array(A_ss), np.array(median_)
        
    return A_ss, median_