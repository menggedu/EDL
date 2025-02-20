import numpy as np
import random
from functools import partial
from scipy.optimize import minimize

from evaluation.PDE_find import TrainSTRidge

class ScipyMinimize:
    """SciPy's non-linear optimizer"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    
    def __call__(self, f, x0):
        with np.errstate(divide='ignore'):
            opt_result = partial(minimize, method='BFGS')(f, x0)
        x = opt_result['x']
        return x

def merge_dict(dict1,dict2):
    new_dict = {}
    for key, value in dict1.items():
        if key in dict2:
            new_dict[key] = dict1[key]+dict2[key]
        else:
            new_dict[key] = value
    for key, value in dict2.items():
        if key not in new_dict:
            new_dict[key] = value
    
    return new_dict


def make_metric(name, *args):
    metrics = {

            "sparse_reward":  (lambda y, y_hat,n : (1-args[0]*n)/(1 + np.sqrt(np.mean((y - y_hat)**2)/np.var(y))),
                            1),
            "inv_nrmse" :    (lambda y, y_hat,n : 1/(1 + np.sqrt(np.mean((y - y_hat)**2)/np.var(y))),
                            1),
            "R2": (lambda y, y_hat,n: 1 - ((y_hat - y)**2).sum() / ((y - y.mean())**2).sum(),
                   1 )

    }
    assert name in metrics, "Unrecognized reward function name."

    return metrics[name][0]

def linear_calculate(func_terms,func_strs, lhs,add_const,mode):
            # const added
    # import pdb;pdb.set_trace() 
    if add_const:
        try:
            length = func_terms[0].shape[0]
        except:
            return [],0,0,False,'false shape'
        const = np.ones(length)
        func_terms.append(const)
        func_strs.append('c')

    if mode == 'sparse_regression':
        y_rhs, coef, valid , error_type = sparse_calculate(func_terms, lhs)
    elif mode =='regression':
        y_rhs, coef, valid , error_type= coef_calculate(func_terms, lhs)
    else:
        print(mode)
        assert False
    return func_strs, y_rhs, coef, valid , error_type

def sparse_calculate(func_terms, lhs, l0_penalty=10**-5):
    y_rhs, coef, valid , error_type = coef_calculate(func_terms, lhs)

    if error_type == 'lstsq_error':
        return y_rhs, coef, valid , error_type

    values = np.array(func_terms).T
    valid = True
    error_type = None
    w,mse= TrainSTRidge(values, lhs, l0_penalty=l0_penalty)
    if not valid_coef(w):
        valid = False
        error_type = 'abnormal coef'
    
    y_hat = values.dot(w)
    w = w.reshape(-1).tolist()
    return y_hat, w, valid, error_type

def valid_coef(coefs):
    
    for i in range(len(coefs)):        
        if coefs[i]==0:
            continue    
        if np.abs(coefs[i])<1e-4 or np.abs(coefs[i])>1e4:
            # print("abnormal coef" )
            return False
    return True
       
def coef_calculate(func_terms, lhs):
    # add_const

    try:
        rhs = np.array(func_terms).T
        w_best = np.linalg.lstsq(rhs, lhs)[0]
    except Exception as e:
        # print(e)
        # import pdb;pdb.set_trace()
        return 0, 0, False, "lstsq_error"
    
    y_hat = rhs.dot(w_best)
    
    if not valid_coef(w_best):
        return 0, 0, False, "coef_error"  
    
    y_hat = rhs.dot(w_best)
    w_best = w_best.reshape(-1).tolist()
    return y_hat, w_best, True, None

def remove_redundants(terms, tokens):
    unique_tokens = []
    unique_values = []
    duplicate = False
    for i, (arr_current, str_current) in enumerate(zip(terms, tokens)):
        duplicate_found = False
        for j, (arr_compare, str_compare) in enumerate(zip(terms[i+1:], tokens[i+1:])):
            # if the sum of differences is less than 1e-5, consider them the same
            if np.abs(np.sum(np.abs(arr_compare) - np.abs(arr_current))) < 1e-5:
                duplicate_found = True
                duplicate=True
                # if the current string is shorter, replace the compared one
                if len(str_current) < len(str_compare):
                    safe_idx = i+j+1
                    idx = tokens.index(str_compare)
                    assert safe_idx == idx
                    tokens[idx] = str_current
                    tokens[idx]=tokens[i]
                    terms[idx]=terms[i]

        if not duplicate_found:
            unique_values.append(arr_current)
            unique_tokens.append(tokens[i])

    return unique_values, unique_tokens, duplicate

def reorganize(func_strs):
    link_symbol = [' + ']
    new_string = ''
    terms_str = []
    id_convert = []
    for i, term in enumerate(func_strs):
        if term == 'c':
            # constant is not included
            continue
        if term.startswith('-'):
            term = term[1:]
            id_convert.append(i)
        term_converted = term.replace('**', '^')
        terms_str.append(term_converted)
        new_string+= term_converted +random.choice(link_symbol)
    return new_string[:-3], id_convert

def FiniteDiff(u, dx):
    
    n = u.size
    ux = np.zeros(n)

    # for i in range(1, n - 1):
    ux[1:n-1] = (u[2:n] - u[0:n-2]) / (2 * dx)

    ux[0] = (-3.0 / 2 * u[0] + 2 * u[1] - u[2] / 2) / dx
    ux[n - 1] = (3.0 / 2 * u[n - 1] - 2 * u[n - 2] + u[n - 3] / 2) / dx
    return ux


def FiniteDiff2(u, dx):

    n = u.size
    ux = np.zeros(n)

    ux[1:n-1] = (u[2:n] - 2 * u[1:n-1] + u[0:n-2]) / dx ** 2

    ux[0] = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / dx ** 2
    ux[n - 1] = (2 * u[n - 1] - 5 * u[n - 2] + 4 * u[n - 3] - u[n - 4]) / dx ** 2
    return ux



# @jit(nopython=True)  
def Diff(u, dxt, dim, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    n, m = u.shape
    uxt = np.zeros((n, m))
    
    if len(dxt.shape) == 2:
        dxt = dxt[:,0]
    if name == 'x':
        dxt = dxt[2]-dxt[1]
        # for i in range(m):
        #     uxt[:, i] = FiniteDiff(u[:, i], dxt)
        uxt[1:n-1,:] = (u[2:n,:] - u[0:n-2,:]) / (2 * dxt)

        uxt[0,:] = (-3.0 / 2 * u[0,:] + 2 * u[1,:] - u[2,:] / 2) / dxt
        uxt[n - 1,:] = (3.0 / 2 * u[n - 1,:] - 2 * u[n - 2,:] + u[n - 3,:] / 2) / dxt
    # elif name == 't':
    #     for i in range(n):
    #         uxt[i, :] = FiniteDiff(u[i, :], dxt)

    else:
        assert False
        NotImplementedError()

    return uxt

# @jit(nopython=True)  
def Diff2(u, dxt, dim, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """
    if len(dxt.shape) == 2:
        dxt = dxt[:,0]
    n, m = u.shape
    uxt = np.zeros((n, m))
    dxt = dxt[2]-dxt[1]
    if name == 'x':

        uxt[1:n-1,:] =(u[2:n,:] - 2 * u[1:n-1,:] + u[0:n-2,:]) / dxt ** 2

        uxt[0,:] = (2 * u[0,:] - 5 * u[1,:] + 4 * u[2,:] - u[3,:]) / dxt ** 2
        uxt[n - 1,:] = (2 * u[n - 1,:] - 5 * u[n - 2,:] + 4 * u[n - 3,:] - u[n - 4,:]) / dxt ** 2
    else:
        assert False
        NotImplementedError()

    return uxt

# @jit(nopython=True)  
def Diff3(u, dxt, dim, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    n, m = u.shape
    uxt = np.zeros((n, m))

    if name == 'x':
        uxt=Diff2(u,dxt,dim,name)
        uxt = Diff(uxt,dxt, dim, name)
        # dxt = dxt[2]-dxt[1]

        # for i in range(m):
        #     uxt[:, i] = FiniteDiff2(u[:, i], dxt)
        #     uxt[:,i] = FiniteDiff(uxt[:,i],dxt )

    else:
        assert False
        NotImplementedError()

    return uxt

# @jit(nopython=True)  
def Diff4(u, dxt, dim, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    n, m = u.shape
    uxt = np.zeros((n, m))

    
    if name == 'x':
        uxt=Diff2(u,dxt,dim,name)
        uxt = Diff2(uxt,dxt, dim, name)
    else:
        assert False
        NotImplementedError()
 

    return uxt

def Diff_2(u, dxt, name=1):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    # import pdb;pdb.set_trace()
    if u.shape == dxt.shape:
        return u/dxt
    t,n,m = u.shape
    uxt = np.zeros((t, n, m))
    if len(dxt.shape) == 2:
        dxt = dxt[:,0]
    dxt = dxt.ravel()
    # import pdb;pdb.set_trace()
    if name == 1:
        dxt = dxt[2]-dxt[1]
        uxt[:,1:n-1,:] = (u[:,2:n,:]-u[:,:n-2,:])/2/dxt
        
        uxt[:,0,:] = (u[:,1,:]-u[:,-1,:])/2/dxt
        uxt[:,-1,:] = (u[:,0,:]-u[:,-2,:])/2/dxt
    elif name == 2:
        dxt = dxt[2]-dxt[1]
        uxt[:,:,1:m-1] = (u[:,:,2:m]-u[:,:,:m-2])/2/dxt
        
        uxt[:,:,0] = (u[:,:,1]-u[:,:,-1])/2/dxt
        uxt[:,:,-1] = (u[:,:,0]-u[:,:,-2])/2/dxt
        # uxt[:,:,0] = (-3.0 / 2 * u[:,:,0] + 2 * u[:,:,1] - u[:,:,2] / 2) / dxt
        # uxt[:,:,n - 1] = (3.0 / 2 * u[:,:,n - 1] - 2 * u[:,:,n - 2] + u[:,:,n - 3] / 2) / dxt
    else:
        assert False, 'not supported'     

    return uxt

# @jit(nopython=True)
def Diff2_2(u, dxt, name=1): 
    """
    Here dx is a scalar, name is a str indicating what it is
    """
  
    
    if u.shape == dxt.shape:
        return u/dxt
    t,n,m = u.shape
    uxt = np.zeros((t, n, m))
    dxt = dxt.ravel()
    # try: 
    if name == 1:
        dxt = dxt[2]-dxt[1]
        uxt[:,1:n-1,:]= (u[:,2:n,:] - 2 * u[:,1:n-1,:] + u[:,0:n-2,:]) / dxt ** 2
        uxt[:,0,:] = (u[:,1,:]+u[:,-1,:]-2*u[:,0,:])/dxt ** 2
        uxt[:,-1,:] = (u[:,0,:]+u[:,-2,:]-2*u[:,-1,:])/dxt ** 2
        # uxt[:,0,:] = (2 * u[:,0,:] - 5 * u[:,1,:] + 4 * u[:,2,:] - u[:,3,:]) / dxt ** 2
        # uxt[:,n - 1,:] = (2 * u[:,n - 1,:] - 5 * u[:,n - 2,:] + 4 * u[:,n - 3,:] - u[:,n - 4,:]) / dxt ** 2
    elif name == 2:
        dxt = dxt[2]-dxt[1]
        uxt[:,:,1:m-1]= (u[:,:,2:m] - 2 * u[:,:,1:m-1] + u[:,:,0:m-2]) / dxt ** 2
        uxt[:,:,0] = (u[:,:,1]+u[:,:,-1]-2*u[:,:,0])/dxt ** 2
        uxt[:,:,-1] = (u[:,:,0]+u[:,:,-2]-2*u[:,:,-1])/dxt ** 2  
        # uxt[:,:,0] = (2 * u[:,:,0] - 5 * u[:,:,1] + 4 * u[:,:,2] - u[:,:,3]) / dxt ** 2
        # uxt[:,:,n - 1] = (2 * u[:,:,n - 1] - 5 * u[:,:,n - 2] + 4 * u[:,:,n - 3] - u[:,:,n - 4]) / dxt ** 2
        
    else:
        NotImplementedError()
# except:
    #     import pdb;pdb.set_trace()

    return uxt
