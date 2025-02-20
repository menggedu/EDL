import numpy as np
import copy
import collections
import random
import itertools
import scipy

def TrainSTRidge(R, Ut, lam=1e-5, d_tol=1, maxit=100, STR_iters = 10, l0_penalty = None, normalize = 2, split = 0, 
                    print_best_tol = False):            
    """
    Sparse regression with STRidge

    Args:
        R (_type_): _description_
        Ut (_type_): _description_
        lam (_type_, optional): _description_. Defaults to 1e-5.
        d_tol (int, optional): _description_. Defaults to 1.
        maxit (int, optional): _description_. Defaults to 100.
        STR_iters (int, optional): _description_. Defaults to 10.
        l0_penalty (_type_, optional): _description_. Defaults to None.
        normalize (int, optional): _description_. Defaults to 2.
        split (int, optional): _description_. Defaults to 0.
        print_best_tol (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # Split data into 80% training and 20% test, then search for the best tolderance.
    if split != 0:
        np.random.seed(0) # for consistancy
        n,_ = R.shape
        train = np.random.choice(n, int(n*split), replace = False)
        test = [i for i in np.arange(n) if i not in train]
        TrainR = R[train,:]
        TestR = R[test,:]
        TrainY = Ut[train,:]
        TestY = Ut[test,:]
    else:
        TrainR = R
        TestR = R
        TrainY = Ut
        TestY = Ut
        
    # Set up the initial tolerance and l0 penalty
    d_tol = float(d_tol)
    tol = d_tol
    
    if l0_penalty == None:    
        l0_penalty = 0.001
        
    D = TrainR.shape[1]        
    w = np.zeros((D,1))
    w_best = np.linalg.lstsq(TrainR, TrainY,rcond=None)[0]
    err_best = np.mean((TestY - TestR.dot(w_best))**2)  + l0_penalty*np.count_nonzero(w_best)
    tol_best = 0
                
    # Now increase tolerance until test performance decreases
    for iter in range(maxit):

        # Get a set of coefficients and error
        w = STRidge(TrainR, TrainY, lam, STR_iters, tol, normalize =normalize )
        
        err = np.mean((TestY - TestR.dot(w))**2)  + l0_penalty*np.count_nonzero(w)
        
        # Has the accuracy improved?
        if err <= err_best:
            err_best = err
            w_best = w
            tol_best = tol
            tol = tol + d_tol
        else:
            tol = max([0,tol - 2*d_tol])
            d_tol =2*d_tol / (maxit - iter)# d_tol/1.618
            tol = tol + d_tol

    if print_best_tol: print ("Optimal tolerance:", tol_best)
    test_err =  np.mean((TestY - TestR.dot(w_best))**2)   
    return w_best, test_err


def STRidge(X0, y, lam, maxit, tol, normalize=0, print_results=False):
    """
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.
    This assumes y is only one column
    """
    n, d = X0.shape
    X = np.zeros((n, d))

    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d, 1))
        for i in range(0, d):
            Mreg[i] = 1.0 / (np.linalg.norm(X0[:, i], normalize))
            X[:, i] = Mreg[i] * X0[:, i]
    else:
        X = X0
    # Get the standard ridge estimate
    
    if lam != 0:
        w = np.linalg.lstsq(X.T.dot(X) + lam * np.eye(d), X.T.dot(y))[0]
    else:
        w = np.linalg.lstsq(X, y)[0]
    num_relevant = d
    biginds = np.where(abs(w) > tol)[0]
    # Threshold and continue
    for j in range(maxit):
        # Figure out which items to cut out
        smallinds = np.where(abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]
        # If nothing changes then stop
        if num_relevant == len(new_biginds):
            break
        else:
            num_relevant = len(new_biginds)
        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0:
                return w
            else:
                break
        biginds = new_biginds
        # Otherwise get a new guess
        w[smallinds] = 0
        if lam != 0:
            w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam * np.eye(len(biginds)), X[:, biginds].T.dot(y))[0]
        else:
            w[biginds] = np.linalg.lstsq(X[:, biginds], y)[0]
    # Now that we have the sparsity pattern, use standard least squares to get w
    if len(biginds)>0: 
        
        w[biginds] = np.linalg.lstsq(X[:, biginds], y)[0]
    
    if normalize != 0:
        
        return np.multiply(Mreg, w)
    else:
        return w
####
def PolyDiff(u, x, deg = 3, diff = 1, width = 5):
    
    """
    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    width = width of window to fit to polynomial

    This throws out the data close to the edges since the polynomial derivative only works
    well when we're looking at the middle of the points fit.
    """

    u = u.flatten()
    x = x.flatten()

    n = len(x)
    du = np.zeros((n - 2*width,diff))

    # Take the derivatives in the center of the domain
    for j in range(width, n-width):

        # Note code originally used an even number of points here.
        # This is an oversight in the original code fixed in 2022.
        points = np.arange(j - width, j + width + 1)

        # Fit to a polynomial
        poly = np.polynomial.chebyshev.Chebyshev.fit(x[points],u[points],deg)

        # Take derivatives
        for d in range(1,diff+1):
            du[j-width, d-1] = poly.deriv(m=d)(x[j])

    return du

def PolyDiffPoint(u, x, deg = 3, diff = 1, index = None):
    
    """
    Same as above but now just looking at a single point

    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    """
    
    n = len(x)
    if index == None: index = (n-1)//2

    # Fit to a polynomial
    poly = np.polynomial.chebyshev.Chebyshev.fit(x,u,deg)
    
    # Take derivatives
    derivatives = []
    for d in range(1,diff+1):
        derivatives.append(poly.deriv(m=d)(x[index]))
        
    return derivatives


def load_ns_data():


    data = scipy.io.loadmat('./evaluation/data_new/Vorticity_ALL.mat')        
    steps = 151
    n = 449
    m = 199
    dt = 0.2
    dx = 0.02
    dy = 0.02
    
    xmin = 100
    xmax = 425
    ymin = 15
    ymax = 185 
    W = data['VORTALL'].reshape(n,m,steps)   # vorticity
    U = data['UALL'].reshape(n,m,steps)      # x-component of velocity
    V = data['VALL'].reshape(n,m,steps)      # y-component of velocity  
    W = W[xmin:xmax,ymin:ymax,:]
    U = U[xmin:xmax,ymin:ymax,:]
    V = V[xmin:xmax,ymin:ymax,:]
    n,m,steps = W.shape

    np.random.seed(0)

    num_xy = 2000
    num_t = 50
    num_points = num_xy * num_t
    boundary = 5
    boundary_x = 10
    points = {}
    count = 0

    for p in range(num_xy):
        x = np.random.choice(np.arange(boundary_x,n-boundary_x),1)[0]
        y = np.random.choice(np.arange(boundary,m-boundary),1)[0]
        for t in range(num_t):
            points[count] = [x,y,2*t+12]
            count = count + 1

    # Take up to second order derivatives.
    w = np.zeros((num_points,1))
    u = np.zeros((num_points,1))
    v = np.zeros((num_points,1))
    wt = np.zeros((num_points,1))
    wx = np.zeros((num_points,1))
    wy = np.zeros((num_points,1))
    wxx = np.zeros((num_points,1))
    wxy = np.zeros((num_points,1))
    wyy = np.zeros((num_points,1))

    N = 2*boundary-1  # odd number of points to use in fitting
    Nx = 2*boundary_x-1  # odd number of points to use in fitting
    deg = 5 # degree of polynomial to use
    # 
    i = 0
    for p in points.keys():
        i+=1
        [x,y,t] = points[p]
        w[p] = W[x,y,t]
        u[p] = U[x,y,t]
        v[p] = V[x,y,t]
        
        wt[p] = PolyDiffPoint(W[x,y,t-(N-1)//2:t+(N+1)//2], np.arange(N)*dt, deg, 1)[0]
        
        x_diff = PolyDiffPoint(W[x-(Nx-1)//2:x+(Nx+1)//2,y,t], np.arange(Nx)*dx, deg, 2)
        y_diff = PolyDiffPoint(W[x,y-(N-1)//2:y+(N+1)//2,t], np.arange(N)*dy, deg, 2)
        wx[p] = x_diff[0]
        wy[p] = y_diff[0]
        
        x_diff_yp = PolyDiffPoint(W[x-(Nx-1)//2:x+(Nx+1)//2,y+1,t], np.arange(Nx)*dx, deg, 2)
        x_diff_ym = PolyDiffPoint(W[x-(Nx-1)//2:x+(Nx+1)//2,y-1,t], np.arange(Nx)*dx, deg, 2)
        
        wxx[p] = x_diff[1]
        wxy[p] = (x_diff_yp[0]-x_diff_ym[0])/(2*dy)
        wyy[p] = y_diff[1]
    return w,u,v,wt, wx,wxx,wy, wyy, points