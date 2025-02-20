import numpy as np
import scipy.io as scio
import scipy
import pickle
import math
from functools import partial
from sklearn.metrics import r2_score
import sympy as sp
from scipy.optimize import minimize

from evaluation.sr_utils import *
from evaluation.PDE_find import load_ns_data
from evaluation.strogatz_equations import equations
from evaluation.solve_and_plot import config, process_equations, plot_trajectories,solve_equations, plot_prediction

def data_load(dataset):

    feature_dict = {}
    if dataset == 'chafee-infante': # 301*200

        u = np.load("./evaluation/data_new/chafee_infante_CI.npy")
        x = np.load("./evaluation/data_new/chafee_infante_x.npy").reshape(-1,1)
        t = np.load("./evaluation/data_new/chafee_infante_t.npy").reshape(-1,1)
        n_input_var = 1    
        n, m = u.shape 

    elif dataset == 'Burgers':

        data = scio.loadmat('./evaluation/data_new/burgers.mat')
        u=data.get("usol")
        x=np.squeeze(data.get("x")).reshape(-1,1)
        t=np.squeeze(data.get("t").reshape(-1,1))
        sym_true = 'add,mul,u1,diff,u1,x1,diff2,u1,x1'
        right_side_origin = 'right_side_origin = -1*u_origin*ux_origin+0.1*uxx_origin'
        n_input_var = 1
    
    elif dataset == 'PDE_divide':
        u=np.load("./evaluation/data_new/PDE_divide.npy").T[3:-3,3:-3]
        nx = 100
        nt = 251
        x=np.linspace(1,2,nx).reshape(-1,1)[3:-3]
        t=np.linspace(0,1,nt).reshape(-1,1)[3:-3]

        sym_true = 'add,div,diff,u1,x1,x1,diff2,u1,x1'
        right_side_origin = 'right_side_origin = -config.divide(ux_origin, x_all) + 0.25*uxx_origin'
        n_input_var = 1

    elif dataset == 'KS':
        data = scipy.io.loadmat('./evaluation/data/kuramoto_sivishinky.mat') # course temporal grid 
        t = np.real(data['t'].flatten()[:,None])
        x = np.real(data['x'].flatten()[:,None])
        u = np.real(data['u'])
    
        sym_true = 'add,mul,u1,diff,u1,x1,add,diff2,u1,x1,diff4,u1,x1'
        n_input_var = 1
        
    elif dataset == 'KS2':
        data = scipy.io.loadmat('./evaluation/data/KS.mat') # course temporal grid 
        # import pdb;pdb.set_trace()
        
        t = np.real(data['t'].flatten()[:,None])
        x = np.real(data['x'].flatten()[:,None])
        u = np.real(data['usol'])
           
        sym_true = 'add,mul,u1,diff,u1,x1,add,diff2,u1,x1,diff2,diff2,u1,x1,x1'
        n_input_var = 1

    elif dataset == "fisher":
    
        data=scipy.io.loadmat('./evaluation/data_new/fisher_nonlin_groundtruth.mat')

        # D=data['D'] #0.02
        # r=data['r'] #10
        # K=data['K']
        x=np.squeeze(data['x'])[1:-1].reshape(-1,1)[3:-3]
        t=np.squeeze(data['t'])[1:-1].reshape(-1,1)[3:-3]
        u=data['U'][3:-3,3:-3][1:-1,1:-1].T
        sym_true = "add,mul,u1,diff2,u1,x1,add,n2,diff,u1,x1,add,u1,n2,u1"
        n_input_var = 1
    elif dataset == 'NS':
        w,u,v,w_t, w_x, w_xx, w_y, w_yy, xyt = load_ns_data()
        # feature_dict['x'] = x_data.reshape(-1)
        feature_dict['w_x'] = w_x.reshape(-1)
        feature_dict['w_xx'] = w_xx.reshape(-1)
        feature_dict['w_y'] = w_y.reshape(-1)
        feature_dict['w_yy'] = w_yy.reshape(-1)
        feature_dict['w'] = w.reshape(-1)
        feature_dict['u'] = u.reshape(-1)
        feature_dict['v'] = v.reshape(-1)
        return w_t.reshape(-1,1),  feature_dict

    else:
        assert False, "Unknown dataset"
    
    n, m = u.shape
    ut = np.zeros((n, m))
    dt = t[1]-t[0]

    for idx in range(n):
        ut[idx, :] = FiniteDiff(u[idx, :], dt)
    # ut = ut[math.floor(n*0.03):math.ceil(n*0.97), math.floor(m*0.03):math.ceil(m*0.97)]
    # x fist
        
    u_x = Diff(u, x, 0)
    u_xx = Diff2(u,x,0)
    u_xxx = Diff3(u, x, 0)
    feature_dict['x'] = np.repeat(x.reshape(-1,1), m, axis =1).reshape(-1)
    feature_dict['u_x'] = u_x.reshape(-1)
    feature_dict['u_xx'] = u_xx.reshape(-1)
    feature_dict['u_xxx'] = u_xxx.reshape(-1)
    feature_dict['u'] = u.reshape(-1)
    return ut.reshape(-1,1),  feature_dict



class ODEData:

    def __init__(self):
        self.index_data = [1,2,3,5,6,7,8,13,16,17,21,23]

    def load_data(self,data_name, noise = 0):
        index = int(data_name.split('_')[1])
        # assert index in self.index_data, 'not in ode dataset'
        indices = [index-1]
        feature_dict = {}
        from odeformer.odebench.strogatz_equations import equations 
        equations = [equations[i] for i in indices]
        process_equations(equations)
        solve_equations(equations, config)

        eq_dict = equations[0]
        eq_string = eq_dict['eq']

        individual_eqs = eq_string.split('|')
        eqs_sympy = [sp.sympify(eq) for eq in individual_eqs][0]

        # import pdb;pdb.set_trace()
        # substitute const
        consts_values = eq_dict['consts']
        const_symbols = sp.symbols([f'c_{i}' for i in range(len(consts_values[0]))])
        var_symbols = sp.symbols([f'x_{i}' for i in range(eq_dict['dim'])])

        const_subs = dict(zip(const_symbols, consts_values[0]))
        eq_subs = eqs_sympy.subs(const_subs)
        print(eq_subs)
        f = sp.lambdify(var_symbols, eq_subs, 'numpy')

        print("true expression", eqs_sympy)
        print("parameters", consts_values)
        # import pdb;pdb.set_trace()
        t = np.array(eq_dict['solutions'][0][0]['t'])[1:-1]
        y = np.array(eq_dict['solutions'][0][0]['y']).reshape(-1)
        
        #noise
        y *= (1+noise*np.random.randn(*y.shape))

        t_test = np.array(eq_dict['solutions'][0][1]['t'])[1:-1]
        y_test = np.array(eq_dict['solutions'][0][1]['y']).reshape(-1) 
        # lhs
        y_rhs = y[1:-1]
        y_rhs_test = y_test[1:-1]
        lhs = (y[2:]-y[:-2])/(t[1]-t[0])/2
        lhs_test = (y_test[2:]-y_test[:-2])/(t_test[1]-t_test[0])/2
        # rhs real
        y_rhs_train = [y_rhs]
        rhs_train = f(*y_rhs_train)
        # import pdb;pdb.set_trace()
        y_rhs_test = [y_rhs_test]
        rhs_test = f(*y_rhs_test)  

        feature_dict['x'] = y_rhs

        return lhs.reshape(-1,1) , feature_dict , rhs_train, rhs_test,y_rhs_test,lhs_test.reshape(-1,1)



# def process_sym( consts):
    
#     const_subs = dict(zip(const_symbols, consts))
#     eq_subs = eqs_sympy.subs(const_subs)
#     print(eq_subs)
#     # import pdb;pdb.set_trace()
#     f = sp.lambdify(var_symbols, eq_subs, 'numpy')
#     y_input = [y_rhs]
#     rhs =f(*y_input)
    
#     # print('rhs',rhs)
#     loss = np.mean(np.square(rhs-lhs))
#     print(loss)
#     return loss

#     # return 






