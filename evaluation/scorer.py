
import numpy as np
import re
from sklearn.metrics import r2_score
import sympy as sp
import random
import warnings

from evaluation.sympy_utils import str2sympy, create_sympy_symbols, walking_tree, check_error, check_symbols_valid, count_functerms
from evaluation.sr_utils import  make_metric,reorganize, remove_redundants, linear_calculate, ScipyMinimize, merge_dict
from evaluation.load_data import data_load, ODEData
from evaluation.expression import Equation, PriorityQueue

def extract_eqs(output,invalids=['{', "}"]):

    eq_list = []
    for st, et in zip(re.finditer("<res>" , output), re.finditer("</res>" , output)):
        eq_list.append(output[st.end():et.start()])
    eq_str = "\n".join(eq_list)
    # delete invalid symbols
    for invalid in invalids:
        eq_str = eq_str.replace(invalid, "")
    return eq_str

def replace_consts(input_string):
    # Counter to keep track of the index for replacements
    count = 0
    
    # Custom replacement function
    def repl(match):
        nonlocal count
        # Construct replacement string with the current count
        replacement = f"c_{count}"
        # Increment the count
        count += 1
        return replacement
    
    # Use 're.sub' with the custom replacement function
    result_string = re.sub(r'const', repl, input_string)
    
    return result_string,count

class Program:

    def __init__(self, expr, lhs, features):
        self.optimizer = ScipyMinimize()
        self.expr = expr
        exp_str,count = replace_consts(expr)
        self.eqs_sympy = sp.sympify(exp_str)
        self.const_symbols = sp.symbols([f'c_{i}' for i in range(count)])
        self.var_symbols = sp.symbols([i for i in features.keys()])
        self.lhs = lhs.reshape(-1)
        self.y_rhs = [value for key, value in features.items()]
        self.init_const = [ random.random() for _ in range(count)] 

    def loss(self,rhs):  
        loss = np.mean(np.square(rhs-self.lhs))
        return loss
        
    def process_sym(self, consts):
        const_subs = dict(zip(self.const_symbols, consts))
        eq_subs = self.eqs_sympy.subs(const_subs)
        f = sp.lambdify(self.var_symbols, eq_subs, 'numpy')
        rhs =f(*self.y_rhs)
        # 
        loss = self.loss(rhs)
        # print(loss)
        return loss
    
    def rhs_evaluate(self,consts,y_rhs):
        const_subs = dict(zip(self.const_symbols, consts))
        eq_subs = self.eqs_sympy.subs(const_subs)
        f = sp.lambdify(self.var_symbols, eq_subs, 'numpy')
        rhs =f(*y_rhs)   
        return rhs
         
    def optimize_constants(self,): 
        
        if len(self.init_const)>0: 
            # import pdb;pdb.set_trace()
            consts = self.optimizer(self.process_sym, self.init_const)
        else:
            consts=self.init_const
        return consts
    
class Evaluator:
    """
    receive expressions and give their rewards
    """

    cache = {}
    def __init__(self, 
                 data_name,
                 metric = 'sparse_reward',
                 metric_params = [0.01],
                 top_k = 5,
                 max_terms = 5,
                 l0_penalty =10**-5,
                 mode = 'sparse_regression',
                 add_const = 0,
                 noise = 0
                 ):
        
        if 'ODE' in data_name:
            dataclass = ODEData()
            self.lhs, self.feature_dict,self.train_rhs,self.test_rhs,self.y_test,self.lhs_test = dataclass.load_data(data_name, noise)
        else:
            self.train_rhs,self.test_rhs = None, None
            self.lhs, self.feature_dict = data_load(data_name)

        self.data_name = data_name
        self.feature_names = self.feature_dict.keys()
        self.max_terms = max_terms
        self.l0_penalty = l0_penalty
        # num_features* dim
        self.features = [self.feature_dict[name] for name in self.feature_names]
        self.mode = mode
        self.metrics = make_metric(metric, *metric_params)
        self.invalid = {}
        self.pq = PriorityQueue(top_k)
        self.add_const = add_const

    def evaluate_score_nonlinear(self,eq_list):
        equations = []
        eq_scores = {}
        invalid = []
        symbols = create_sympy_symbols(list(self.feature_names)+['const'])
        for i, eq in enumerate(eq_list):
            try:
                eq_sympy = str2sympy(eq, list(self.feature_names)+['const'])

                # check error( symbol, order)
                string_error = check_symbols_valid(eq_sympy, symbols)
                if string_error :
                    self.invalid[string_error] = 1+self.invalid.get(string_error,0)
                    continue
                func_strs = count_functerms( eq_sympy)
                if len(func_strs)>6:
                    continue   
                    
                        
                with warnings.catch_warnings():
                    p_eq = Program( eq, self.lhs, self.feature_dict)

                    if len(p_eq.init_const)>5:
                        continue
                    coefs = p_eq.optimize_constants()
                    y_rhs = p_eq.rhs_evaluate(coefs, self.features)
                    score = self.metrics(self.lhs.reshape(-1), y_rhs, len(func_strs))   
                    
                    extra_metric = {}
                    r2_train =  r2_score(self.train_rhs, y_rhs)
                    extra_metric['r2_train'] = r2_train

                # import pdb;pdb.set_trace()
                if self.test_rhs is not None:
                    y_rhs_test = p_eq.rhs_evaluate(coefs, self.y_test)
                    r2_test = r2_score(self.test_rhs, y_rhs_test)
                    extra_metric['r2_test'] = r2_test

            except ValueError as e:
                self.invalid[e] = 1+self.invalid.get(e,0)
                # print(e)
                continue
            except Exception as e2:
                # print(e2)
                self.invalid[e2] = 1+self.invalid.get(e2,0)
                continue
            
            if eq not in eq_scores:
                eq_scores[eq] = round(score,4)
                len_ori = len(func_strs)
                equation = Equation(eq, round(score,4), coefs, eq_sympy, len_ori, extra_metric)
                equations.append(equation) 

        return  invalid, equations        

    def evaluate_score(self,eq_strings):
        '''
        
        '''
        eq_list = self.preprocess(eq_strings)
        eq_linear,eq_nonlinear = [],[]
        # import pdb;pdb.set_trace()
        for eq in eq_list:
            if len(eq)<2:
                continue
            if 'const' not in eq:
                eq_linear.append(eq)
            else:
                eq_nonlinear.append(eq)

        original_mode = self.mode
        if self.mode == 'nonlinear':    
            self.mode = 'sparse_regression'
     
        invalid_linear, eqs_linear = self.evaluate_score_linear(eq_linear)
        self.mode = original_mode
        invalid_nonlinear, eqs_nonlinear  = self.evaluate_score_nonlinear(eq_nonlinear)
        
        # invalid = merge_dict(invalid_linear, invalid_nonlinear)
        eqs = eqs_linear+eqs_nonlinear

        return len(eqs_nonlinear), eqs
   
    def evaluate_score_linear(self, eq_list):
        # 1. convert strings to eq_list
        
        eq_scores = {}
        const_flag = False
        symbols = create_sympy_symbols(self.feature_names)
        invalid = [] 
        equations = []
        for i, eq in enumerate(eq_list):

            valid = False
            len_ori = 0
            # convert eq to sympy expressions
            # print(eq)
            # import pdb;pdb.set_trace()
            try:
                eq_sympy = str2sympy(eq, self.feature_names)

                # check error( symbol, order)
                string_error = check_error(eq,eq_sympy, symbols)
                if string_error is not None:
                    self.invalid[string_error] = 1+self.invalid.get(string_error,0)
                    continue
                
                func_terms, func_strs = walking_tree(symbols, eq_sympy, self.features)
                len_ori = len(func_terms)
                func_terms, func_strs, duplicates = remove_redundants(func_terms, func_strs)

            except SyntaxError as Se:
                print(Se)    
                continue       
            except Exception as e:  
                # print(e)
                self.invalid[e] = 1+self.invalid.get(e,0)
                continue
            else:
                pass

            func_strs, y_rhs, coef, valid , error_type = linear_calculate(func_terms,func_strs, self.lhs,self.add_const,self.mode)  
            
            if not valid:
                self.invalid[error_type] = 1+self.invalid.get(error_type,0)
                continue
            
            if valid:
                # if self.mode == 'sparse_regression' and  coef[-1]!=0:
                #     #extract constant number
                #     const_flag = coef[-1]
                #     coef = coef[:-1]

                func_strs = [func_strs[i] for i in range(len(coef)) if coef[i]!=0][:self.max_terms]
                func_terms = [func_terms[i] for i in range(len(coef)) if coef[i]!=0][:self.max_terms]
                coef = [c for c in coef if c!=0][:self.max_terms]
                duplicates=True                
            
            if (duplicates or ("(" in eq and ")" in eq)) and valid :
                try:
                    eq, id_convert = reorganize(func_strs)
                    eq_sympy = str2sympy(eq, self.feature_names)
                except Exception as e:
                    continue
                
                new_coef = [coef[i]*-1 if i in id_convert else coef[i] for i in range(len(func_strs)) ]
                coef = new_coef
                
            if valid:
                score = self.metrics(self.lhs, y_rhs, len(func_terms))   
                if self.test_rhs is not None:
                    r2_train =  r2_score(self.train_rhs, y_rhs)
                    try:
                        func_terms_test, fun_strs_test = walking_tree(symbols, eq_sympy, self.y_test)
                        func_strs, y_rhs_test, _, valid , _ = linear_calculate(func_terms_test,fun_strs_test, self.lhs_test,self.add_const,self.mode) 
                        r2_test = r2_score(self.test_rhs, y_rhs_test)
                    except:
                        r2_test = 0
                    extra_metrics={}
                    extra_metrics['r2_train'] = r2_train
                    extra_metrics['r2_test'] = r2_test
                else:
                    extra_metrics=None
                if eq not in eq_scores:
                    eq_scores[eq] = round(score,4)
                    equation = Equation(eq, round(score,4), coef, eq_sympy, len_ori, extra_metrics)
                    equations.append(equation)
            else:
                self.invalid[error_type] = 1+self.invalid.get(error_type,0)

        return  invalid, equations
       
    def preprocess(self, llm_out):

        if '<' in llm_out or '>' in llm_out:
            # 
            llm_out =  extract_eqs(llm_out)
        # import pdb;pdb.set_trace()
        eq_list = llm_out.split('\n')
        standard_eqs = []
        split_symbol = ['.', ':']
        
        for eq in eq_list:
            # Remove irregular spaces            
            expression = re.sub(r'\s+', ' ', eq)
            # replace possible ,
            expression = re.sub(r',', '+', eq)
            try:
                ind=None
                for sym in split_symbol:               
                    if sym in expression:
                        ind = expression.index(sym)
                        if ind<10:
                            ind+=1
                            while expression[ind]==' ':
                                ind+=1
                            break

                if ind is None:
                    ind=0
                expression = expression[ind:]
                standard_eqs.append(expression)
            except Exception as e:
                print("Error type:", e, "; Exp:", eq)
                continue
        return standard_eqs


if __name__ == "__main__":

    data_name = 'chafee-infante'
    eva = Evaluator(data_name)

    output = """ 
    1. u_xxx^2 - u^3 + u_x - u_xx - u*u_x^2 - u*u_x^2 + u + u*u_x + u_xxx^2 + u_xx^2"""
    result = eva.evaluate_score(output, True)
    print(result)

