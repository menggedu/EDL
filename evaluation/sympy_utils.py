"""Define utilities to export to sympy"""
from typing import Callable, Dict, List, Optional
import sympy
from sympy import sympify,lambdify,expand, Add, Mul
from sympy.parsing.sympy_parser import (parse_expr, _token_splittable,
standard_transformations,convert_xor, implicit_multiplication,split_symbols_custom, implicit_multiplication_application)
import functools
import numpy as np
import re

def protected_div(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return np.where(np.abs(x2) > 0.00001, np.divide(x1, x2), 1.)

lamdy_mappings = {
   "/": protected_div
}
sympy_mappings = {
    "^2": lambda x: x**2,
    "^3": lambda x: x**3,
    
}

def is_valid_numeric(value):
    if np.isnan(value).any() or np.isinf(value).any():
        return False
    else:
        return True

def can_split(symbol, unsplit_names = []):
    if symbol not in unsplit_names:
        return _token_splittable(symbol)
    return False


def create_sympy_symbols(
    operands_names_in: List[str],
) -> List[sympy.Symbol]:
    
    return [sympy.Symbol(variable,real=True) for variable in operands_names_in]


def str2sympy(
    expr_str: str,
    operands: List, 
    extra_sympy_mappings: Optional[Dict[str, Callable]] = None
):
    local_sympy_mappings = {
        **(extra_sympy_mappings if extra_sympy_mappings else {}),
        **sympy_mappings,
    }

    call_can_split = functools.partial(can_split, 
                                unsplit_names =operands)
    symbol_split_trans = split_symbols_custom(call_can_split)
    transformations=(standard_transformations + ( symbol_split_trans,convert_xor, implicit_multiplication))
    
    expr_sympy = parse_expr(expr_str, local_dict = local_sympy_mappings,transformations=transformations)
    expr_sympy = expand(expr_sympy)
    return expr_sympy

def check_symbols_valid(expr, symbols):
    free_symbols = list(expr.free_symbols)
    # import pdb;pdb.set_trace()
    symbols = [str(sym) for sym in symbols]
    for sym in free_symbols:
        if str(sym) not in symbols:
            return True
    return False

def count_functerms( expr):
    if not isinstance(expr,Add ):
        return [ str(expr).replace('**', '^')]
    sub_strs = []

    for subtree in expr.args:
        sub_str = str(subtree).replace('**', '^')
        sub_strs.append(sub_str)
    return  sub_strs

def walking_tree(symbols, expr, input):
    """
    evaluate sub terms

    Args:
        expr (_type_): sympy expression
        input (_type_): input features 

    Returns:
        _type_: function term values
    """
    sub_values = []
    sub_strs = []
    # np.seterr(divide='ignore', invalid='ignore')
    if not isinstance(expr,Add ):
        value = evaluate_sympy(symbols, expr, input)
        sub_str = str(expr).replace('**', '^')
        return [value], [sub_str]

    for subtree in expr.args:
        value = evaluate_sympy(symbols, subtree, input)
        sub_values.append(value)
        sub_str = str(subtree).replace('**', '^')
        sub_strs.append(sub_str)
    return sub_values, sub_strs

def evaluate_sympy(symbols, subtree, input):
    f = lambdify(symbols, subtree, modules = 'numpy', )
    value = f(*input)
    if not is_valid_numeric(value):
        raise Exception(f"Sorry, invald expressions: {subtree} detected")
    return value

def check_order(eq, orders = ['2','3','4','5']):

    match =  re.findall(r"\^(\d+)",eq)
    for order in match:
        if order not in orders:
            return True
    
    return False

def check_error(eq, expr, symbols):
    string_error = None

    if  check_symbols_valid(expr, symbols):
        print(f"Sorry, illegal expression, {eq}")   
        string_error = 'undefined operands'

    if  check_order(eq):
        print(f"Sorry, illegal order, {eq}") 
        string_error = 'undefined operators'
    return string_error
        
