import numpy as np
from sympy import sympify, Add, Mul, Pow, cos, Function,srepr
from sympy.core.symbol import Symbol
from sympy import sympify, preorder_traversal
from sympy.parsing.sympy_parser import (parse_expr, _token_splittable,
standard_transformations,convert_xor, implicit_multiplication,split_symbols_custom, implicit_multiplication_application)
import sympy
import functools
import unicodedata
from sympy import lambdify


from evaluation.sympy_utils import create_sympy_symbols

def test_accept(sample):
    if sample.extra_metric is not None:
        for key, item in sample.extra_metric.items():
            if item<0.99:
                return False
    return True

def infix_to_prefix(infix_expr, operands):
    """Converts an infix expression to a prefix expression.

    Args:
    infix_expr: The infix expression to convert.
    operands: u,x, etc

    Returns:
    The prefix expression.
    """

    # Create a stack to store operators.
    stack = []

    # Create a queue to store operands.
    queue = []

    # Iterate over the infix expression.
    for token in infix_expr:
    # If the token is an operand, add it to the queue.
        if token in operands:
            queue.append(token)

        # If the token is an operator, pop the two operands from the stack,
        # apply the operator to them, and add the result to the stack.
        elif token in ["+", "-", "*", "/", "^2"]:
            operand1 = queue.pop()
            operand2 = queue.pop()
            result = token + operand1 + operand2
            stack.append(result)

    # The prefix expression is the contents of the stack, in reverse order.
    return "".join(stack[::-1])
  
def can_split(symbol, unsplit_names = []):
    if symbol not in unsplit_names:
        return _token_splittable(symbol)
    return False

# Function to convert expression to prefix traversal
def to_prefix(expr):
    stack = []

    # Preorder traversal to produce prefix notation
    for node in preorder_traversal(expr):
        if node.is_symbol or node.is_number:
            stack.append(str(node))
        elif node.is_Add or node.is_Mul or node.is_Pow or node.is_Function:
            op = '+' if node.is_Add else '*' if node.is_Mul else '**' if node.is_Pow else node.func.__name__
            stack.append(op)
    return stack # Reverse to get the correct prefix order

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
    for subtree in expr.args:
        f = lambdify(symbols, subtree)
        sub_values.append(f(*input))
    
    return sub_values

def initilization():
    info = """
    0. {u_x*u_xx, u_xxx^2}
    1. {u_x*u_xxx, u_xx*x^3}
    2. {u, u_x*u_xxx, u_xxx}
    3. {u_x*u_xxx, u_xx*x^2}
    4. {u_xxx*x, u_xx^2}
    5. {u_xx*x, u_x*u_xxx}
    6. {u_x*u_xx, u_xx, u_xxx^2}
    7. {u^2, u_xxx, u_x*u_xx}
    8. {u_x*u_xxx, u_xx^2*x^2}
    9. {u_x*u_xxx, u_xx^2*x}
  """
#     info = """
#     0. u_x*u_xx - u_xxx^2  
#     1. u_x*u_xxx + u_xx*x^3 
# """
#     info = """
# 1. {u_x*u_xx - u_xxx^2}; {u_x*u_xxx + u_xx*x^3}
# 2. {u_x*u_xxx + u_xx*x^3}; { u + u_x*u_xxx + u_xxx}
# 3. {u_x*u_xxx - u_xx*x^2}; {u_xxx*x + u_xx^2}
# 4. {u_xx*x - u_x*u_xxx}; {u_x*u_xx - u_xx + u_xxx^2}
# 5. {u^2 - u_xxx + u_x*u_xx}; {u_x*u_xxx - u_xx^2*x^2}
# """
    info="""
0. u_x*u_xx - u_xxx^2   score: 0.5887
1. u_x*u_xxx + u_xx*x^3   score: 0.5922
2. u + u_x*u_xxx + u_xxx   score: 0.5981
3. u_x*u_xxx - u_xx*x^2   score: 0.6011
4. u_xxx*x + u_xx^2   score: 0.6012
5. u_xx*x - u_x*u_xxx   score: 0.6087
6. u_x*u_xx - u_xx + u_xxx^2   score: 0.6112
7. u^2 - u_xxx + u_x*u_xx   score: 0.6153
8. u_x*u_xxx - u_xx^2*x^2   score: 0.7243
9. u_x*u_xxx + u_xx^2*x   score: 0.7369
"""
    info = """
0: exp(2)*exp(x)*log(x + 3)
1: exp(2)*exp(x)*cos(x) - exp(2)*exp(x)
2: const*x - const - 2 + 2*x
3: 16/(2*x - 1)
4: 9*const - const*x^2 + 6*const*x
5: 112*x
6: 2*x*exp(x) + 4*exp(x)
7: 16/log(x) + 16*x/log(x)
8: 5*x - 4*exp(x)
9: 2*log(x)
10: 3*x + 4*log(x + 1)
11: log(x + 2) + exp(x)
12: 4*x*log(x) + 8*x
13: 5*x - 2*sin(x)
14: x*log(x + 2) - log(x + 2)
"""
    return info

    # """
    # 1. u_x*u_xx - u_xxx^2  and u_x*u_xxx + u_xx*x^3 
    # 2. u_x*u_xxx + u_xx*x^3 and u + u_x*u_xxx + u_xxx 
    # 3. u_x*u_xxx - u_xx*x^2 and u_xxx*x + u_xx^2 
    # 4. u_xx*x - u_x*u_xxx and u_x*u_xx - u_xx + u_xxx^2 
    # 5. u^2 - u_xxx + u_x*u_xx and  u_x*u_xxx - u_xx^2*x^2
    # """
if __name__ == "__main__":
    operators_operands = ['+', '-', '*', '/', '^2']
    operands = ['u', 'x','u_x', 'u_xx' ,'u_xxx']
    symbols = create_sympy_symbols(operands)

    symbols = [sympy.Symbol(i) for i in operands]
    # Define the expression
    expr_str = "u_x + (u-x) + u* u_xx ^2"  # Changed ^2 to **2 for correct parsing

    call_can_split = functools.partial(can_split, 
                                    unsplit_names =operands)
    symbol_split_trans = split_symbols_custom(call_can_split)

    # Apply standard transformations and implicit multiplication support
    transformations=(standard_transformations + ( symbol_split_trans,convert_xor, implicit_multiplication))
    # Define the list of operators and operands
    # Parse the string into a SymPy expression
    expr = parse_expr(expr_str, local_dict = { "^2": lambda x: x**2,} , evaluate=False, transformations=transformations)
    # expr = sympy.sympify(expr_str, locals = { "^2": lambda x: x**2,})
    print(expr)
    print(expr.args)

    print(srepr(expr))

    # Convert to prefix notation
    prefix_expr = to_prefix(expr)

    print(prefix_expr)



    f = lambdify(symbols, expr)
    print(f(*np.ones((3,5)).T))

    print(walking_tree(expr,np.ones((3,5)).T ))