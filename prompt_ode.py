

intial_prompt = """
You will help me find the optimal expression from  data. You are required to generate expressions using symbolic representations. \
I will evaluate them and provide their corresponding scores based on their fitness to data. \
Your task is to find the expression with the highest score.  \
The available symbol library for representing expressions includes two categories: operators and operands. \
Operators include {0} and operands include {1}.\
Now randomly generate {2} diverse expressions with different lengths in the following form:
1. ... 
2. ...
Be as creative as you can under the constraints below.
(1) Do not include coefficients. 
(2) Do not omit the multiplication operator. 
(3) Only use the symbols provided in the symbol library. 

Do not write code and do not give any explanation.
"""

optimize_prompt="""
Task: Given the symbol library including operators:{0} and operands: {1}, You will help me generate expressions using symbolic representations. \
I will evaluate the generated expressions and provide their corresponding scores based on their fitness to data. Your task is to find the expression with the highest score.

Below are some previous expressions and their scores, which range from 0 to 1. \
The expressions are arranged in ascending order based on their scores, where higher values are better. 
{2}
Motivated by the expressions above, please help me generate {3} new expressions with higher scores, and bracketed them with <res> and </res>. 
Try to recognize and remove redundant terms and add new possibly correct terms.
Pay attention to the format of outputs, do not give the scores of the expression and do not give additional explanations
"""


evolution_prompt = """
Task: Given the symbol library including operators:{0} and operands: {1}, and following expressions: 
{2}
please follow the instructions step-by-step to generate new expressions:
1. Select two different expressions from above and bracket them with <select> and </select>.
2. Crossover two expressions chosen in step 1 and generate a new expression bracketed with <cross> and </cross>.
3. Mutate the expression generated in Step 2 and generate a new expression bracketed with <res> and </res>.
4. Repeat step 1, 2, 3 for {3} times and directly give me the generated expressions of each step.

Define Crossover and Mutate.
Crossover: exchanged or swapped terms of selected parent expressions and generate a new expression.
Mutate: Randomly alter operators or operands based on the provided symbol library and generate a new expression.

Pay attention to the format of outputs, and do not include symbols outside of the library. Do not give any explanation.
"""
