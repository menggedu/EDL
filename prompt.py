
intial_prompt2 = """
You will help me find the optimal governing equation from  data. You are required to generate equations using symbolic representations. \
I will evaluate them and provide their corresponding scores based on their fitness to data. \
Your task is to find the equation with the highest score.  \
The available symbol library for representing equations includes two categories: operators and operands. \
Operators include {0}. \
Operands include {1}, where w,u,v denote the state variable and others are derivatives. \
Now randomly generate {2} diverse equations with different lengths in the following form:
1. ... 
2. ...
Be as creative as you can under the constraints below.
(1) Do not include coefficients. 
(2) Do not omit the multiplication operator. 
(3) Only use the symbols provided in the symbol library. 

Do not write code and do not give any explanation.
"""


intial_prompt = """
You will help me find the optimal governing equation from  data. You are required to generate equations using symbolic representations. \
I will evaluate them and provide their corresponding scores based on their fitness to data. \
Your task is to find the equation with the highest score.  \
The available symbol library for representing equations includes two categories: operators and operands. \
Operators include {0}. \
Operands include {1}, where u denotes the state variable, x denotes the spatial variable, and others are derivatives. \
Now randomly generate {2} diverse equations with different lengths in the following form:
1. ... 
2. ...
Be as creative as you can under the constraints below.
(1) Do not include coefficients. 
(2) Do not omit the multiplication operator. 
(3) Only use the symbols provided in the symbol library. 

Do not write code and do not give any explanation.
"""

optimize_prompt="""
Task: Given the symbol library including operators:{0} and operands: {1}, You will help me generate governing equations using symbolic representations. \
I will evaluate the generated equations and provide their corresponding scores based on their fitness to data. Your task is to find the equation with the highest score.

Below are some previous equations and their scores, which range from 0 to 1. \
The equations are arranged in ascending order based on their scores, where higher values are better. 
{2}
Motivated by the equations above, please help me generate {3} new equations with higher scores, and bracketed them with <res> and </res>. 
Try to recognize and avoid redundant terms and generate new possibly correct terms.
Pay attention to the format, do not give the scores of the equation and do not give additional explanations
"""


evolution_prompt = """
Task: Given the symbol library including operators:{0} and operands: {1}, and following set of terms: 
{2}
please follow the instructions step-by-step to generate new equations:
1. Select two different set of terms from above and bracket them with <select> and </select>.
2. Crossover two set of terms chosen in step 1 and generate a new equation bracketed with <cross> and </cross>.
3. Mutate the equation generated in Step 2 and generate a new equation bracketed with <res> and </res> and .
4. Repeat step 1, 2, 3 for {3} times and directly give me the generated equations of each step.

Define Crossover and Mutate.
Crossover: Select half terms of each set and recombine the selected with '+' and '-' to generate a new equation.
Mutate: Randomly replace operators or operands of the equation with new ones defined in the symbol library.
Pay attention to the format and do not include symbols outside of the library. Do not give any explanation.
"""
