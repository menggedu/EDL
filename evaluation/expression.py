import heapq
import random
import sympy
from sympy import count_ops, preorder_traversal

class Equation:
    def __init__(self, exp_str, score, coef,sympy_exp,len_ori = None, extra_metric = None):
        self.exp_string = exp_str
        self.reward = score
        self.sympy_exp = sympy_exp
        self.coef = coef
        self.coef_str = " coef:["+",".join([str(round(i,4)) for i in self.coef])+"]"
        self.len_ori = len_ori
        self.split_and_permutation()
        self.extra_metric=extra_metric

    def split_and_permutation(self):
        # split terms and permutation
        link_symbol = [' + ',' - ']
        # terms = list(self.sympy_exp.args)
        terms = list(sympy.Add.make_args(self.sympy_exp))
        random.shuffle(terms)
        new_string = ''
        terms_str = []
        for term in terms:
            if str(term).startswith('-'):
                term = str(term)[1:]
            else:
                term = str(term)
            term_converted = term.replace('**', '^')
            terms_str.append(term_converted)
            new_string+= term_converted + random.choice(link_symbol)
        self.permutation_string = new_string[:-3]
        self.terms_str = terms_str

    @property
    def score(self):
        return self.reward
    
    @property
    def exp_str(self):
        return self.exp_string
    
    @property
    def complexity(self):
        oper_sums = len(list(preorder_traversal(self.sympy_exp)))
        return oper_sums
        # return count_ops(self.sympy_exp)+2*len(self.coef)

    @property
    def permutation_allowed(self):
        return len(self.sympy_exp.args) >1
    
    def __repr__(self):
        return self.exp_str+ self.coef_str 
    
    def __lt__(self, other):
        assert isinstance(other, type(self))
        return self.score < other.score

    def __eq__(self, other):
        assert isinstance(other, type(self))
        return self.score == other.score
    
    def __len__(self):
        return len(self.terms_str)
    
    @property
    def terms(self):
        return "|".join(self.terms_str)
    
    @property
    def permutation_str(self): 
        return self.permutation_string

    def print_extra(self):
        if self.extra_metric is not None:
            for key, item in self.extra_metric.items():
                print(f'{key}: {item}')

        
    # def __iter__(self):
    #     """Allows unpacking like a tuple."""
    #     yield self.score
    #     yield self.exp_str
    #     yield self.coef

class PriorityQueue:
    def __init__(self, k):
        self.k = k
        self.heap = []
        self.scores = []

    def push(self, sample):
        # Sample should be a tuple of (score, coefficient, string)
        if not isinstance(sample,list):
            samples = [sample]
        else:
            samples = sample   

        for sample in samples:
            # If the heap is smaller than k, we push the new sample onto the heap
            if len(self.heap) < self.k:
                heapq.heappush(self.heap, sample)
                self.scores.append(sample.score)
            else:
                # Check if the new sample's score is greater than the smallest score in the heap
                if sample.score > self.heap[0].score and sample.score not in self.scores:
                    # Pop the smallest item and push the new sample
                    heapq.heappushpop(self.heap, sample)
                    self.scores.append(sample.score)

    def get_top_samples(self):
        # Return the k samples with the highest scores in descending order
        return sorted(self.heap, key=lambda x: x.score, reverse=True)
    
    def get_samples(self):
        samples = []
        for eq in heapq.nlargest(len(self.heap), self.heap):
            samples.append(eq)
        
        return samples

    def __str__(self):
        return str(self.get_top_samples())
    
    def pop(self):
        eq = heapq.heappop(self.heap)
        return eq

    def iter_in_order(self, reverse=True):
        func = heapq.nsmallest if not reverse else heapq.nlargest
        for eq in func(len(self.heap), self.heap):
            yield eq

    def get_max(self,):
        eq  =heapq.nlargest(1, self.heap)[0]
        return eq
    
    def __len__(self):
        return len(self.heap)
    
    @property
    def prompt_str(self, reverse =True):
        out_info = ""
        if len(self.heap)>0:
            for i, eq in enumerate(self.iter_in_order(reverse)):
                out_info+=f"{str(eq)}   score: {eq.score}\n"

        return out_info