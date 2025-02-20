import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import re
import os


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import os
import numpy as np

from datetime import datetime
import pandas as pd
from itertools import compress
from io import StringIO
import shutil
from collections import defaultdict
plt.style.use(['ggplot'])

class StatsLogger():


    def __init__(self,
                  args,
                  save_all_rewards = True
                 ):
        
        self.save_all_rewards = save_all_rewards
        self.args = args

        self.out_file = self.make_out_file()
        os.makedirs(os.path.dirname(self.out_file), exist_ok=True)
        prefix, _ = os.path.splitext(self.out_file)
        self.save_all_output_file = "{}_all_r.csv".format(prefix)
        self.save_summary_file = "{}_summary.csv".format(prefix)

        self.equations = []
        self.all_info = []
        self.summary = []
        self.r_best = 0

    def make_out_file(self):

        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        # Generate save path
        log_dir = self.args.logdir
        save_path = os.path.join(log_dir, "_".join([self.args.job_name, timestamp]))

        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        seed = self.args.seed

        output_file = os.path.join(save_path,
                                   "llm_{}_{}.csv".format(self.args.job_name, seed))

        return output_file
    
    def save_stats(self, equations, epoch):

        self.equations.append(equations)
        
        best_eq = None
        r = []
        all_func = []
        local_best_r = 0
        for eq in equations:
            r.append(eq.score)
            if self.r_best<eq.score:
                self.r_best = eq.score

            if local_best_r<eq.score:
                local_best_r= eq.score
                best_eq = eq

            all_func.append([epoch, len(eq), eq.len_ori,eq.complexity, eq.terms, eq.score, eq.coef_str])

        r_mean = np.mean(np.array(r))
        r_max = np.max(np.array(r))
        r_str = ",".join([str(r_single) for r_single in r])

        single_agg = [epoch, self.r_best, r_max, r_mean, best_eq.terms, repr(best_eq), len(best_eq), r_str]       
        self.summary.append(single_agg)
        self.all_info.extend(all_func)


    def save_results(self,save_plots = True):
        columns_sum = [
            'epoch', 'r_best','r_max',  'r_mean', 'func_terms', 'equation', 'length', 'r_all'
        ]
        columns_all = [
           'epoch',  'len', 'real_len', 'complexity','func_terms', 'reward', 'coefficients'
        ]

        all_info = pd.DataFrame(self.all_info,  columns = columns_all)
        all_info.to_csv(self.save_all_output_file)

        summary = pd.DataFrame(self.summary, columns = columns_sum)
        summary.to_csv(self.save_summary_file)

        if save_plots:
            save_reward_path = os.path.join(self.save_path, 'llm_reward.png')
            self.plot_line_result(summary, 'epoch', ['r_best', 'r_mean', 'r_max'], save_reward_path)
            save_length_path = os.path.join(self.save_path, 'llm_length.png')
            
    def plot_line_result(self, results, x, y_list, save_path, x_label ='Iterations', y_label='Rewards'):
        if not isinstance(y_list, list):
            y_list = [y_list]
        fig,ax = plt.subplots(1,1)
        for y in y_list:
            sns.lineplot(data = results, x = x, y = y )
        ax.legend()
        ax.set_xlabel(x_label)   
        ax.set_ylabel(y_label)
        plt.savefig(save_path)





