
import datetime
import functools
import json
import os
import re
import sys
import numpy as np
import openai
import argparse
from time import time
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Inner
from optimzier_utils import call_optimizer,organize
from evaluation import *
import prompt_utils
from utils import initilization,test_accept
from logger import StatsLogger

def call_openai(api_key ):
    client = OpenAI(
        base_url="https://oneapi.xty.app/v1",
        api_key=api_key
    )

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    )

    print(completion)

def test_gpt(openai_api_key,llm_name='gpt-3.5-turbo', ):
    openai.api_key = openai_api_key
    openai.api_base = "https://oneapi.xty.app/v1"
    gpt_max_decode_steps = 1024
    gpt_temperature = 1.0

    llm_dict = dict()
    llm_dict['name'] = llm_name
    llm_dict["max_decode_steps"] = gpt_max_decode_steps
    llm_dict["temperature"] = gpt_temperature
    llm_dict["batch_size"] = 1
    call_server_func = functools.partial(
            prompt_utils.call_openai_server_func,
            model=llm_dict['name'],
            max_decode_steps=llm_dict['max_decode_steps'],
            temperature=llm_dict['temperature'],
        )
    test_output = call_server_func(
        "tell me ff the sun rise from west and set down to the east",
        temperature=1.0,
    )
    print(f" test output: \n {test_output[0]}")

def main():
    parser = argparse.ArgumentParser(description='LLM for knowledge discovery')
    parser.add_argument('--LLM_name', type=str, default="gpt-3.5-turbo", help='utilized llm model')
    parser.add_argument('--openai-api-key', type=str, default="sk-xxxxx", help='utilized llm model')
    parser.add_argument('--new-add', type=int,default = 0, help='change address')
    parser.add_argument('--gpt_temperature', type=float,default = 1, help='temperature')
    parser.add_argument('--max_decode_length', type=int,default = 1024, help='decode_step')

    parser.add_argument('--N', type=int, default=4, help='Number of generated samples per iteration')
    parser.add_argument('--init-num', type=int, default=20, help='Number of generated samples at initilization')
    parser.add_argument('--operators', type=str, default="{+, -, *, /, ^2}", help='symbol library')
    parser.add_argument('--operands', type=str, default="{u, u_x, u_xx, u_xxx, x}", help='operands library')
    parser.add_argument('--optimize-type', type=str, default='optimize_optimize',help='optimize type')
    parser.add_argument('--evo-type', type=str, default='term',choices=['term', 'equation'], help='optimize type')

    parser.add_argument('--reward_limit', type=float, default=0.5, help='reward limit for initial population')
    parser.add_argument('--sort', type=str, default="not_reverse",choices=['not_reverse', 'reverse', 'Not sorted'],  help='sort or not')
    
    parser.add_argument('--max_epoch', type = int, default = 50)
    parser.add_argument('--threshold', type = float, default = 0.995)

    #task
    parser.add_argument('--data-name', type = str, default = 'chafee-infante')
    parser.add_argument('--seed', type = int, default = 1)
    parser.add_argument('--max-terms', type = int, default = 5)
    parser.add_argument('--metric', type = str, default = "sparse_reward")
    parser.add_argument('--mode', type = str, default = "sparse_regression",choices = ['sparse_regression','regression','nonlinear'])
    # parser.add_argument('--metric_params', type = list, default = "sparse_reward")      
    parser.add_argument('--metric_params', nargs='+', type=float, default = [0.01])
    parser.add_argument('--job-name', type=str, default = 'name')
    parser.add_argument('--logdir', type=str, default = "./log")
    parser.add_argument('--use-pqt', type=int, default = 1)
    parser.add_argument('--add-const', type=int, default = 0)
    parser.add_argument('--noise', type=float, default = 0)
    args = parser.parse_args()
    print(args)
    llm_name = args.LLM_name
    openai_api_key = args.openai_api_key
    openai.api_key = openai_api_key
    if args.new_add:
        openai.api_base = "https://oneapi.xty.app/v1"

    if llm_name in {"gpt-3.5-turbo", "gpt-4"}:
        assert openai_api_key, "The OpenAI API key must be provided."
    
    # =================== create the result directory ==========================
    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    datetime_str = (
        str(datetime.datetime.now().replace(microsecond=0))
        .replace(" ", "-")
        .replace(":", "-")
    )

    assert llm_name in {"gpt-3.5-turbo", "gpt-4","llama-7b", "llama-13b"}
    gpt_max_decode_steps = args.max_decode_length
    # gpt_temperature = 1.0
    if 'llama' in llm_name:
        # import pdb;pdb.set_trace()
        model_pretrained, tokenizer = prompt_utils.load_model(llm_name)
    else:
        model_pretrained,tokenizer = None, None

    llm_dict = dict()
    llm_dict['name'] = llm_name
    llm_dict["max_decode_steps"] = gpt_max_decode_steps
    llm_dict["temperature"] = args.gpt_temperature
    llm_dict["batch_size"] = 1
    llm_dict["model_pretrained"] = model_pretrained
    llm_dict["tokenizer"] = tokenizer

    ODE_flag = 'ODE' in args.data_name

    optimize_type = args.optimize_type.split("_")
    evaluator = Evaluator(args.data_name, metric= args.metric, 
                          metric_params = args.metric_params,
                          max_terms=args.max_terms,
                          mode = args.mode,
                          add_const = args.add_const,
                          noise = args.noise
                          )
    start_time = time()
    # initilization
    logger = StatsLogger(args)
    total_request_time = 0
    result_init, best_sample, eqs,dur_time = call_optimizer(llm_dict, args, evaluator, call_type = 'initialization')
    optimizer_input = result_init
    print(dur_time)
    # info_str = optimizer_input = initilization()
    # dur_time=10
    total_request_time +=dur_time
    results = {'total_expression':len(eqs)}
    
    for i in range(args.max_epoch):
        begin_time = time()
        cur_optimize_type = optimize_type[(i)%2]
        logger.save_stats(optimizer_input, i)
        info_str, info_score = organize(optimizer_input, cur_optimize_type, args.sort, evaluator.pq, use_pqt = args.use_pqt,ode = ODE_flag )
        evaluator.pq.push(optimizer_input)
        print(f"{i} iteration:\n ",info_score)  
        
        optimizer_input,best_sample_cur,all_eqs,dur_time = call_optimizer(llm_dict, args, evaluator, 
                                          info=info_str, call_type =cur_optimize_type)
        total_request_time+=dur_time
        print(f"{cur_optimize_type}:  best expression: {str(best_sample_cur)}, score: {best_sample_cur.score}, duration: {time()-begin_time}")
        best_sample_cur.print_extra() # print extra metric 
        
        best_sample = best_sample_cur
        print(f"Cur best expression:{best_sample.exp_str}, reward: {best_sample.score}")
        if  best_sample.score>=args.threshold and test_accept(best_sample):
            print("early stopping")
            print(f"best expression:{best_sample.exp_str}, reward: {best_sample.score}")
            break

    
    info_str, info_score = organize(optimizer_input, cur_optimize_type, args.sort, evaluator.pq,ode = ODE_flag )
    evaluator.pq.push(optimizer_input)
    logger.save_stats(optimizer_input, args.max_epoch)
    logger.save_results()
    print(f"{i} iteration:\n", info_score)
    print(f"total duration is {time()-start_time}")
    print(f"total request time is ", total_request_time )

    print(evaluator.pq.prompt_str)
    total_invalid = sum(list(evaluator.invalid.values()))
    print(f"total invalid: {total_invalid}")
    # for key,value in evaluator.invalid.items():
    #     print(str(key)+": "+str(value))
    # (2) print(evaluator.invalid)
if __name__ == "__main__":
    main()

    #test_api
    # api = 'sk-SVjwqbhaZyhMjNxx3cF60eB1E2Aa4801Ab9cFc3c9a439e6a'
    # test_gpt(api, "gpt-4")

    # """test evaluator """
    # data_name = 'chafee-infante'
    # eva = Evaluator(data_name)

    # output = """ 1. (u_x)^2 + (u_xx)*(u_x)
    #             2. (u_xxx) - (u_xx)*(u_x)
    #             3. (u_x) + (u_xx)*(1 + (u_x)^2)
    #             4. (u_x)/(u_xx)
    #             5. (u_xx)*(u_x) - (u_x)*(u_xxx) + (u)^2
    #             6. u_xx+u+u*u^2"""
    # result, invalid = eva.evaluate_score(output, False)
    # print(result)
    # print(invalid)

