"""
Utilizing llm to generate the initial samples for the discovery.

"""

import functools
import random
import time


import prompt_utils as prompt_utils


def organize(info, optimize_type, sort_type="not_reverse", pqt=None, use_pqt=True, ode=False):
    out_info = ""
    info_score = ""
    if pqt is not None and len(pqt) and use_pqt:
        new_samples = pqt.get_samples()
        scores = [term.score for term in info]
        new_samples_filtered = [term for term in new_samples if term.score not in scores]
        info.extend(new_samples_filtered)

        if sort_type == 'not_reverse':
            info = sorted(info, key = lambda x: x.score, reverse=False)
        elif sort_type == 'reverse':
            info = sorted(info, key = lambda x: x.score, reverse=True)
        else:
            pass

    print(f"{len(info)} equations are utilized")
    for i in range(len(info)):
        # import pdb;pdb.set_trace()
        eq_str = info[i].permutation_str
        if optimize_type == 'evolution':
            if not ode:
                terms = info[i].terms_str
                terms_str = "{"+ ", ".join(terms)+"}"
                out_info+=f"{i}: {terms_str}\n"
            else:
                out_info += f"{i}: {eq_str}\n"
            
        else:
            out_info+=f"{i}: {eq_str}   score: {info[i].score}\n"

        info_score += f"{i}: {eq_str}   score: {info[i].score}\n"
    return out_info, info_score

def filter_score(equations, limit,score_set,cache, filter_same=True):
    eqs = []
    assert isinstance(score_set, list)
    score_set = set(score_set)

    for i in range(len(equations)):
        if equations[i].score>limit:
            if equations[i].score in score_set and filter_same:
                continue
            else:
                score_set.add(equations[i].score)
                cache[equations[i].score] = equations[i].exp_str
                eqs.append(equations[i])

    return eqs

def filter_score_from_start(eq_score,equations, limit,cache, filter_same=True):
    eq_score_new = []
    eqs = []
    assert len(equations) == len(eq_score)
    for i in range(len(eq_score)):
        if eq_score[i][1]>limit:
            if eq_score[i][1] in cache and filter_same:
                continue
            else:
                cache[eq_score[i][1]] = eq_score[i][0]
                eq_score_new.append(eq_score[i])
                eqs.append(equations[i])

    return eq_score_new,eqs


def prompt_generation(args,call_type, info, sampling):
    if 'ODE' in args.data_name:
        if args.mode == 'nonlinear':
            if 'llama' in args.LLM_name:
                from prompt_llama import intial_prompt, optimize_prompt, evolution_prompt
            else:
                from prompt_ode2 import intial_prompt, optimize_prompt, evolution_prompt

        else:
            from prompt_ode import intial_prompt, optimize_prompt, evolution_prompt
    else:
        from prompt import intial_prompt, optimize_prompt, evolution_prompt

    utlized_prompt = None
    operators = args.operators
    operands = args.operands
    if sampling:
        operator_list = operators[1:-1].split(',')[2:] #+ - not included
        random.shuffle(operator_list)
        
        operator_sub = operator_list[:len(operator_list)//2]
        operators = '{+, -,'+','.join(operator_sub)+ '}'
        print(operators)

    if call_type == 'initialization':
        print("run initialization")
        prompt = intial_prompt
        utlized_prompt = prompt
        prompt_used = utlized_prompt.format(operators,operands, args.init_num)
        # print(prompt_used)

    elif call_type == 'optimize':
        print("run optimization") 
        utlized_prompt =  optimize_prompt
        prompt_used = utlized_prompt.format(operators,operands,info, args.N )
        # print(prompt_used)

    elif call_type == 'evolution':
        # print("run evolution") 
        # prompt_used = evolution_prompt2.format(args.operators,args.operands,info, args.N )
        utlized_prompt = evolution_prompt
        if args.evo_type == 'term':
            prompt_used = utlized_prompt.format(operators,operands,info, args.N)
        elif args.evo_type == 'equation':
            prompt_used = utlized_prompt.format(operators,operands,info, args.N)
        else:
            assert False
        # print(prompt_used)
    else:
        print(call_type)
        assert False, "unrecognized calling type"

    return prompt_used

def call_optimizer(llm_dict, args, evaluator, info=None, call_type='optimize'):
    """
    pass

    Args:
        llm_dict (_type_): 
        args
    """
    
    prompt_used = prompt_generation(args, call_type, info, sampling = False)
    # print(prompt_used)

    GENERATION_NUM = args.N
    all_eqs=[]
    eqs = []
    scores = []
    call_server_func = functools.partial(
            prompt_utils.call_openai_server_func,
            model=llm_dict['name'],
            max_decode_steps=llm_dict['max_decode_steps'],
            model_pretrain = llm_dict['model_pretrained'],
            tokenizer = llm_dict['tokenizer']
        )
    seed = args.seed
    dur_time = []  # time_cost by request of llm
    flag = False
    num_nonlinear = 0
    temperature=llm_dict['temperature']
    while len(eqs)<GENERATION_NUM or flag:
        # ====================== try calling the servers ============================
        print("\n======== launch the llm server ===========")
        start_time = time.time()
        test_output = call_server_func(
            prompt_used,
            temperature=temperature,
            seed = seed
        )
        
        visit_time = time.time()
        dur_time.append(visit_time-start_time)

        if 'ODE' in args.data_name:
            # sampling library to form a new prompt
            prompt_used = prompt_generation(args, call_type, info, sampling = True)
            # print(prompt_used)
            if num_nonlinear>0:
                flag=False
            else:
                # if call_type == 'initialization':
                flag =True
            if seed>args.seed+10 and len(eqs)>0:
                flag=False
                break
        # print(f" test output: \n{test_output[0]}")
        # print("\n=================================================")
    
        num_nonlinear, equations = evaluator.evaluate_score(test_output[0])
        if 'llama' in args.LLM_name:
            tem = random.random()
            temperature = round(tem, 2)

        if len(equations)>0:
            all_eqs.extend(equations)
            equations_filter =  filter_score(equations,args.reward_limit,scores, evaluator.cache)
            scores.extend([term.score for term in equations_filter])
            # eq_score_filtered,equations = filter_score_from_start( equations,args.reward_limit, evaluator.cache)
            eqs.extend(equations_filter)
        seed +=1
        
        eval_time = time.time()-visit_time
        dur_time.append(eval_time)
    print(dur_time)
    dur_time = sum(dur_time)
    sorted_eqs = sorted(eqs, key = lambda x: x.score, reverse=False)
    if args.sort == 'not_reverse':
        
        return sorted_eqs[-args.N:],sorted_eqs[-1],all_eqs,dur_time
    
    elif args.sort == 'reverse':

        sorted_eqs = sorted(eqs, key = lambda x: x.score, reverse=True)

        return sorted_eqs[:args.N], sorted_eqs[0], all_eqs,dur_time
    else:
        return sorted_eqs[-args.N:],sorted_eqs[-1], all_eqs, dur_time






