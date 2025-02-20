import openai
import time

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import time

import transformers
# Use a pipeline as a high-level helper
from transformers import pipeline


from transformers import LlamaForCausalLM, LlamaTokenizer
import os



def call_openai_server_single_prompt(
    prompt, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8,seed=1
):
  """The function to call OpenAI server with an input string."""
  try:
    completion = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        max_tokens=max_decode_steps,
        messages=[
            {"role": "user", "content": prompt},
        ],
        seed = seed
    )
    return completion.choices[0].message.content

  except openai.error.Timeout as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 60
    print(f"Timeout error occurred. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.RateLimitError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 60
    print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.APIError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 60
    print(f"API error occurred. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.APIConnectionError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 60
    print(f"API connection error occurred. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.ServiceUnavailableError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 60
    print(f"Service unavailable. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except OSError as e:
    retry_time = 60  # Adjust the retry time as needed
    print(
        f"Connection error occurred: {e}. Retrying in {retry_time} seconds..."
    )
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )


def call_openai_server_func(
    inputs, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8,seed= 1,
    model_pretrain = None, 
    tokenizer = None
):
	"""The function to call OpenAI server with a list of input strings."""
	if isinstance(inputs, str):
		inputs = [inputs]
	outputs = []
	if 'llama' in model:
		output = call_llama_single_prompt(
			inputs[0],
			max_decode_steps=max_decode_steps,
			temperature=temperature,
			seed = seed,
			model_pretrain = model_pretrain,
			tokenizer=tokenizer
			)
		outputs.append(output)
	else:

		for input_str in inputs:

			output = call_openai_server_single_prompt(
			input_str,
			model=model,
			max_decode_steps=max_decode_steps,
			temperature=temperature,
			seed = seed
			)
			outputs.append(output)
	return outputs


def load_model(model_name):
	model_size = model_name.split('-')[1]
	model_dir=f"pretrain/Llama-2-{model_size}-chat-hf" 

	print('load model')
   	# mid_st = time.time()
	# print(f"cost time {mid_st-st}")
	model = LlamaForCausalLM.from_pretrained(model_dir)
	tokenizer = LlamaTokenizer.from_pretrained(model_dir)
# print(f"cost time {mid_st-st}")
	return model, tokenizer

def call_llama_single_prompt(
	input_str,
	max_decode_steps,
	temperature,
	seed,
	model_pretrain, 
	tokenizer
	):

	print('pipeline building')
	pipeline = transformers.pipeline(
		"text-generation",

		model=model_pretrain,

		tokenizer=tokenizer,

		torch_dtype=torch.float16,

		device = 0

		)
	print("inferences")

	sequences = pipeline(
		input_str,

		do_sample=True,

		top_k=1,

		num_return_sequences=1,

		eos_token_id=tokenizer.eos_token_id,

		max_length=max_decode_steps,

		temperature = temperature

		)
	# delete 
	generations = sequences[0]['generated_text'].split('\n')
	generations_del = [gen for gen in generations if len(gen)>0]
	return "\n".join(generations_del)