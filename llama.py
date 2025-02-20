import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
chat_model = AutoModelForCausalLM.from_pretrained(
    "pretrain/Llama-2-13b-chat-hf",
    # cache_dir="/data/yash/base_models",
    device_map='auto'
)

chat_tokenizer = AutoTokenizer.from_pretrained("pretrain/Llama-2-7b-chat-hf")
def get_llama2_chat_reponse(prompt, max_new_tokens=50):
    inputs = chat_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = chat_model.generate(**inputs, max_new_tokens=max_new_tokens, temperature= 0.00001)
    response = chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response 

prompt = "Q:what is the capital of India? A:"
response = get_llama2_chat_reponse(prompt, max_new_tokens=50)
print(response)
