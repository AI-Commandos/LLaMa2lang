import bitsandbytes as bnb
from functools import partial
from peft import AutoPeftModelForCausalLM
import torch
from transformers import AutoTokenizer
import sys

model_name = sys.argv[1]
system_instruction = sys.argv[2]
input = sys.argv[3]

model = AutoPeftModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
# Merge adapter with base
model = model.merge_and_unload()
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare input in LLaMa2 chat format
input_text = f"<s>[INST] <<SYS>> {system_instruction} <</SYS>> {input} [/INST]"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# Generate response and decode
output_sequences = model.generate(
    input_ids=inputs['input_ids'],
    max_length=200,
    repetition_penalty=1.2 # LLaMa2 is sensitive to repetition
)
generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print(generated_text)