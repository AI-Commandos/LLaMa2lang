import bitsandbytes as bnb
from functools import partial
from peft import AutoPeftModelForCausalLM
import torch
from transformers import AutoTokenizer
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Script to run inference on a tuned model.")
    parser.add_argument('model_name', type=str, 
                        help='The name of the tuned model that you pushed to Huggingface in the previous step.')
    parser.add_argument('instruction_prompt', type=str, 
                        help='An instruction message added to every prompt given to the chatbot to force it to answer in the target language.')
    parser.add_argument('input', type=str,
                        help='The actual chat input prompt. The script is only meant for testing purposes and exits after answering.')
    args = parser.parse_args()
    model_name = args.model_name
    instruction_prompt = args.instruction_prompt
    input = args.input

    model = AutoPeftModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    # Merge adapter with base
    model = model.merge_and_unload()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # Prepare input in LLaMa2 chat format
    input_text = f"<s>[INST] <<SYS>> {instruction_prompt} <</SYS>> {input} [/INST]"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Generate response and decode
    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        max_length=200,
        repetition_penalty=1.2 # LLaMa2 is sensitive to repetition
    )
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    print(generated_text)

if __name__ == "__main__":
    main()