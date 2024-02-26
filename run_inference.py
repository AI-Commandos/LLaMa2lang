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
                        help='The name of the tuned model that you pushed to Huggingface after finetuning or DPO.')
    parser.add_argument('instruction_prompt', type=str, 
                        help='An instruction message added to every prompt given to the chatbot to force it to answer in the target language.')
    parser.add_argument('--cpu', action='store_true',
                        help="Forces usage of CPU. By default GPU is taken if available.")
    parser.add_argument('--thread_template', type=str, default="threads/template_default.txt",
                        help='A file containing the thread template to use. Default is threads/template_fefault.txt')

    args = parser.parse_args()
    model_name = args.model_name
    instruction_prompt = args.instruction_prompt
    thread_template_file = args.thread_template
    force_cpu = args.cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() and not (force_cpu) else "cpu")

    # Get the template
    with open(thread_template_file, 'r', encoding="utf8") as f:
        chat_template = f.read()
    
    # Load the model and merge with base
    model = AutoPeftModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    thread = [
        {'role': 'system', 'content': instruction_prompt}
    ]
    while True:
        user_input = input("Enter your input, use ':n' for a new thread or ':q' to quit: ")
        if user_input.lower() == ':q':
            break
        elif user_input.lower() == ':n':
            thread = [{'role': 'system', 'content': instruction_prompt}]
            continue
        
        # Prepare input in LLaMa2 chat format
        thread.append({
            'role': 'user', 'content': user_input
        })
        input_chat = tokenizer.apply_chat_template(thread, tokenize=False, chat_template=chat_template)
        inputs = tokenizer(input_chat, return_tensors="pt").to(device)

        # Generate response and decode
        output_sequences = model.generate(
            input_ids=inputs['input_ids'],
            max_length=200,
            repetition_penalty=1.2 # LLaMa2 is sensitive to repetition
        )
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        print(generated_text)
        # Get the answer only
        answer = generated_text[(len(input_chat)-len(tokenizer.bos_token)+1):]
        thread.append({
            'role': 'assistant', 'content': answer
        })

if __name__ == "__main__":
    main()
