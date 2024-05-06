from peft import AutoPeftModelForCausalLM
import torch
import os
import argparse
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Script to run merge a (Q)LoRA adapter into the base model.")
    parser.add_argument('model_name', type=str, 
                        help='The name of the tuned adapter model that you pushed to Huggingface after finetuning or DPO.')
    parser.add_argument('output_name', type=str, 
                        help='The name of the output (merged) model. Can either be on Huggingface or on disk')
    parser.add_argument('--cpu', action='store_true',
                        help="Forces usage of CPU. By default GPU is taken if available.")

    args = parser.parse_args()
    model_name = args.model_name
    force_cpu = args.cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() and not (force_cpu) else "cpu")
    output_model = args.output_model
    
    # Load the model and merge with base
    model = AutoPeftModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if os.path.isdir(output_model):
        model.save_to_disk(output_model)
        tokenizer.save_to_disk(output_model)
    else:
        # Try to push to hub, requires HF_TOKEN environment variable to be set, see https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hftoken
        model.push_to_hub(output_model)
        tokenizer.push_to_hub(output_model)


if __name__ == "__main__":
    main()
