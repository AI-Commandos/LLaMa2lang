import os
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import sys
import argparse
from threads import finetune_prompts
from datasets import Dataset
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Finetune a base instruct/chat model using (Q)LoRA and PEFT")
    parser.add_argument('tuned_model', type=str,
                        help='The name of the resulting tuned model.')
    parser.add_argument('dataset_name', type=str,
                        help='The name of the dataset to use for fine-tuning. This should be the output of the combine_checkpoints script.')
    parser.add_argument('instruction_prompt', type=str, 
                        help='An instruction message added to every prompt given to the chatbot to force it to answer in the target language. Example: "You are a generic chatbot that always answers in English."')
    parser.add_argument('--base_model', type=str, default="NousResearch/Llama-2-7b-chat-hf",
                        help='The base foundation model. Default is "NousResearch/Llama-2-7b-chat-hf".')
    parser.add_argument('--base_dataset_text_field', type=str, default="text",
                        help="The dataset's column name containing the actual text to translate. Defaults to text")
    parser.add_argument('--base_dataset_rank_field', type=str, default="rank",
                        help="The dataset's column name containing the rank of an answer given to a prompt. Defaults to rank")
    parser.add_argument('--base_dataset_id_field', type=str, default="message_id",
                        help="The dataset's column name containing the id of a text. Defaults to message_id")
    parser.add_argument('--base_dataset_parent_field', type=str, default="parent_id",
                        help="The dataset's column name containing the parent id of a text. Defaults to parent_id")
    parser.add_argument('--base_dataset_role_field', type=str, default="role",
                        help="The dataset's column name containing the role of the author of the text (eg. prompter, assistant). Defaults to role")
    parser.add_argument('--quant8', action='store_true',
                        help='Finetunes the model in 8 bits. Requires more memory than the default 4 bit.')
    parser.add_argument('--noquant', action='store_true',
                        help='Do not quantize the finetuning. Requires more memory than the default 4 bit and optional 8 bit.')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='The maximum sequence length to use in finetuning. Should most likely line up with your base model\'s default max_seq_length. Default is 512.')
    parser.add_argument('--num_train_epochs', type=int, default=2,
                        help='Number of epochs to use. 2 is default and has been shown to work well.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='The batch size to use in finetuning. Adjust to fit in your GPU vRAM. Default is 4')
    
    args = parser.parse_args()
    base_model = args.base_model
    tuned_model = args.tuned_model
    dataset_name = args.dataset_name
    instruction_prompt = args.instruction_prompt
    base_dataset_text_field = args.base_dataset_text_field
    base_dataset_rank_field = args.base_dataset_rank_field
    base_dataset_id_field = args.base_dataset_id_field
    base_dataset_parent_field = args.base_dataset_parent_field
    base_dataset_role_field = args.base_dataset_role_field
    quant8 = args.quant8
    noquant = args.noquant
    max_seq_length = args.max_seq_length
    num_train_epochs = args.num_train_epochs
    per_device_train_batch_size = args.batch_size

    # Check for HF_TOKEN
    if 'HF_TOKEN' not in os.environ:
        print("[WARNING] Environment variable 'HF_TOKEN' is not set!")
        user_input = input("Do you want to continue? (yes/no): ").strip().lower()

        if user_input != "yes":
            print("Terminating the program.")
            exit()
    
    # Load the base translated dataset
    if os.path.isdir(dataset_name):
        dataset = load_from_disk(dataset_name)
    else:
        dataset = load_dataset(dataset_name)
    
    # Load base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    # Compute the threads
    threads = dataset['train']
    prompts = finetune_prompts.create_prompts(threads, tokenizer, base_dataset_rank_field, base_dataset_parent_field, base_dataset_id_field, base_dataset_text_field, base_dataset_role_field, instruction_prompt)
    prompts = Dataset.from_pandas(pd.DataFrame(data=prompts))

    if noquant:
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(base_model, device_map={"": 0})
    elif quant8:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="qat8",
            bnb_8bit_compute_dtype=getattr(torch, "float32"),
            bnb_8bit_use_double_quant=False
        )
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quant_config, device_map={"": 0})
    else:
        # Set up quantization config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, "float16"),
            bnb_4bit_use_double_quant=True,
        )
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quant_config, device_map={"": 0})

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Just like Alpaca, because we allow to add history in the prompts, it makes more sense to do left-padding to have the most informative text at the end.
    # In this case, we need a different pad token than EOS because we actually do _not_ pad end of sentence.
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # Set up LoRA configuration
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Pass quant and lora to trainer
    use_fp16 = not(noquant or quant8)
    training_params = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=400,
        logging_steps=200,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=use_fp16,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=prompts,
        peft_config=peft_params,
        dataset_text_field=base_dataset_text_field,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )

    # Before starting training, free up memory
    torch.cuda.empty_cache()
    # Train the model
    trainer.train()

    # Check if output location is a valid directory
    if os.path.isdir(tuned_model):
        trainer.model.save_to_disk(tuned_model)
        trainer.tokenizer.save_to_disk(tuned_model)
    else:
        # Try to push to hub, requires HF_TOKEN environment variable to be set, see https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hftoken
        trainer.model.push_to_hub(tuned_model)
        trainer.tokenizer.push_to_hub(tuned_model)

if __name__ == "__main__":
    main()