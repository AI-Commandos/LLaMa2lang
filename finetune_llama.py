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

def main():
    parser = argparse.ArgumentParser(description="Finetune a base model using QLoRA and PEFT")
    parser.add_argument('tuned_model', type=str,
                        help='The name of the resulting tuned model. This will be pushed to Huggingface. Ensure HF_TOKEN is set.')
    parser.add_argument('dataset_name', type=str,
                        help='The name of the dataset to use for fine-tuning. This should be the output of the combine_checkpoints script.')
    parser.add_argument('instruction_prompt', type=str, 
                        help='An instruction message added to every prompt given to the chatbot to force it to answer in the target language. Example: "You are a generic chatbot that always answers in English."')
    parser.add_argument('thread_format', type=str, choices=['llama2', 'chatml'],
                        help="The format of the threads to use. 'llama2' is the format used by Llama2. 'chatml' is the format used by ChatML.")
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
    
    args = parser.parse_args()
    base_model = args.base_model
    tuned_model = args.tuned_model
    dataset_name = args.dataset_name
    instruction_prompt = args.instruction_prompt
    output_location = args.output_location
    base_dataset_text_field = args.base_dataset_text_field
    base_dataset_rank_field = args.base_dataset_rank_field
    base_dataset_id_field = args.base_dataset_id_field
    base_dataset_parent_field = args.base_dataset_parent_field
    thread_format = args.thread_format

    # Check for HF_TOKEN because otherwise we run a long time before dying :)
    if 'HF_TOKEN' not in os.environ:
        print("Environment variable 'HF_TOKEN' is not set. Terminating.")
        sys.exit(1)
    
    # Load the base translated dataset
    if os.path.isdir(dataset_name):
        dataset = load_from_disk(dataset_name)
    else:
        dataset = load_dataset(dataset_name)
    
    # Compute the threads
    if (thread_format == 'llama2'):
        # TODO: Implement this
        threads = dataset['train']

    compute_dtype = getattr(torch, "float16")

    # Set up quantization config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map={"": 0}
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
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
    training_params = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=400,
        logging_steps=200,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
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
        train_dataset=dataset['train'],
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )

    # Use bigger precision in normalization layers
    #for name, module in trainer.model.named_modules():
        #if "norm" in name:
            #module = module.to(torch.float32)

    # Before starting training, free up memory
    torch.cuda.empty_cache()
    # Train and push model to HF (make sure to set HF_TOKEN in env variables)
    trainer.train()
    trainer.model.push_to_hub(tuned_model)
    trainer.tokenizer.push_to_hub(tuned_model)


if __name__ == "__main__":
    main()