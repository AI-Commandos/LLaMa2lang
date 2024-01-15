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
                        help='The name of the dataset to use for fine-tuning.')
    parser.add_argument('--base_model', type=str, default="NousResearch/Llama-2-7b-chat-hf",
                        help='The base foundation model. Default is "NousResearch/Llama-2-7b-chat-hf".')
    args = parser.parse_args()
    base_model = args.base_model
    tuned_model = args.tuned_model
    dataset_name = args.dataset_name

    # Check for HF_TOKEN because otherwise we run a long time before dying :)
    if 'HF_TOKEN' not in os.environ:
        print("Environment variable 'HF_TOKEN' is not set. Terminating.")
        sys.exit(1)
    
    if os.path.isdir(dataset_name):
        dataset = load_from_disk(dataset_name)
    else:
        dataset = load_dataset(dataset_name)

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