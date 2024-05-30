# 🚀 Now with LLaMa3 support 🚀


# LLaMa2lang v0.6
This repository contains convenience scripts to finetune LLaMa3-8B (or any other foundation model) for chat towards any language (that isn't English). The rationale behind this is that LLaMa3 is trained on primarily English data and while it works to some extent for other languages, its performance is poor compared to English.

# TL;DR

```
pip install -r requirements.txt

# Translate OASST1 to target language
python translate.py m2m target_lang checkpoint_location

# Combine the checkpoint files into a dataset
python combine_checkpoints.py input_folder output_location

# Finetune
python finetune.py tuned_model dataset_name instruction_prompt

# Optionally finetune with DPO (RLHF)
python finetune_dpo.py tuned_model dataset_name instruction_prompt

# Run inference
python run_inference.py model_name instruction_prompt input
```

# What it does
The process we follow to tune a foundation model such as LLaMa3 for a specific language is as follows:

1. Load a dataset that contains Q&A/instruction pairs.
2. Translate the entire dataset to a given target language.
3. Load the translated dataset and extract threads by recursively selecting prompts with their respective answers with the highest rank only, through to subsequent prompts, etc.
4. Turn the threads into prompts following a given template (customizable).
5. Use QLoRA and PEFT to finetune a base foundation model's instruct finetune on this dataset.
6. * Use QLoRA and PEFT to finetune with [DPO](https://huggingface.co/docs/trl/main/en/dpo_trainer) to extend the model's capacities even further and teach it preferred answers over rejected ones. Note that your base dataset must have this information.
   * Alternatively to DPO, you can achieve the same with [ORPO](https://huggingface.co/docs/trl/main/en/orpo_trainer)
7. Run inference using the newly trained model.

# Supported paradigms
## Translation
* OPUS
* M2M
* MADLAD
* mBART
* NLLB
* Seamless (Large only)
* Tower Instruct (Can correct spelling mistakes)
## Base datasets
The following have been tested but potentially more will work
* OASST1
* OASST2
## Supported foundation models
* **LLaMa3**
* LLaMa2
* Mistral
* (Unofficial) Mixtral 8x7B

# Roadmap
* [L2L-6] Investigate interoperability with other libraries (Axolotl, llamacpp, unsloth)
* [L2L-7] Allow for different quantizations next to QLoRA (GGUF, GPTQ, AWQ)
* [L2L-10] Support extending the tokenizer and vocabulary

## Cost and runtime

The above process can be fully run on a free Google Colab T4 GPU. The last step however, can only be successfully run with short enough context windows and a batch of at most 2. In addition, the translation in step 2 takes about 36 hours in total for any given language so should be run in multiple steps if you want to stick with a free Google Colab GPU.

Our fine-tuned models for step 5 were performed using an A40 on [vast.ai](https://vast.ai/) and cost us less than a dollar for each model, completing in about 1.5 hours.

# Usage
1. Make sure pytorch is installed and working for your environment (use of CUDA preferable): https://pytorch.org/get-started/locally/

2. Clone the repo and install the requirements.

`pip install -r requirements.txt`

2. Translate your base dataset to your designated target language.

```
usage: translate.py [-h] [--quant8] [--quant4] [--base_dataset BASE_DATASET] [--base_dataset_text_field BASE_DATASET_TEXT_FIELD] [--base_dataset_lang_field BASE_DATASET_LANG_FIELD]
                    [--checkpoint_n CHECKPOINT_N] [--batch_size BATCH_SIZE] [--max_length MAX_LENGTH] [--cpu] [--source_lang SOURCE_LANG]
                    {opus,mbart,madlad,m2m,nllb,seamless_m4t_v2,towerinstruct} ... target_lang checkpoint_location

Translate an instruct/RLHF dataset to a given target language using a variety of translation models

positional arguments:
  {opus,mbart,madlad,m2m,nllb,seamless_m4t_v2,towerinstruct}
                        The model/architecture used for translation.
    opus                Translate the dataset using HelsinkiNLP OPUS models.
    mbart               Translate the dataset using mBART.
    madlad              Translate the dataset using Google's MADLAD models.
    m2m                 Translate the dataset using Facebook's M2M models.
    nllb                Translate the dataset using Facebook's NLLB models.
    seamless_m4t_v2     Translate the dataset using Facebook's SeamlessM4T-v2 multimodal models.
    towerinstruct       Translate the dataset using Unbabel's Tower Instruct. Make sure your target language is in the 10 languages supported by the model.
  target_lang           The target language. Make sure you use language codes defined by the translation model you are using.
  checkpoint_location   The folder the script will write (JSONized) checkpoint files to. Folder will be created if it doesn't exist.

options:
  -h, --help            show this help message and exit
  --quant8              Optional flag to load the translation model in 8 bits. Decreases memory usage, increases running time
  --quant4              Optional flag to load the translation model in 4 bits. Decreases memory usage, increases running time
  --base_dataset BASE_DATASET
                        The base dataset to translate, defaults to OpenAssistant/oasst1
  --base_dataset_text_field BASE_DATASET_TEXT_FIELD
                        The base dataset's column name containing the actual text to translate. Defaults to text
  --base_dataset_lang_field BASE_DATASET_LANG_FIELD
                        The base dataset's column name containing the language the source text was written in. Defaults to lang
  --checkpoint_n CHECKPOINT_N
                        An integer representing how often a checkpoint file will be written out. To start off, 400 is a reasonable number.
  --batch_size BATCH_SIZE
                        The batch size for a single translation model. Adjust based on your GPU capacity. Default is 10.
  --max_length MAX_LENGTH
                        How much tokens to generate at most. More tokens might be more accurate for lengthy input but creates a risk of running out of memory. Default is unlimited.
  --cpu                 Forces usage of CPU. By default GPU is taken if available.
  --source_lang SOURCE_LANG
                        Source language to select from OASST based on lang property of dataset
```

If you want more parameters for the different translation models, run:
```
python translate.py [MODEL] -h
```

Be sure to specify model-specific parameters first before you specify common parameters from the list above. Example calls:
```
# Using M2M with 4bit quantization and differen batch sizes to translate Dutch
python translate.py m2m nl ./output_nl --quant4 --batch_size 20

# Using madlad 7B with 8bit quantization for German with different max_length
python translate.py madlad --model_size 7b de ./output_de --quant8 --batch_size 5 --max_length 512

# Be sure to use target language codes that the model you use understands
python translate.py mbart xh_ZA ./output_xhosa
python translate.py nllb nld_Latn ./output_nl
```

3. Combine the JSON arrays from the checkpoints' files into a Huggingface Dataset and then either write it to disk or publish it to Huggingface. The script will try to write to disk by default and fall back to publishing to Huggingface if the folder doesn't exist on disk. For publishing to Huggingface, make sure you have your `HF_TOKEN` environment variable set up as per [the documentation](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hftoken).

```
usage: combine_checkpoints.py [-h] input_folder output_location

Combine checkpoint files from translation.

positional arguments:
  input_folder     The checkpoint folder used in translation, with the target language appended.
                   Example: "./output_nl".
  output_location  Where to write the Huggingface Dataset. Can be a disk location or a Huggingface
                   Dataset repository.

options:
  -h, --help       show this help message and exit
```

5. Turn the translated messages into chat/instruct/prompt threads and finetune a foundate model's instruct using LoRA and PEFT.

```
usage: finetune.py [-h] [--base_model BASE_MODEL] [--base_dataset_text_field BASE_DATASET_TEXT_FIELD] [--base_dataset_rank_field BASE_DATASET_RANK_FIELD] [--base_dataset_id_field BASE_DATASET_ID_FIELD] [--base_dataset_parent_field BASE_DATASET_PARENT_FIELD]
                   [--base_dataset_role_field BASE_DATASET_ROLE_FIELD] [--quant8] [--noquant] [--max_seq_length MAX_SEQ_LENGTH] [--num_train_epochs NUM_TRAIN_EPOCHS] [--batch_size BATCH_SIZE] [--threads_output_name THREADS_OUTPUT_NAME] [--thread_template THREAD_TEMPLATE]
                   [--padding PADDING]
                   tuned_model dataset_name instruction_prompt

Finetune a base instruct/chat model using (Q)LoRA and PEFT

positional arguments:
  tuned_model           The name of the resulting tuned model.
  dataset_name          The name of the dataset to use for fine-tuning. This should be the output of the combine_checkpoints script.
  instruction_prompt    An instruction message added to every prompt given to the chatbot to force it to answer in the target language. Example: "You are a generic chatbot that always answers in English."

options:
  -h, --help            show this help message and exit
  --base_model BASE_MODEL
                        The base foundation model. Default is "NousResearch/Meta-Llama-3-8B-Instruct".
  --base_dataset_text_field BASE_DATASET_TEXT_FIELD
                        The dataset's column name containing the actual text to translate. Defaults to text
  --base_dataset_rank_field BASE_DATASET_RANK_FIELD
                        The dataset's column name containing the rank of an answer given to a prompt. Defaults to rank
  --base_dataset_id_field BASE_DATASET_ID_FIELD
                        The dataset's column name containing the id of a text. Defaults to message_id
  --base_dataset_parent_field BASE_DATASET_PARENT_FIELD
                        The dataset's column name containing the parent id of a text. Defaults to parent_id
  --base_dataset_role_field BASE_DATASET_ROLE_FIELD
                        The dataset's column name containing the role of the author of the text (eg. prompter, assistant). Defaults to role
  --quant8              Finetunes the model in 8 bits. Requires more memory than the default 4 bit.
  --noquant             Do not quantize the finetuning. Requires more memory than the default 4 bit and optional 8 bit.
  --max_seq_length MAX_SEQ_LENGTH
                        The maximum sequence length to use in finetuning. Should most likely line up with your base model's default max_seq_length. Default is 512.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Number of epochs to use. 2 is default and has been shown to work well.
  --batch_size BATCH_SIZE
                        The batch size to use in finetuning. Adjust to fit in your GPU vRAM. Default is 4
  --threads_output_name THREADS_OUTPUT_NAME
                        If specified, the threads created in this script for finetuning will also be saved to disk or HuggingFace Hub.
  --thread_template THREAD_TEMPLATE
                        A file containing the thread template to use. Default is threads/template_fefault.txt
  --padding PADDING     What padding to use, can be either left or right.
```

6.1 [OPTIONAL] Finetune using DPO (similar to RLHF)
```
usage: finetune_dpo.py [-h] [--base_model BASE_MODEL] [--base_dataset_text_field BASE_DATASET_TEXT_FIELD] [--base_dataset_rank_field BASE_DATASET_RANK_FIELD] [--base_dataset_id_field BASE_DATASET_ID_FIELD] [--base_dataset_parent_field BASE_DATASET_PARENT_FIELD] [--quant8]
                       [--noquant] [--max_seq_length MAX_SEQ_LENGTH] [--max_prompt_length MAX_PROMPT_LENGTH] [--num_train_epochs NUM_TRAIN_EPOCHS] [--batch_size BATCH_SIZE] [--threads_output_name THREADS_OUTPUT_NAME] [--thread_template THREAD_TEMPLATE] [--max_steps MAX_STEPS]
                       [--padding PADDING]
                       tuned_model dataset_name instruction_prompt

Finetune a base instruct/chat model using (Q)LoRA and PEFT using DPO (RLHF)

positional arguments:
  tuned_model           The name of the resulting tuned model.
  dataset_name          The name of the dataset to use for fine-tuning. This should be the output of the combine_checkpoints script.
  instruction_prompt    An instruction message added to every prompt given to the chatbot to force it to answer in the target language. Example: "You are a generic chatbot that always answers in English."

options:
  -h, --help            show this help message and exit
  --base_model BASE_MODEL
                        The base foundation model. Default is "NousResearch/Meta-Llama-3-8B-Instruct".
  --base_dataset_text_field BASE_DATASET_TEXT_FIELD
                        The dataset's column name containing the actual text to translate. Defaults to text
  --base_dataset_rank_field BASE_DATASET_RANK_FIELD
                        The dataset's column name containing the rank of an answer given to a prompt. Defaults to rank
  --base_dataset_id_field BASE_DATASET_ID_FIELD
                        The dataset's column name containing the id of a text. Defaults to message_id
  --base_dataset_parent_field BASE_DATASET_PARENT_FIELD
                        The dataset's column name containing the parent id of a text. Defaults to parent_id
  --quant8              Finetunes the model in 8 bits. Requires more memory than the default 4 bit.
  --noquant             Do not quantize the finetuning. Requires more memory than the default 4 bit and optional 8 bit.
  --max_seq_length MAX_SEQ_LENGTH
                        The maximum sequence length to use in finetuning. Should most likely line up with your base model's default max_seq_length. Default is 512.
  --max_prompt_length MAX_PROMPT_LENGTH
                        The maximum length of the prompts to use. Default is 512.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Number of epochs to use. 2 is default and has been shown to work well.
  --batch_size BATCH_SIZE
                        The batch size to use in finetuning. Adjust to fit in your GPU vRAM. Default is 4
  --threads_output_name THREADS_OUTPUT_NAME
                        If specified, the threads created in this script for finetuning will also be saved to disk or HuggingFace Hub.
  --thread_template THREAD_TEMPLATE
                        A file containing the thread template to use. Default is threads/template_fefault.txt
  --max_steps MAX_STEPS
                        The maximum number of steps to run DPO for. Default is -1 which will run the data through fully for the number of epochs but this will be very time-consuming.
  --padding PADDING     What padding to use, can be either left or right.
```

6.1 [OPTIONAL] Finetune using ORPO (similar to RLHF)
```
usage: finetune_orpo.py [-h] [--base_model BASE_MODEL] [--base_dataset_text_field BASE_DATASET_TEXT_FIELD] [--base_dataset_rank_field BASE_DATASET_RANK_FIELD] [--base_dataset_id_field BASE_DATASET_ID_FIELD] [--base_dataset_parent_field BASE_DATASET_PARENT_FIELD] [--quant8]
                        [--noquant] [--max_seq_length MAX_SEQ_LENGTH] [--max_prompt_length MAX_PROMPT_LENGTH] [--num_train_epochs NUM_TRAIN_EPOCHS] [--batch_size BATCH_SIZE] [--threads_output_name THREADS_OUTPUT_NAME] [--thread_template THREAD_TEMPLATE] [--max_steps MAX_STEPS]
                        [--padding PADDING]
                        tuned_model dataset_name instruction_prompt

Finetune a base instruct/chat model using (Q)LoRA and PEFT using ORPO (RLHF)

positional arguments:
  tuned_model           The name of the resulting tuned model.
  dataset_name          The name of the dataset to use for fine-tuning. This should be the output of the combine_checkpoints script.
  instruction_prompt    An instruction message added to every prompt given to the chatbot to force it to answer in the target language. Example: "You are a generic chatbot that always answers in English."

options:
  -h, --help            show this help message and exit
  --base_model BASE_MODEL
                        The base foundation model. Default is "NousResearch/Meta-Llama-3-8B-Instruct".
  --base_dataset_text_field BASE_DATASET_TEXT_FIELD
                        The dataset's column name containing the actual text to translate. Defaults to text
  --base_dataset_rank_field BASE_DATASET_RANK_FIELD
                        The dataset's column name containing the rank of an answer given to a prompt. Defaults to rank
  --base_dataset_id_field BASE_DATASET_ID_FIELD
                        The dataset's column name containing the id of a text. Defaults to message_id
  --base_dataset_parent_field BASE_DATASET_PARENT_FIELD
                        The dataset's column name containing the parent id of a text. Defaults to parent_id
  --quant8              Finetunes the model in 8 bits. Requires more memory than the default 4 bit.
  --noquant             Do not quantize the finetuning. Requires more memory than the default 4 bit and optional 8 bit.
  --max_seq_length MAX_SEQ_LENGTH
                        The maximum sequence length to use in finetuning. Should most likely line up with your base model's default max_seq_length. Default is 512.
  --max_prompt_length MAX_PROMPT_LENGTH
                        The maximum length of the prompts to use. Default is 512.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Number of epochs to use. 2 is default and has been shown to work well.
  --batch_size BATCH_SIZE
                        The batch size to use in finetuning. Adjust to fit in your GPU vRAM. Default is 4
  --threads_output_name THREADS_OUTPUT_NAME
                        If specified, the threads created in this script for finetuning will also be saved to disk or HuggingFace Hub.
  --thread_template THREAD_TEMPLATE
                        A file containing the thread template to use. Default is threads/template_fefault.txt
  --max_steps MAX_STEPS
                        The maximum number of steps to run ORPO for. Default is -1 which will run the data through fully for the number of epochs but this will be very time-consuming.
  --padding PADDING     What padding to use, can be either left or right.
```

7. Run inference using the newly created QLoRA model.

```
usage: run_inference.py [-h] model_name instruction_prompt input

Script to run inference on a tuned model.

positional arguments:
  model_name          The name of the tuned model that you pushed to Huggingface in the previous
                      step.
  instruction_prompt  An instruction message added to every prompt given to the chatbot to force
                      it to answer in the target language.
  input               The actual chat input prompt. The script is only meant for testing purposes
                      and exits after answering.

options:
  -h, --help          show this help message and exit

```

# Choosing the right translation model
> How do I know which translation model to choose for my target language?

**We got you covered** with out `benchmark.py` script that helps make somewhat of a good guess (the dataset we use is the same as the OPUS models are trained on so the outcomes are always favorable towards OPUS). For usage, see the help of this script below. Models are loaded in 4-bit quantization and run on a small sample of the OPUS books subset.

Be sure to use the most commonly occurring languages in your base dataset as source_language and your target translation language as target_language. For OASST1 for example, be sure to at least run `en` and `es` as source languages.

```
usage: benchmark.py [-h] [--cpu] [--start START] [--n N] [--max_length MAX_LENGTH] source_language target_language included_models

Benchmark all the different translation models for a specific source and target language to find out which performs best. This uses 4bit quantization to limit GPU usage. Note:
the outcomes are indicative - you cannot assume corretness of the BLEU and CHRF scores but you can compare models against each other relatively.

positional arguments:
  source_language       The source language you want to test for. Check your dataset to see which occur most prevalent or use English as a good start.
  target_language       The source language you want to test for. This should be the language you want to apply the translate script on. Note: in benchmark, we use 2-character
                        language codes, in constrast to translate.py where you need to specify whatever your model expects.
  included_models       Comma-separated list of models to include. Allowed values are: opus, m2m_418m, m2m_1.2b, madlad_3b, madlad_7b, madlad_10b, madlad_7bbt, mbart,
                        nllb_distilled600m, nllb_1.3b, nllb_distilled1.3b, nllb_3.3b, seamless

options:
  -h, --help            show this help message and exit
  --cpu                 Forces usage of CPU. By default GPU is taken if available.
  --start START         The starting offset to include sentences from the OPUS books dataset from. Defaults to 0.
  --n N                 The number of sentences to benchmark on. Defaults to 100.
  --max_length MAX_LENGTH
                        How much tokens to generate at most. More tokens might be more accurate for lengthy input but creates a risk of running out of memory. Default is 512.
```

# Datasets and models

We have created and will continue to create numerous datasets and models already. **Want to help democratize LLMs?** Clone the repo and create datasets and models for other languages, then create a PR.

## Translated oasst1 datasets

|  |  |  |  |
|---------|---------|---------|---------|
| Dutch [UnderstandLing/oasst1_nl](https://huggingface.co/datasets/UnderstandLing/oasst1_nl) | Spanish [UnderstandLing/oasst1_es](https://huggingface.co/datasets/UnderstandLing/oasst1_es) | French [UnderstandLing/oasst1_fr](https://huggingface.co/datasets/UnderstandLing/oasst1_fr) | German [UnderstandLing/oasst1_de](https://huggingface.co/datasets/UnderstandLing/oasst1_de) |
| Catalan [xaviviro/oasst1_ca](https://huggingface.co/datasets/xaviviro/oasst1_ca) | Portuguese [UnderstandLing/oasst1_pt](https://huggingface.co/datasets/UnderstandLing/oasst1_pt) | Arabic [HeshamHaroon/oasst-arabic](https://huggingface.co/datasets/HeshamHaroon/oasst-arabic) | Italian [UnderstandLing/oasst1_it](https://huggingface.co/datasets/UnderstandLing/oasst1_it) |
| Russian [UnderstandLing/oasst1_ru](https://huggingface.co/datasets/UnderstandLing/oasst1_ru) | Hindi [UnderstandLing/oasst1_hi](https://huggingface.co/datasets/UnderstandLing/oasst1_hi) | Chinese [UnderstandLing/oasst1_zh](https://huggingface.co/datasets/UnderstandLing/oasst1_zh) | Polish [chrystians/oasst1_pl](https://huggingface.co/datasets/chrystians/oasst1_pl) |
| Japanese [UnderstandLing/oasst1_jap](https://huggingface.co/datasets/UnderstandLing/oasst1_jap) | Basque [xezpeleta/oasst1_eu](https://huggingface.co/datasets/xezpeleta/oasst1_eu) | Bengali [UnderstandLing/oasst1_bn](https://huggingface.co/datasets/UnderstandLing/oasst1_bn) | Turkish [UnderstandLing/oasst1_tr](https://huggingface.co/datasets/UnderstandLing/oasst1_tr) |

## Language-specific ❗LLaMa3-8B❗ chat model adapters

Make sure you have access to Meta's [LLaMa3-8B model](https://huggingface.co/meta-llama/Meta-Llama-3-8B) and set your HF_TOKEN before using these models.

|  |  |  |  |
|---------|---------|---------|---------|
| [UnderstandLing/Llama-3-8B-Instruct-nl](https://huggingface.co/UnderstandLing/Llama-3-8B-Instruct-nl) Dutch | [UnderstandLing/Llama-3-8B-Instruct-es](https://huggingface.co/UnderstandLing/Llama-3-8B-Instruct-es) Spanish | [UnderstandLing/Llama-3-8B-Instruct-fr](https://huggingface.co/UnderstandLing/Llama-3-8B-Instruct-fr) French | [UnderstandLing/Llama-3-8B-Instruct-de](https://huggingface.co/UnderstandLing/Llama-3-8B-Instruct-de) German |
| [UnderstandLing/Llama-3-8B-Instruct-pt](https://huggingface.co/UnderstandLing/Llama-3-8B-Instruct-pt) Portuguese | [UnderstandLing/Llama-3-8B-Instruct-it](https://huggingface.co/UnderstandLing/Llama-3-8B-Instruct-it) Italian | [UnderstandLing/Llama-3-8B-Instruct-hi](https://huggingface.co/UnderstandLing/Llama-3-8B-Instruct-hi) Hindi | [UnderstandLing/Llama-3-8B-Instruct-ru](https://huggingface.co/UnderstandLing/Llama-3-8B-Instruct-ru) Russian |


## Translated LLaMa2 thread chat prompt datasets

|  |  |  |  |
|---------|---------|---------|---------|
| Dutch [UnderstandLing/oasst1_nl_threads](https://huggingface.co/datasets/UnderstandLing/oasst1_nl_threads) | Spanish [UnderstandLing/oasst1_es_threads](https://huggingface.co/datasets/UnderstandLing/oasst1_es_threads) | French [UnderstandLing/oasst1_fr_threads](https://huggingface.co/datasets/UnderstandLing/oasst1_fr_threads) | German [UnderstandLing/oasst1_de_threads](https://huggingface.co/datasets/UnderstandLing/oasst1_de_threads) |
| Catalan [xaviviro/oasst1_ca_threads](https://huggingface.co/datasets/xaviviro/oasst1_ca_threads) | Portuguese [UnderstandLing/oasst1_pt_threads](https://huggingface.co/datasets/UnderstandLing/oasst1_pt_threads) | Arabic [HeshamHaroon/oasst-arabic_threads](https://huggingface.co/datasets/HeshamHaroon/oasst-arabic_threads) | Italian [UnderstandLing/oasst1_it_threads](https://huggingface.co/datasets/UnderstandLing/oasst1_it_threads) |
| Russian [UnderstandLing/oasst1_ru_threads](https://huggingface.co/datasets/UnderstandLing/oasst1_ru_threads) | Hindi [UnderstandLing/oasst1_hi_threads](https://huggingface.co/datasets/UnderstandLing/oasst1_hi_threads) | Chinese [UnderstandLing/oasst1_zh_threads](https://huggingface.co/datasets/UnderstandLing/oasst1_zh_threads) | Polish [chrystians/oasst1_pl_threads](https://huggingface.co/datasets/chrystians/oasst1_pl_threads) |
| Japanese [UnderstandLing/oasst1_jap_threads](https://huggingface.co/datasets/UnderstandLing/oasst1_jap_threads) | Basque [xezpeleta/oasst1_eu_threads](https://huggingface.co/datasets/xezpeleta/oasst1_eu_threads) | Bengali [UnderstandLing/oasst1_bn_threads](https://huggingface.co/datasets/UnderstandLing/oasst1_bn_threads) | Turkish [UnderstandLing/oasst1_tr_threads](https://huggingface.co/datasets/UnderstandLing/oasst1_tr_threads) |

## Language-specific LLaMa2-7B chat model adapters
|  |  |  |  |
|---------|---------|---------|---------|
| [UnderstandLing/llama-2-7b-chat-nl](https://huggingface.co/UnderstandLing/llama-2-7b-chat-nl) Dutch | [UnderstandLing/llama-2-7b-chat-es](https://huggingface.co/UnderstandLing/llama-2-7b-chat-es) Spanish | [UnderstandLing/llama-2-7b-chat-fr](https://huggingface.co/UnderstandLing/llama-2-7b-chat-fr) French |[UnderstandLing/llama-2-7b-chat-de](https://huggingface.co/UnderstandLing/llama-2-7b-chat-de) German |
[xaviviro/llama-2-7b-chat-ca](https://huggingface.co/xaviviro/llama-2-7b-chat-ca) Catalan | [UnderstandLing/llama-2-7b-chat-pt](https://huggingface.co/UnderstandLing/llama-2-7b-chat-pt) Portuguese | [HeshamHaroon/llama-2-7b-chat-ar](https://huggingface.co/HeshamHaroon/llama-2-7b-chat-ar) Arabic | [UnderstandLing/llama-2-7b-chat-it](https://huggingface.co/UnderstandLing/llama-2-7b-chat-it) Italian |
[UnderstandLing/llama-2-7b-chat-ru](https://huggingface.co/UnderstandLing/llama-2-7b-chat-ru) Russian | [UnderstandLing/llama-2-7b-chat-hi](https://huggingface.co/UnderstandLing/llama-2-7b-chat-hi) Hindi | [UnderstandLing/llama-2-7b-chat-zh](https://huggingface.co/UnderstandLing/llama-2-7b-chat-zh) Chinese | [chrystians/llama-2-7b-chat-pl-polish-polski](https://huggingface.co/chrystians/llama-2-7b-chat-pl-polish-polski) Polish |
| [xezpeleta/llama-2-7b-chat-eu](https://huggingface.co/xezpeleta/llama-2-7b-chat-eu) Basque | [UnderstandLing/llama-2-7b-chat-bn](https://huggingface.co/UnderstandLing/llama-2-7b-chat-bn) Bengali | [UnderstandLing/llama-2-7b-chat-tr](https://huggingface.co/UnderstandLing/llama-2-7b-chat-tr) Turkish | |

## Language-specific Mistral chat model adapters
|  |  |  |  |
|---------|---------|---------|---------|
| [UnderstandLing/Mistral-7B-Instruct-v0.2-nl](https://huggingface.co/UnderstandLing/Mistral-7B-Instruct-v0.2-nl) Dutch | [UnderstandLing/Mistral-7B-Instruct-v0.2-es](https://huggingface.co/UnderstandLing/Mistral-7B-Instruct-v0.2-es) Spanish | [UnderstandLing/Mistral-7B-Instruct-v0.2-de](https://huggingface.co/UnderstandLing/Mistral-7B-Instruct-v0.2-de) German | |

## Language-specific LLaMa2-13B chat model adapters
|  |  |  |  |
|---------|---------|---------|---------|
| [UnderstandLing/llama-2-13b-chat-nl](https://huggingface.co/UnderstandLing/llama-2-13b-chat-nl) Dutch | [UnderstandLing/llama-2-13b-chat-es](https://huggingface.co/UnderstandLing/llama-2-13b-chat-es) Spanish | [UnderstandLing/llama-2-13b-chat-fr](https://huggingface.co/UnderstandLing/llama-2-13b-chat-fr) French | |

## Language-specific Mixtral-8x7B chat model adapters
|  |  |  |  |
|---------|---------|---------|---------|
| [UnderstandLing/Mixtral-8x7B-Instruct-nl](https://huggingface.co/UnderstandLing/Mixtral-8x7B-Instruct-nl) Dutch | | | |

# Empirical performance

## Dutch

`<s>[INST] <<SYS>> Je bent een generieke chatbot die altijd in het Nederlands antwoord geeft. <</SYS>> Wat is de hoofdstad van Nederland? [/INST] Amsterdam</s>`

`<s>[INST] <<SYS>> Je bent een generieke chatbot die altijd in het Nederlands antwoord geeft. <</SYS>> Wat is de hoofdstad van Nederland? [/INST] Amsterdam</s><s>[INST] Hoeveel inwoners heeft die stad? [/INST] 850 duizend inwoners (2023)</s>`

`<s>[INST] <<SYS>> Je bent een generieke chatbot die altijd in het Nederlands antwoord geeft. <</SYS>> Wat is de hoofdstad van Nederland? [/INST] Amsterdam</s><s>[INST] Hoeveel inwoners heeft die stad? [/INST] 850 duizend inwoners (2023)</s><s>[INST] In welke provincie ligt die stad? [/INST] In de provincie Noord-Holland</s>`

`<s>[INST] <<SYS>> Je bent een generieke chatbot die altijd in het Nederlands antwoord geeft. <</SYS>> Wie is de minister-president van Nederland? [/INST] Mark Rutte is sinds 2010 minister-president van Nederland. Hij is meerdere keren herkozen.</s>`

# FAQ

* Q: Why do you translate the full OASST1/2 dataset first? Wouldn't it be faster to only translate highest ranked threads?
* A: While you can gain quite a lot in terms of throughput time by first creating the threads and then translating them, we provide full OASST1/2 translations to the community as we believe they can be useful on their own.

* Q: How well do the fine-tunes perform compared to vanilla LLaMa3?
* A: While we do not have formal benchmarks, getting LLaMa3 to consistently speak another language than English to begin with is challenging if not impossible. The non-English language it does produce is often grammatically broken. Our fine-tunes do not show this behavior.

* Q: Can I use other frameworks for fine-tuning?
* A: Yes you can, we use [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) for training on multi-GPU setups.

* Q: Can I mix different translation models?
* A: Absolutely, we think it might even increase performance to have translation done by multiple models. You can achieve this by early-stopping a translation and continuing from the checkpoints by reruning the translate script with a different translation model.

# Funding
We are actively looking for funding to democratize AI and advance its applications. Contact us at funding@understandling.com if you want to invest.
