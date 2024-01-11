# LLaMa2lang v0.3
This repository contains convenience scripts to finetune LLaMa2-7b for chat towards any language (that isn't English). The rationale behind this is that LLaMa2 is trained on primarily English data and while it works to some extent for other languages, its performance is poor compared to English.

# Change info
_v0.3_
* **[2024-01-09]** We have significantly refactored the translation process. Please follow the readme carefully if you come from v0.2.
* **[2024-01-09]** We now support translation through M2M.
* **[2024-01-04]** We now support translation through MADLAD. Especially for models where Helsinki has a low BLEU score (less than 40), MADLAD (or the faster M2M) is preferred. Using MADLAD drastically slows down training time, especially if you quantize (4 bit is even slower than 8 bit).
* **[2024-01-04]** We now use argparser to parse command line arguments. Make sure you update your calls to our scripts accordingly. Use `-h` on all scripts to get help.

_v0.2_
* **[2023-12-29]** We now batch translations in `translate.py` for a 30-60% speed increase. If you have checkpoints from before this date, you can **not** continue using the main branch but instead must use the [v0.1 branch](https://github.com/UnderstandLingBV/LLaMa2lang/tree/v0.1).

# TL;DR

```
pip install -r requirements.txt

# Translate OASST1 to target language
python translate.py m2m target_lang checkpoint_location

# Combine the checkpoint files into a dataset
python combine_checkpoints.py input_folder output_location

# Create threaded prompts
python create_thread_prompts.py dataset_name instruction_prompt output_location

# Finetune
python finetune_llama.py tuned_model dataset_name

# Run inference
python run_inference.py model_name instruction_prompt input
```

# What it does
The process we follow to tune a foundation model such as LLaMa2 for a specific language is as follows:

1. Load a dataset that contains Q&A/instruction pairs.
2. Translate the entire dataset to a given target language.
3. Load the translated dataset and extract threads by recursively selecting prompts with their respective answers with the highest rank only, through to subsequent prompts, etc.
4. Turn the threads into texts using [LLaMa's prompt format](https://huggingface.co/blog/llama2#how-to-prompt-llama-2).
5. Use QLoRA and PEFT to finetune a base foundation model's instruct finetune on this dataset.

# Supported paradigms
## Translation
* OPUS
* M2M
* MADLAD
* mBART
## Base datasets
The following have been tested but potentially more will work
* OASST1
* OASST2
## Supported foundation models
* LLaMa2
* Mistral
* (Unofficial) Mixtral 8x7B

# Roadmap
* [L2L-4] Add DPO training as RLHF alternative
* [L2L-6] Investigate interoperability with other libraries (Axolotl, llamacpp, unsloth)
* [L2L-7] Allow for different quantizations next to QLoRA (GGUF, GPTQ, AWQ)

## Cost and runtime

The above process can be fully run on a free Google Colab T4 GPU. The last step however, can only be successfully run with short enough context windows and a batch of at most 2. In addition, the translation in step 2 takes about 36 hours in total for any given language so should be run in multiple steps if you want to stick with a free Google Colab GPU.

Our fine-tuned models for step 5 were performed using an A40 on [vast.ai](https://vast.ai/) and cost us less than a dollar for each model, completing in about 1.5 hours.

# Usage
1. Make sure pytorch is installed and working for your environment (use of CUDA preferable): https://pytorch.org/get-started/locally/

2. Clone the repo and install the requirements.

`pip install -r requirements.txt`

2. Translate your base dataset to your designated target language.

```
usage: translate.py [-h] [--quant8] [--quant4] [--base_dataset BASE_DATASET] [--base_dataset_text_field BASE_DATASET_TEXT_FIELD] [--base_dataset_lang_field BASE_DATASET_LANG_FIELD] [--checkpoint_n CHECKPOINT_N] [--batch_size BATCH_SIZE] [--max_length MAX_LENGTH] [--cpu]
                    {opus,mbart,madlad,m2m,nllb} ... target_lang checkpoint_location

Translate an instruct/RLHF dataset to a given target language using a variety of translation models

positional arguments:
  {opus,mbart,madlad,m2m,nllb}
                        The model/architecture used for translation.
    opus                Translate the dataset using HelsinkiNLP OPUS models.
    mbart               Translate the dataset using mBART.
    madlad              Translate the dataset using Google's MADLAD models.
    m2m                 Translate the dataset using Facebook's M2M models.
    nllb                Translate the dataset using Facebook's NLLB models.
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

4. Turn the translated dataset into threads in LLaMa2-chat format. We do this by always using the highest ranking answer following a given input prompt.

```
usage: create_thread_prompts.py [-h] [--base_dataset_text_field BASE_DATASET_TEXT_FIELD]
                                [--base_dataset_rank_field BASE_DATASET_RANK_FIELD]
                                [--base_dataset_id_field BASE_DATASET_ID_FIELD]
                                [--base_dataset_parent_field BASE_DATASET_PARENT_FIELD]
                                dataset_name instruction_prompt output_location

Turn the translated dataset into threads in LLaMa2-chat format. We do this by always using the
highest ranking answer following a given input prompt.

positional arguments:
  dataset_name          The input dataset, loaded from Huggingface datasets or disk. This should
                        be the result of the previous step.
  instruction_prompt    An instruction message added to every prompt given to the chatbot to force
                        it to answer in the target language. Example: "You are a generic chatbot
                        that always answers in English."
  output_location       Where to write the Huggingface Dataset to. Can be a disk location or a
                        Huggingface Dataset repository. Be sure to set up HF_TOKEN.

options:
  -h, --help            show this help message and exit
  --base_dataset_text_field BASE_DATASET_TEXT_FIELD
                        The dataset's column name containing the actual text to translate.
                        Defaults to text
  --base_dataset_rank_field BASE_DATASET_RANK_FIELD
                        The dataset's column name containing the rank of an answer given to a
                        prompt. Defaults to rank
  --base_dataset_id_field BASE_DATASET_ID_FIELD
                        The dataset's column name containing the id of a text. Defaults to
                        message_id
  --base_dataset_parent_field BASE_DATASET_PARENT_FIELD
                        The dataset's column name containing the parent id of a text. Defaults to
                        parent_id
```

5. Fine-tune a foundate model's instruct using LoRA and PEFT.

```
usage: finetune_llama.py [-h] [--base_model BASE_MODEL] tuned_model dataset_name

Finetune a base model using QLoRA and PEFT

positional arguments:
  tuned_model           The name of the resulting tuned model. This will be pushed to Huggingface.
                        Ensure HF_TOKEN is set.
  dataset_name          The name of the dataset to use for fine-tuning.

options:
  -h, --help            show this help message and exit
  --base_model BASE_MODEL
                        The base foundation model. Default is "NousResearch/Llama-2-7b-chat-hf".
```

6. Run inference using the newly created QLoRA model.

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

# Datasets and models

We have created and will continue to create numerous datasets and models already. **Want to help democratize LLMs?** Clone the repo and create datasets and models for other languages, then create a PR.

## Translated oasst1 datasets using OPUS

- [UnderstandLing/oasst1_nl](https://huggingface.co/datasets/UnderstandLing/oasst1_nl) The oasst1 dataset translated to Dutch.
- [UnderstandLing/oasst1_es](https://huggingface.co/datasets/UnderstandLing/oasst1_es) The oasst1 dataset translated to Spanish.
- [UnderstandLing/oasst1_fr](https://huggingface.co/datasets/UnderstandLing/oasst1_fr) The oasst1 dataset translated to French.
- [UnderstandLing/oasst1_de](https://huggingface.co/datasets/UnderstandLing/oasst1_de) The oasst1 dataset translated to German.
- [xaviviro/oasst1_ca](https://huggingface.co/datasets/xaviviro/oasst1_ca) The oasst1 dataset translated to Catalan.
- [UnderstandLing/oasst1_pt](https://huggingface.co/datasets/UnderstandLing/oasst1_pt) The oasst1 dataset translated to Portuguese.
- [HeshamHaroon/oasst-arabic](https://huggingface.co/datasets/HeshamHaroon/oasst-arabic) The oasst1 dataset translated Arabic.
- [UnderstandLing/oasst1_it](https://huggingface.co/datasets/UnderstandLing/oasst1_it) The oasst1 dataset translated to Italian.
- [UnderstandLing/oasst1_ru](https://huggingface.co/datasets/UnderstandLing/oasst1_ru) The oasst1 dataset translated to Russian.
- [UnderstandLing/oasst1_hi](https://huggingface.co/datasets/UnderstandLing/oasst1_hi) The oasst1 dataset translated to Hindi.
- [UnderstandLing/oasst1_zh](https://huggingface.co/datasets/UnderstandLing/oasst1_zh) The oasst1 dataset translated to Chinese.
- [chrystians/oasst1_pl](https://huggingface.co/datasets/chrystians/oasst1_pl) The oasst1 dataset translated to Polish.
- [UnderstandLing/oasst1_jap](https://huggingface.co/datasets/UnderstandLing/oasst1_jap) The oasst1 dataset translate to Japanese.
- [xezpeleta/oasst1_eu](https://huggingface.co/datasets/xezpeleta/oasst1_eu) The oasst1 dataset translated to Basque.

## Translated LLaMa2 thread chat prompt datasets

- [UnderstandLing/oasst1_nl_threads](https://huggingface.co/datasets/UnderstandLing/oasst1_nl_threads) The LLaMa2 chat prompts with history from threads in oasst1 for Dutch.
- [UnderstandLing/oasst1_es_threads](https://huggingface.co/datasets/UnderstandLing/oasst1_es_threads) The LLaMa2 chat prompts with history from threads in oasst1 for Spanish.
- [UnderstandLing/oasst1_fr_threads](https://huggingface.co/datasets/UnderstandLing/oasst1_fr_threads) The LLaMa2 chat prompts with history from threads in oasst1 for French.
- [UnderstandLing/oasst1_de_threads](https://huggingface.co/datasets/UnderstandLing/oasst1_de_threads) The LLaMa2 chat prompts with history from threads in oasst1 for German.
- [xaviviro/oasst1_ca_threads](https://huggingface.co/datasets/xaviviro/oasst1_ca_threads) The LLaMa2 chat prompts with history from threads in oasst1 for Catalan.
- [UnderstandLing/oasst1_pt_threads](https://huggingface.co/datasets/UnderstandLing/oasst1_pt_threads) The LLaMa2 chat prompts with history from threads in oasst1 for Portuguese.
- [HeshamHaroon/oasst1-ar-threads](https://huggingface.co/datasets/HeshamHaroon/oasst1-ar-threads) The LLaMa2 chat prompts with history from threads in oasst1 for Arabic.
- [UnderstandLing/oasst1_it_threads](https://huggingface.co/datasets/UnderstandLing/oasst1_it_threads) The LLaMa2 chat prompts with history from threads in oasst1 for Italian.
- [UnderstandLing/oasst1_ru_threads](https://huggingface.co/datasets/UnderstandLing/oasst1_ru_threads) The LLaMa2 chat prompts with history from threads in oasst1 for Russian.
- [UnderstandLing/oasst1_hi_threads](https://huggingface.co/datasets/UnderstandLing/oasst1_hi_threads) The LLaMa2 chat prompts with history from threads in oasst1 for Hindi.
- [UnderstandLing/oasst1_zh_threads](https://huggingface.co/datasets/UnderstandLing/oasst1_zh_threads) The LLaMa2 chat prompts with history from threads in oasst1 for Chinese.
- [chrystians/Jestes](https://huggingface.co/datasets/chrystians/Jestes) The LLaMa2 chat prompts with history from threads in oasst1 for Polish.
- [xezpeleta/oasst1_eu_threads](https://huggingface.co/datasets/xezpeleta/oasst1_eu_threads) The LLaMa2 chat prompts with history from threads in oasst1 for Basque.


## Language-specific LLaMa2-7B chat model adapters

- [UnderstandLing/llama-2-7b-chat-nl](https://huggingface.co/UnderstandLing/llama-2-7b-chat-nl) QLoRA adapter for LLaMa2-7b-chat in Dutch.
- [UnderstandLing/llama-2-7b-chat-es](https://huggingface.co/UnderstandLing/llama-2-7b-chat-es) QLoRA adapter for LLaMa2-7b-chat in Spanish.
- [UnderstandLing/llama-2-7b-chat-fr](https://huggingface.co/UnderstandLing/llama-2-7b-chat-fr) QLoRA adapter for LLaMa2-7b-chat in French.
- [UnderstandLing/llama-2-7b-chat-de](https://huggingface.co/UnderstandLing/llama-2-7b-chat-de) QLoRA adapter for LLaMa2-7b-chat in German.
- [xaviviro/llama-2-7b-chat-ca](https://huggingface.co/xaviviro/llama-2-7b-chat-ca) QLoRA adapter for LLaMa2-7b-chat in Catalan.
- [UnderstandLing/llama-2-7b-chat-pt](https://huggingface.co/UnderstandLing/llama-2-7b-chat-pt) QLoRA adapter for LLaMa2-7b-chat in Portuguese.
- [HeshamHaroon/llama-2-7b-chat-ar](https://huggingface.co/HeshamHaroon/llama-2-7b-chat-ar) QLoRA adapter for LLaMa2-7b-chat in Arabic.
- [UnderstandLing/llama-2-7b-chat-it](https://huggingface.co/UnderstandLing/llama-2-7b-chat-it) QLoRA adapter for LLaMa2-7b-chat in Italian.
- [UnderstandLing/llama-2-7b-chat-ru](https://huggingface.co/UnderstandLing/llama-2-7b-chat-ru) QLoRA adapter for LLaMa2-7b-chat in Russian.
- [UnderstandLing/llama-2-7b-chat-hi](https://huggingface.co/UnderstandLing/llama-2-7b-chat-hi) QLoRA adapter for LLaMa2-7b-chat in Hindi.
- [UnderstandLing/llama-2-7b-chat-zh](https://huggingface.co/UnderstandLing/llama-2-7b-chat-zh) QLoRA adapter for LLaMa2-7b-chat in Chinese.
- [chrystians/llama-2-7b-chat-pl-polish-polski](https://huggingface.co/chrystians/llama-2-7b-chat-pl-polish-polski) QLoRA adapter for LLaMa2-7b-chat in Polish.
- [xezpeleta/llama-2-7b-chat-eu](https://huggingface.co/xezpeleta/llama-2-7b-chat-eu) QLoRA adapter for LLaMa2-7b-chat in Basque.

## Language-specific LLaMa2-13B chat model adapters

- [UnderstandLing/llama-2-13b-chat-nl](https://huggingface.co/UnderstandLing/llama-2-13b-chat-nl) QLoRA adapter for LLaMa2-13B in Dutch.
- [UnderstandLing/llama-2-13b-chat-es](https://huggingface.co/UnderstandLing/llama-2-13b-chat-es) QLoRA adapter for LLaMa2-13B in Spanish.
- [UnderstandLing/llama-2-13b-chat-fr](https://huggingface.co/UnderstandLing/llama-2-13b-chat-fr) QLoRA adapter for LLaMa2-13B in French.

## Language-specific Mixtral-8x7B chat model adapters

- [UnderstandLing/Mixtral-8x7B-Instruct-nl](https://huggingface.co/UnderstandLing/Mixtral-8x7B-Instruct-nl) QLoRA adapter for Mixtral-8x7B in Dutch.

# Empirical performance

## Dutch

`<s>[INST] <<SYS>> Je bent een generieke chatbot die altijd in het Nederlands antwoord geeft. <</SYS>> Wat is de hoofdstad van Nederland? [/INST] Amsterdam</s>`

`<s>[INST] <<SYS>> Je bent een generieke chatbot die altijd in het Nederlands antwoord geeft. <</SYS>> Wat is de hoofdstad van Nederland? [/INST] Amsterdam</s><s>[INST] Hoeveel inwoners heeft die stad? [/INST] 850 duizend inwoners (2023)</s>`

`<s>[INST] <<SYS>> Je bent een generieke chatbot die altijd in het Nederlands antwoord geeft. <</SYS>> Wat is de hoofdstad van Nederland? [/INST] Amsterdam</s><s>[INST] Hoeveel inwoners heeft die stad? [/INST] 850 duizend inwoners (2023)</s><s>[INST] In welke provincie ligt die stad? [/INST] In de provincie Noord-Holland</s>`

`<s>[INST] <<SYS>> Je bent een generieke chatbot die altijd in het Nederlands antwoord geeft. <</SYS>> Wie is de minister-president van Nederland? [/INST] Mark Rutte is sinds 2010 minister-president van Nederland. Hij is meerdere keren herkozen.</s>`

# FAQ

* Q: Why do you translate the full OASST1/2 dataset first? Wouldn't it be faster to only translate highest ranked threads?
* A: While you can gain quite a lot in terms of throughput time by first creating the threads and then translating them, we provide full OASST1/2 translations to the community as we believe they can be useful on their own.

* Q: How well do the fine-tunes perform compared to vanilla LLaMa2?
* A: While we do not have formal benchmarks, getting LLaMa2 to consistently speak another language than English to begin with is challenging if not impossible. The non-English language it does produce is often grammatically broken. Our fine-tunes do not show this behavior.

* Q: Can I use other frameworks for fine-tuning?
* A: Yes you can, we use [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) for training on multi-GPU setups.

* Q: Can I mix different translation models?
* A: Absolutely, we think it might even increase performance to have translation done by multiple models. You can achieve this by early-stopping a translation and continuing from the checkpoints by reruning the translate script with a different translation model.

# Funding
We are based in the Netherland and actively looking for funding to democratize AI and advance its applications. Contact us at funding@understandling.com if you want to invest.
