# LLaMa2lang v0.2
This repository contains convenience scripts to finetune LLaMa2-7b for chat towards any language (that isn't English). The rationale behind this is that LLaMa2 is trained on primarily English data and while it works to some extent for other languages, its performance is poor compared to English.

# Change info
* **[2024-01-04]** We now support translation through MADLAD. Especially for models where Helsinki has a low BLEU score (less than 40), MADLAD is preferred. Using MADLAD drastically slows down training time, especially if you quantize (4 bit is even slower than 8 bit).
* **[2024-01-04]** We now use argparser to parse command line arguments. Make sure you update your calls to our scripts accordingly. Use `-h` on all scripts to get help.
* **[2023-12-29]** We now batch translations in `translate_oasst.py` for a 30-60% speed increase. If you have checkpoints from before this date, you can **not** continue using the main branch but instead must use the [v0.1 branch](https://github.com/UnderstandLingBV/LLaMa2lang/tree/v0.1).

# TL;DR

```
pip install -r requirements.txt

# Translate OASST1 to target language
python translate_oasst.py target_lang checkpoint_location

# Combine the checkpoint files into a dataset
python combine_checkpoints.py input_folder output_location

# Create threaded prompts
python create_thread_prompts.py dataset_name instruction_prompt output_location

# Finetune
python finetune_llama.py tuned_model dataset_name

# Run inference
python run_inference.py model_name instruction_prompt input
```

# Roadmap
* [L2L-2] Make the base model (llama2-7b-chat) configurable so you can also finetune Mistral, Mixtral or others.
* [L2L-4] Add DPO training as RLHF alternative
* [L2L-5] Investigate multi-GPU support
* [L2L-6] Investigate interoperability with other libraries (Axolotl, llamacpp, unsloth)
* [L2L-7] Allow for different quantizations next to QLoRA (GGUF, GPTQ, AWQ)
* [L2L-8] Add mBART translation as an option
* [L2L-9] Add m2m translation as an option

# What it does
The process we follow to tune LLaMa2 for a specific language is as follows:

1. We use the [Open Assistant dataset](https://huggingface.co/datasets/OpenAssistant/oasst1) from Huggingface as our base instruct data.
2. The dataset is fully translated into a specific target language using [Helsinki-NLP's OPUS translation models](https://huggingface.co/Helsinki-NLP). This follows a two-step process:

    2.1 Try to translate from the source language specified in OASST1 to the desired target language using a model for that language pair.

    2.2 If there is no such model, try to translate from the source language to English first and then from English to the target language.
3. Load the translated OASST1 dataset and extract threads by recursively selecting prompts with their respective answers with the highest rank only, through to subsequent prompts, etc.
4. Turn the threads into texts using [LLaMa's prompt format](https://huggingface.co/blog/llama2#how-to-prompt-llama-2).
5. Use QLoRA and PEFT to finetune LLaMa2-chat on this dataset.

## Cost and runtime

The above process can be fully run on a free Google Colab T4 GPU. The last step however, can only be successfully run with short enough context windows and a batch of at most 2. In addition, the translation in step 2 takes about 36 hours in total for any given language so should be run in multiple steps if you want to stick with a free Google Colab GPU.

Our fine-tuned models for step 5 were performed using an A40 on [vast.ai](https://vast.ai/) and cost us less than a dollar for each model, completing in about 1.5 hours.

# Usage
1. Make sure pytorch is installed and working for your environment (use of CUDA preferable): https://pytorch.org/get-started/locally/

2. Clone the repo and install the requirements.

`pip install -r requirements.txt`

2. Translate the OASST1 dataset into your target language. This script writes out intermediate results to a `checkpoint_location` because its runtime is quite lengthy (about 30-40 hours on a T4 Google Colab GPU).

```
usage: translate_oasst.py [-h] [--base_dataset BASE_DATASET]
                          [--base_dataset_text_field BASE_DATASET_TEXT_FIELD]
                          [--base_dataset_lang_field BASE_DATASET_LANG_FIELD]
                          [--checkpoint_n CHECKPOINT_N] [--batch_size BATCH_SIZE]
                          target_lang checkpoint_location

Translate a dataset (default: oasst1) to target language

positional arguments:
  target_lang           The target language, using language codes as used in Helsinki-NLP's OPUS
                        translation models.
  checkpoint_location   The folder the script will write (JSONized) checkpoint files to. Folder
                        will be created if it doesn't exist.

options:
  -h, --help            show this help message and exit
  --base_dataset BASE_DATASET
                        The base dataset to translate, defaults to OpenAssistant/oasst1
  --base_dataset_text_field BASE_DATASET_TEXT_FIELD
                        The base dataset's column name containing the actual text to translate.
                        Defaults to text
  --base_dataset_lang_field BASE_DATASET_LANG_FIELD
                        The base dataset's column name containing the language the source text was
                        written in. Defaults to lang
  --checkpoint_n CHECKPOINT_N
                        An integer representing how often a checkpoint file will be written out.
                        For OASST1, 400 is a reasonable number.
  --batch_size BATCH_SIZE
                        The batch size for a single translation model. Adjust based on your GPU
                        capacity. Default is 20.
```

3. Combine the JSON arrays from the checkpoints' files into a Huggingface Dataset and then either write it to disk or publish it to Huggingface. The script will try to write to disk by default and fall back to publishing to Huggingface if the folder doesn't exist on disk. For publishing to Huggingface, make sure you have your `HF_TOKEN` environment variable set up as per [the documentation](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hftoken).

```
usage: combine_checkpoints.py [-h] input_folder output_location

Combine checkpoint files from translation.

positional arguments:
  input_folder     The checkpoint folder used in translation, with the target language appended.
                   Example: "./checkpoints/nl".
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

5. Fine-tune LLaMa2-7B-chat (or another base model) using LoRA and PEFT.

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

We have created and will continue to create numerous datasets and models already. **Want to help democratize LLMs?** Clone the repor and create datasets and models for other languages, then create a PR.

## Translated oasst1 datasets

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

## Language-specific LLaMa2-13B chat model adapters

- [UnderstandLing/llama-2-13b-chat-nl](https://huggingface.co/UnderstandLing/llama-2-13b-chat-nl) QLoRA adapter for LLaMa2-13B in Dutch.
- [UnderstandLing/llama-2-13b-chat-es](https://huggingface.co/UnderstandLing/llama-2-13b-chat-es) QLoRA adapter for LLaMa2-13B in Spanish.
- [UnderstandLing/llama-2-13b-chat-fr](https://huggingface.co/UnderstandLing/llama-2-13b-chat-fr) QLoRA adapter for LLaMa2-13B in French.

## Language-specific Mixtral-8x7B chat model adapters

- [UnderstandLing/Mixtral-8x7B-Instruct-nl](https://huggingface.co/UnderstandLing/Mixtral-8x7B-Instruct-nl) QLoRA adapter for Mixtral-8x7B in Dutch.

# Empirical performance

## Dutch

`<s>[INST] <<SYS>> Je bent een generieke chatbot die altijd in het Nederlands antwoord geeft. <</SYS>> Wat is de hoofdstad van Nederland? [/INST] Amsterdam`

`<s>[INST] <<SYS>> Je bent een generieke chatbot die altijd in het Nederlands antwoord geeft. <</SYS>> Wat is de hoofdstad van Nederland? [/INST] Amsterdam</s><s>[INST] Hoeveel inwoners heeft die stad? [/INST] 850 duizend inwoners (2023)`

`<s>[INST] <<SYS>> Je bent een generieke chatbot die altijd in het Nederlands antwoord geeft. <</SYS>> Wat is de hoofdstad van Nederland? [/INST] Amsterdam</s><s>[INST] Hoeveel inwoners heeft die stad? [/INST] 850 duizend inwoners (2023)</s><s>[INST] In welke provincie ligt die stad? [/INST] In de provincie Noord-Holland`

`<s>[INST] <<SYS>> Je bent een generieke chatbot die altijd in het Nederlands antwoord geeft. <</SYS>> Wie is de minister-president van Nederland? [/INST] Mark Rutte is sinds 2010 minister-president van Nederland. Hij is meerdere keren herkozen.`

# FAQ

* Q: Why do you translate the full OASST1 dataset first? Wouldn't it be faster to only translate highest ranked threads?
* A: While you can gain quite a lot in terms of throughput time by first creating the threads and then translating them, we provide full OASST1 translations to the community as we believe they can be useful on their own.

* Q: How well do the fine-tunes perform compared to vanilla LLaMa2?
* A: While we do not have formal benchmarks, getting LLaMa2 to consistently speak another language than English to begin with is challenging if not impossible. The non-English language it does produce is often grammatically broken. Our fine-tunes do not show this behavior.

* Q: Can I use other frameworks for fine-tuning?
* A: Yes you can, we use [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) for training on multi-GPU setups.
