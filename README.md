# LLaMa2lang
This repository contains convenience scripts to finetune LLaMa2-7b for chat towards any language (that isn't English). The rationale behind this is that LLaMa2 is trained on primarily English data and while it works to some extent for other languages, its performance is poor compared to English.

# What it does
The process we follow to tune LLaMa2 for a specific language is as follows:

1. We use the [Open Assistant dataset](https://huggingface.co/datasets/OpenAssistant/oasst1) from Huggingface as our base instruct data.
2. The dataset is fully translated into a specific target language using [Helsinki-NLP's OPUS translation models](https://huggingface.co/Helsinki-NLP). This follows a two-step process:

    2.1 Try to translate from the source language specified in OASST1 to the desired target language using a model for that language pair.

    2.2 If there is no such model, try to translate from the source language to English first and then from English to the target language.
3. Load the translated OASST1 dataset and extract threads by recursively selecting prompts with their respective answers with the highest rank only, through to subsequent prompts, etc.
4. Turn the threads into texts using [LLaMa's prompt format](https://huggingface.co/blog/llama2#how-to-prompt-llama-2).
5. Use QLoRA and PEFT to finetune LLaMa2-chat on this dataset.

# Usage
1. Make sure pytorch is installed and working for your environment (use of CUDA preferable): https://pytorch.org/get-started/locally/

2. Clone the repo and install the requirements.

`pip install -r requirements.txt`

2. Translate the OASST1 dataset into your target language. This script writes out intermediate results to a `checkpoint_location` because its runtime is quite lengthy (about 30-40 hours on a T4 Google Colab GPU).

`python translate_oasst.py [TARGET_LANG] [CHECKPOINT_FOLDER] [CHECKPOINT_N]`

Parameters:

- `TARGET_LANG` The target language, use ISO language codes as used in the [Helsinki-NLP's OPUS translation models](https://huggingface.co/Helsinki-NLP).
- `CHECKPOINT_FOLDER` The folder the script will write (JSONized) checkpoint files to. Folder will be created if it doesn't exist.
- `CHECKPOINT_N` An integer representing how often a checkpoint file will be written out. OASST1 contains 84.4k records in train and another 4.4k records in validation. We found `200` to be a reasonable number for this parameter.

3. Combine the JSON arrays from the checkpoints' files into a Huggingface Dataset and then either write it to disk or publish it to Huggingface. The script will try to write to disk by default and fall back to publishing to Huggingface if the folder doesn't exist on disk. For publishing to Huggingface, make sure you have your `HF_TOKEN` environment variable set up as per [the documentation](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hftoken).

`python combine_checkpoints.py [CHECKPOINT_FOLDER] [OUTPUT_LOCATION]`

Parameters:

- `[CHECKPOINT_FOLDER]` The same checkpoint location as used in translation but now with the `[TARGET_LANG]` added to it. Example: checkpoint location used to write: `./checkpoints` for target language `nl` results in this script requiring `./checkpoints/nl`.
- [OUTPUT_LOCATION] Where to write the Huggingface Dataset to. Can either be a location on disk or a Hugginface Dataset repository if the location on disk does not exist. Be sure to set up `HF_TOKEN`.

**Option 1**

5. Fine-tune LLaMa2-7B using LoRA and PEFT. Change the name of the base model in the script if you want to use a different foundation model.

`python finetune.py translated_instruct model_name`

**Option 2**

5. Fine-tune LLaMa2 using Axolotl. This gives you more flexibility over the type of quantization you want to apply (LoRA, QLoRA, ReLoRA) and allows you to swap out LLaMa-7b for a different model.

    1. Install [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) by following its readme. Note that you have to be careful in selecting and installing the right Python, pyTorch, CUDA and Flash-Attention versions. This repo was tested against Python 3.9 with CUDA 11.7, pyTorch 2.0.1 and flash-attn 2.3.3 (installed from the [releases page](https://github.com/Dao-AILab/flash-attention/releases)).

# Datasets and models

- [oasst1_instruct](https://huggingface.co/datasets/UnderstandLing/oasst1_instruct) contains the prompt/assistant pairs that are the result of `create_pairs.py`.
- [oasst1_nl](https://huggingface.co/datasets/UnderstandLing/oasst1_nl) contains the result of translating the instruct dataset to Dutch as a result of `translate_pairs.py`.
- [llama-2-3b-chat-nl-lora](https://huggingface.co/UnderstandLing/llama-2-3b-chat-nl-lora) is a fine-tuned LLaMa2 3B instruct model, taking [open_llama_3b_v2](https://huggingface.co/openlm-research/open_llama_3b_v2) as base model using LoRA