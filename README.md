# LLaMa2lang
This repository contains convenience scripts to finetune LLaMa2-7b for chat towards any language (that isn't English). The rationale behind this is that LLaMa2 is trained on primarily English data and while it works to some extent for other languages, its performance is poor compared to English.

# What it does
The process we follow to tune LLaMa2 for a specific language is as follows:

1. We use the [Open Assistant dataset](https://huggingface.co/datasets/OpenAssistant/oasst1) from Huggingface as our base instruct data.
2. We extract prompt-assistant pairs from the dataset, keeping only the highest ranked assistant replies per prompt.
3. The result is a new dataset which we uploaded for convenience: [UnderstandLing/oasst1_instruct](https://huggingface.co/datasets/UnderstandLing/oasst1_instruct).
4. These new prompt-assitant pairs are then translated using [Helsinki-NLP's OPUS translation models](https://huggingface.co/Helsinki-NLP). This follows a two-step process to translate the prompt and the assistant response separately:

    4.1 Try to translate from the source language to the desired target language using a model for that language pair.

    4.2 If there is no such model, try to translate from the source language to English first and then from English to the target language. If this fails too, the prompt-assistant pair is excluded.
5. The result is an instruct dataset identical to 3. but now translated to the target language.
6. Use QLoRA and PEFT to finetune LLaMa2 on this dataset.

# Usage
0. Make sure pytorch is installed and working for your environment (use of CUDA preferable): https://pytorch.org/get-started/locally/

1. Clone the repo and install the requirements.

`pip install -r requirements.txt`

2. Convert the Open Assistant dataset to prompt-assistant pairs. You can optionally skip this step and use our dataset from Huggingface: [UnderstandLing/oasst1_instruct](https://huggingface.co/datasets/UnderstandLing/oasst1_instruct)

`python create_pairs.py instruct.json`

3. Translate to the desired target language. Replace NL with your target language and make sure the output folder `translated_instruct` exists. The output folder is used to create checkpoints as a full translate takes a lot of time. First the code checks if the input dataset exists on disk. If that fails, it tries to retrieve the dataset from Huggingface.

`python translate_pairs.py UnderstandLing/oasst1_instruct nl translated_instruct 100`

or

`python translate_pairs.py instruct.json nl translated_instruct 100`

4. [*Optional*] Publish your dataset to Huggingface. You will have to combine the JSON arrays from the checkpoints' files into a Huggingface Dataset and can then publish it. Note that the below script will create a Dataset as per the Open Assistant data format that can readily be used in [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl). If you want to use the dataset for fine-tuning using the script in Step 5, be sure to skip this step 4.

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