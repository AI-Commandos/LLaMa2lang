# LLaMa2lang
This repository contains convenience scripts to finetune LLaMa2 for chat towards any language (that isn't English). The rationale behind this is that LLaMa2 is trained on primarily English data and while it works to some extent for other languages, its performance is poor compared to English.

# Working
The process we follow to tune LLaMa2 for a specific language is as follows:

1. We use the [Open Assistant dataset](https://huggingface.co/datasets/OpenAssistant/oasst1) from Huggingface as our base instruct data.
2. We extract prompt-assistant pairs from the dataset, keeping only the highest ranked assistant replies per prompt.
3. The result is a new dataset which we uploaded for convenience here (TBD).
4. These new prompt-assitant pairs are then translated using [Helsinki-NLP's OPUS translation models](https://huggingface.co/Helsinki-NLP). This follows a two-step process to translate the prompt and the assistant response separately:

    4.1 Try to translate from the source language to the desired target language using a model for that language pair.

    4.2 If there is no such model, try to translate from the source language to English first and then from English to the target language. If this fails too, the prompt-assistant pair is excluded.
5. The result is an instruct dataset identical to 3. but now translated to the target language.
6. Use QLoRA and PEFT to finetune LLaMa2 on this dataset.

# Usage
0. Make sure pytorch is installed and working for your environment (use of CUDA preferable): https://pytorch.org/get-started/locally/

1. Clone the repo and install the requirements.

`pip install -r requirements.txt`

2. Convert the Open Assistant dataset to prompt-assistant pairs.

`python create_pairs.py instruct.json`

3. Translate to the desired target language. Replace NL with your target language and make sure the output folder `translated_instruct` exists. The output folder is used to create checkpoints as a full translate takes a lot of time.

`python translate_pairs.py instruct.json nl translated_instruct/`

4. Fine-tune LLaMa2

`python finetune.py translated_instruct model_name`