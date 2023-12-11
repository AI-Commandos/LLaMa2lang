from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import sys
import os
import re
import json

input = sys.argv[1]
target_lang = sys.argv[2]
checkpoint_location = sys.argv[3]
checkpoint_n = int(sys.argv[4])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if the dataset exists on Huggingface first
try:
  instruct_dataset = load_dataset(input)
except FileNotFoundError as e:
  with open(input, 'r', encoding='utf-8') as f:
    instruct_dataset = json.load(f)

# Create output folder
if not os.path.exists(checkpoint_location):
    os.makedirs(checkpoint_location)

# Cache for loaded translation models, seemingly faster than letting Huggingface handle it
model_cache = {}

# Tries to obtain a translation model from the Helsinki-NLP groups OPUS models. Returns None, None if no model is found for this language pair
def get_helsinki_nlp_model(source_lang, target_lang):
    model_key = f'{source_lang}-{target_lang}'

    if model_key in model_cache:
        return model_cache[model_key]

    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    try:
      tokenizer = AutoTokenizer.from_pretrained(model_name)
      model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
      model_cache[model_key] = (model, tokenizer)
      return model, tokenizer
    except Exception as e:
        return None, None

# If a direct translation between two languages isn't possible, we ettempt to use English as a bridge (or any other intermediate lang)
def translate_text_through_english(text, source_lang, target_lang, intermediate_lang='en'):
    # Translate from source to English
    model, tokenizer = get_helsinki_nlp_model(source_lang, intermediate_lang)
    if model is None or tokenizer is None:
        return None
    inputs = tokenizer.encode(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        intermediate_translation = model.generate(inputs, max_length=512)
    text_in_english = tokenizer.decode(intermediate_translation[0], skip_special_tokens=True)

    # Translate from English to target
    model, tokenizer = get_helsinki_nlp_model(intermediate_lang, target_lang)
    if model is None or tokenizer is None:
        return None  # Can't perform translation
    inputs = tokenizer.encode(text_in_english, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        final_translation = model.generate(inputs, max_length=512)
    final_text = tokenizer.decode(final_translation[0], skip_special_tokens=True)

    return final_text

# Translate a given text from a source language into a target language
def translate_text(text, source_lang, target_lang):
    # Try direct translation first
    try:
      model, tokenizer = get_helsinki_nlp_model(source_lang, target_lang)
    except:
      model = None
      tokenizer = None
    if model is not None and tokenizer is not None:
        inputs = tokenizer.encode(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            translated = model.generate(inputs, max_length=512)
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    # If direct translation is not available, use English as intermediate
    return translate_text_through_english(text, source_lang, target_lang)

# Find the max checkpoint number to continue from
def find_largest_checkpoint(checkpoint_location):
    pattern = r'upto_(\d+).json'
    files = os.listdir(checkpoint_location)
    numbers = [int(re.search(pattern, file).group(1)) for file in files if re.match(pattern, file)]
    if numbers:
        return max(numbers)
    else:
        return 0
last_checkpoint_n = find_largest_checkpoint(checkpoint_location)

translated_texts = []
cnt = 0
for record in instruct_dataset:
    # Check if there is already a checkpoint up to this batch
    if cnt <= last_checkpoint_n:
      cnt += 1
      continue

    # Write out checkpoint file
    if cnt % checkpoint_n == 0 and cnt != 0:
      with open(f'{checkpoint_location}/upto_{cnt}.json', 'w', encoding='utf-8') as f:
        json.dump(translated_texts, f, ensure_ascii=False)
      translated_texts = []

    should_include = True
    instruct_translated = record['instruction']
    output_translated = record['output']

    # First translate the instruction input
    if record['i_lang'] != target_lang:
      try:
        translated_text = translate_text(record['instruction'], record['i_lang'], target_lang)
      except:
        translated_text = None
        should_include = False
      if not(translated_text is None):
        instruct_translated = translated_text

    # Next the output
    if record['o_lang'] != target_lang:
      try:
        translated_text = translate_text(record['output'], record['o_lang'], target_lang)
      except:
        translated_text = None
        should_include = False
      if not(translated_text is None):
        output_translated = translated_text
        
    # Only add if both instruction and output got translated
    if should_include:
      full_prompt_answer = {'INSTRUCTION': instruct_translated, 'RESPONSE': output_translated} # Axolotl oasst format
      translated_texts.append(full_prompt_answer)
    cnt += 1

with open(f'{checkpoint_location}/upto_{cnt}.json', 'w', encoding='utf-8') as f:
  json.dump(translated_texts, f, ensure_ascii=False)