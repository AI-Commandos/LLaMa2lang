from translators.base import BaseTranslator
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
import torch

class OPUSTranslator(BaseTranslator):
    def __init__(self, device, quant4, quant4_config, quant8, max_length):
        super().__init__(device, quant4, quant4_config, quant8, max_length)
        # Cache for loaded translation models, seemingly faster than letting Huggingface handle it
        self.model_cache = {}
        # Alternative models that are not created by Helsink-NLP
        self.alternative_models = {
            "en-pl": 'gsarti/opus-mt-tc-en-pl',
            "en-ja": 'gsarti/opus-mt-tc-base-en-ja'
        }

    def translate(self, texts, source_lang, target_lang):
        with torch.no_grad():
            model, tokenizer = self.get_helsinki_nlp_model(source_lang, target_lang)
            if model is None or tokenizer is None:
                # Try via intermediate language
                model_i, tokenizer_i = self.get_helsinki_nlp_model(source_lang, 'en')
                model_t, tokenizer_t = self.get_helsinki_nlp_model('en', target_lang)
                if model_i is None or tokenizer_i is None or model_t is None or tokenizer_t is None:
                    print(f"[---- LLaMa2Lang ----] No translation possible from {source_lang} to {target_lang}")
                    return None

                # To intermediate language first
                if self.max_length is None:
                    # OPUS crashes if we pass it more than 512 tokens
                    inputs = tokenizer_i(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                    translated_outputs = model_i.generate(inputs.input_ids)
                else:
                    inputs = tokenizer_i(texts, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length).to(self.device)
                    translated_outputs = model_i.generate(inputs.input_ids, max_length=self.max_length)
                intermediate_texts = [tokenizer_i.decode(output, skip_special_tokens=True) for output in translated_outputs]

                # Now to target
                if self.max_length is None:
                    inputs = tokenizer_t(intermediate_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                    translated_outputs = model_t.generate(inputs.input_ids)
                else:
                    inputs = tokenizer_t(intermediate_texts, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length).to(self.device)
                    translated_outputs = model_t.generate(inputs.input_ids, max_length=self.max_length)
                translated_texts = [tokenizer_t.decode(output, skip_special_tokens=True) for output in translated_outputs]
                return translated_texts
            else:
                if self.max_length is None:
                    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                    translated_outputs = model.generate(inputs.input_ids)
                else:
                    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length).to(self.device)
                    translated_outputs = model.generate(inputs.input_ids, max_length=self.max_length)
                translated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in translated_outputs]
                return translated_texts

    def load_model(self, model_name, model_key):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Apply quantization if needed
        if self.quant4:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=self.device, quantization_config=self.quant4_config, load_in_4bit=True)
        elif self.quant8:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=self.device, load_in_8bit=True)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model_cache[model_key] = (model, tokenizer)
        return model, tokenizer

    # Tries to obtain a translation model from the Helsinki-NLP groups OPUS models. Returns None, None if no model is found for this language pair
    def get_helsinki_nlp_model(self, source_lang, target_lang):
        # Small fix for odd language codes
        if source_lang == 'pt-BR':
            source_lang = 'bzs'
        if source_lang == 'uk-UA':
            source_lang = 'uk'
        model_key = f'{source_lang}-{target_lang}'

        if model_key in self.model_cache:
            return self.model_cache[model_key]

        model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
        try:
            return self.load_model(model_name, model_key)
        except OSError as e:
            # Try to load the tc-big naming convention files
            try:
                model_name = f'Helsinki-NLP/opus-mt-tc-big-{source_lang}-{target_lang}'
                return self.load_model(model_name, model_key)
            except OSError as e:
                try:
                    model_name = self.alternative_models[model_key]
                    return self.load_model(model_name, model_key)
                except Exception as e:
                    return None, None