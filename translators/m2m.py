from translators.base import BaseTranslator
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer
)
import torch

class M2MTranslator(BaseTranslator):
    def __init__(self, device, quant4, quant4_config, quant8, max_length, model_size):
        super().__init__(device, quant4, quant4_config, quant8, max_length)
        self.model_size = model_size

        model_name = f'facebook/m2m100_{self.model_size}'
        # Load model and tokenizer
        if self.quant4:
            model = M2M100ForConditionalGeneration.from_pretrained(model_name, device_map=device, quantization_config=self.quant4_config, load_in_4bit=True)
        elif self.quant8:
            model = M2M100ForConditionalGeneration.from_pretrained(model_name, device_map=self.device, load_in_8bit=True)
        else:
            model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)

        self.model = model
        self.tokenizer = tokenizer

    def translate(self, texts, source_lang, target_lang):
        # Small fix for odd language codes
        if source_lang == 'pt-BR':
            source_lang = 'pt'
        if source_lang == 'uk-UA':
            source_lang = 'uk'
        with torch.no_grad():
            if source_lang == 'eu':
                # Not supported by M2M
                return None
            # Set the source language for the tokenizer
            self.tokenizer.src_lang = source_lang
            if self.max_length is None:
                encoded_batch = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
                generated_tokens = self.model.generate(**encoded_batch, forced_bos_token_id=self.tokenizer.get_lang_id(target_lang))
            else:
                encoded_batch = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
                generated_tokens = self.model.generate(**encoded_batch, max_length=self.max_length, forced_bos_token_id=self.tokenizer.get_lang_id(target_lang))
            translated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            return translated_texts
