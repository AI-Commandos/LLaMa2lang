from translators.base import BaseTranslator
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer
)
import torch

class MADLADTranslator(BaseTranslator):
    def __init__(self, device, quant4, quant4_config, quant8, max_length, model_size):
        super().__init__(device, quant4, quant4_config, quant8, max_length)
        self.model_size = model_size

        model_name = f'google/madlad400-{self.model_size}-mt'
        # Quick rewrite the model name for bt
        if self.model_size == '7b-bt':
            model_name = f'google/madlad400-{self.model_size}-mt-bt'
        # Load model and tokenizer
        if self.quant4:
            model = T5ForConditionalGeneration.from_pretrained(model_name, device_map=device, quantization_config=self.quant4_config, load_in_4bit=True)
        elif self.quant8:
            model = T5ForConditionalGeneration.from_pretrained(model_name, device_map=self.device, load_in_8bit=True)
        else:
            model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        tokenizer = T5Tokenizer.from_pretrained(model_name)

        self.model = model
        self.tokenizer = tokenizer

    def translate(self, texts, source_lang, target_lang):
        with torch.no_grad():
            # Preprocess texts and add target language prefix
            madlad_texts = [f'<2{target_lang}> ' + text.replace("\n", " ") for text in texts]
            if self.max_length is None:
                encoded_batch = self.tokenizer(madlad_texts, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model.generate(input_ids=encoded_batch['input_ids'], max_new_tokens=2048) # max_new_tokens is required otherwise we get 20
            else:
                encoded_batch = self.tokenizer(madlad_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
                outputs = self.model.generate(input_ids=encoded_batch['input_ids'], max_new_tokens=self.max_length)
            translated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            return translated_texts
