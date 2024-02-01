from translators.base import BaseTranslator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class TowerInstructTranslator(BaseTranslator):
    language_mapping = {
        'en': 'English',
        'pt': 'Portuguese',
        'pt-BR': 'Portuguese',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'nl': 'Dutch',
        'it': 'Italian',
        'ko': 'Korean',
        'zh': 'Chinese',
        'ru': 'Russian',
        'uk': 'Ukrainian'
    }
    def __init__(self, device, quant4, quant4_config, quant8, max_length):
        super().__init__(device, quant4, quant4_config, quant8, max_length)

        model_name = f'Unbabel/TowerInstruct-7B-v0.1'
        # Load model and tokenizer
        if self.quant4:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, quantization_config=self.quant4_config, load_in_4bit=True)
        elif self.quant8:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map=self.device, load_in_8bit=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.nlp_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=self.device)
        self.printed_error_langs = {}

    def translate(self, texts, source_lang, target_lang):
        if source_lang in self.language_mapping and target_lang in self.language_mapping:
            src_lang = self.language_mapping[source_lang]
            trgt_lang = self.language_mapping[target_lang]

            with torch.no_grad():
                texts = [{'role':'user','content': f'Translate the following text from {src_lang} into {trgt_lang}.\n{src_lang}: {t}\n{trgt_lang}:'} for t in texts]
                prompts = [self.nlp_pipeline.tokenizer.apply_chat_template([text], tokenize=False, add_generation_prompt=True).to(self.device) for text in texts]
                if self.max_length is None:
                    outputs = [self.nlp_pipeline(prompt, do_sample=False) for prompt in prompts]
                else:
                    outputs = [self.nlp_pipeline(prompt, max_new_tokens=self.max_length, do_sample=False) for prompt in prompts]
                
                # Remove the prompts from the outputs
                result = []
                for output, prompt in zip(outputs, prompts):
                    result.append(output[0]['generated_text'][len(prompt):])

                return result
        else:
            if not(source_lang in self.printed_error_langs):
                print(f"[---- LLaMa2Lang ----] Tower Instruct cannot translate from source language {source_lang} or to your target language {target_lang}, returning originals")
                self.printed_error_langs[source_lang] = True
            return None