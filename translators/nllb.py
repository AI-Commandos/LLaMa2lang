from translators.base import BaseTranslator
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class NLLBTranslator(BaseTranslator):
    language_mapping = {
        'en': 'eng_Latn',
        'es': 'spa_Latn',
        'de': 'deu_Latn',
        'ru': 'rus_Cyrl',
        'ja': 'jpn_Jpan',
        'pt-BR': 'por_Latn',
        'ca': 'cat_Latn',
        'fr': 'fra_Latn',
        'pl': 'pol_Latn',
        'vi': 'vie_Latn',
        'zh': 'zho_Hant',
        'hu': 'hun_Latn',
        'ko': 'kor_Hang',
        'eu': 'eus_Latn',
        'it': 'ita_Latn',
        'uk-UA': 'ukr_Cyrl',
        'id': 'ind_Latn',
        'ar': 'arb_Arab',
        'fi': 'fin_Latn',
        'tr': 'tur_Latn',
        'da': 'dan_Latn',
        'th': 'tha_Thai',
        'sv': 'swe_Latn',
        'cs': 'ces_Latn'
    }

    def __init__(self, device, quant4, quant4_config, quant8, max_length, model_size):
        super().__init__(device, quant4, quant4_config, quant8, max_length)

        model_name = f'facebook/nllb-200-{model_size}'
        # Load model and tokenizer
        if self.quant4:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=device, quantization_config=self.quant4_config, load_in_4bit=True)
        elif self.quant8:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=self.device, load_in_8bit=True)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = model
        self.tokenizer = tokenizer

    def translate(self, texts, source_lang, target_lang):
        self.tokenizer.src_lang = self.language_mapping[source_lang]
        with torch.no_grad():
            if self.max_length is None:
                encoded_batch = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
            else:
                encoded_batch = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
            outputs = self.model.generate(**encoded_batch, forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang])
            translated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            return translated_texts
