from translators.base import BaseTranslator
from transformers import SeamlessM4Tv2ForTextToText, AutoProcessor
import torch


class Seamless_M4T_V2(BaseTranslator):
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
        'uk': 'ukr_Cyrl',
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

        model_name = f'facebook/seamless-m4t-v2-{model_size}'
        # Load model and tokenizer
        if self.quant4:
            model = SeamlessM4Tv2ForTextToText.from_pretrained(model_name, device_map=device,
                                                               quantization_config=self.quant4_config,
                                                               load_in_4bit=True)
        elif self.quant8:
            model = SeamlessM4Tv2ForTextToText.from_pretrained(model_name, device_map=self.device, load_in_8bit=True)
        else:
            model = SeamlessM4Tv2ForTextToText.from_pretrained(model_name).to(self.device)
        processor = AutoProcessor.from_pretrained(model_name)
        self.model = model
        self.processor = processor

    def translate(self, texts, source_lang, target_lang):
        self.processor.src_lang = self.language_mapping[source_lang]
        with torch.no_grad():
            if self.max_length is None:
                encoded_batch = self.processor(texts, return_tensors="pt", padding=True).to(self.device)
            else:
                encoded_batch = self.processor(texts, return_tensors="pt", padding=True, truncation=True,
                                               max_length=self.max_length).to(self.device)
            decoder_input_ids = self.model.generate(**encoded_batch,
                                                    tgt_lang=self.language_mapping[target_lang])[0].tolist()
            translated_texts = self.processor.decode(decoder_input_ids, skip_special_tokens=True)
            return translated_texts
