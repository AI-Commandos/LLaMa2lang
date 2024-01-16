from translators.base import BaseTranslator
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

class mBARTTranslator(BaseTranslator):
    language_mapping = {
        'ar': 'ar_AR',
        'cs': 'cs_CZ',
        'de': 'de_DE',
        'en': 'en_XX',
        'es': 'es_XX',
        'et': 'et_EE',
        'fi': 'fi_FI',
        'fr': 'fr_XX',
        'gu': 'gu_IN',
        'hi': 'hi_IN',
        'it': 'it_IT',
        'ja': 'ja_XX',
        'kk': 'kk_KZ',
        'ko': 'ko_KR',
        'lt': 'lt_LT',
        'lv': 'lv_LV',
        'my': 'my_MM',
        'ne': 'ne_NP',
        'nl': 'nl_XX',
        'ro': 'ro_RO',
        'ru': 'ru_RU',
        'si': 'si_LK',
        'tr': 'tr_TR',
        'vi': 'vi_VN',
        'zh': 'zh_CN',
        'af': 'af_ZA',
        'az': 'az_AZ',
        'bn': 'bn_IN',
        'fa': 'fa_IR',
        'he': 'he_IL',
        'hr': 'hr_HR',
        'id': 'id_ID',
        'ka': 'ka_GE',
        'km': 'km_KH',
        'mk': 'mk_MK',
        'ml': 'ml_IN',
        'mn': 'mn_MN',
        'mr': 'mr_IN',
        'pl': 'pl_PL',
        'ps': 'ps_AF',
        'pt': 'pt_XX',
        'pt-BR': 'pt_XX',
        'sv': 'sv_SE',
        'sw': 'sw_KE',
        'ta': 'ta_IN',
        'te': 'te_IN',
        'th': 'th_TH',
        'tl': 'tl_XX',
        'uk_UA': 'uk_UA',
        'uk': 'uk_UA',
        'ur': 'ur_PK',
        'xh': 'xh_ZA',
        'gl': 'gl_ES',
        'sl': 'sl_SI'
    }

    def __init__(self, device, quant4, quant4_config, quant8, max_length):
        super().__init__(device, quant4, quant4_config, quant8, max_length)

        model_name = 'facebook/mbart-large-50-many-to-many-mmt'
        # Load model and tokenizer
        if self.quant4:
            model = MBartForConditionalGeneration.from_pretrained(model_name, device_map=device, quantization_config=self.quant4_config, load_in_4bit=True)
        elif self.quant8:
            model = MBartForConditionalGeneration.from_pretrained(model_name, device_map=self.device, load_in_8bit=True)
        else:
            model = MBartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

        self.model = model
        self.tokenizer = tokenizer
        self.printed_error_langs = {}

    def translate(self, texts, source_lang, target_lang):
        if source_lang in self.language_mapping:
            self.tokenizer.src_lang = self.language_mapping[source_lang]
            with torch.no_grad():
                if self.max_length is None:
                    encoded_batch = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
                else:
                    encoded_batch = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
                outputs = self.model.generate(**encoded_batch, forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang])
                translated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                return translated_texts
        else:
            if not(source_lang in self.printed_error_langs):
                print(f"[---- LLaMa2Lang ----] mBART cannot translate from source language {source_lang}, returning originals")
                self.printed_error_langs[source_lang] = True
            return None
