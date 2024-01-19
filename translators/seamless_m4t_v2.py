from translators.base import BaseTranslator
from transformers import SeamlessM4Tv2ForTextToText, AutoProcessor
from stanza.pipeline.core import DownloadMethod
import stanza
import torch


class Seamless_M4T_V2(BaseTranslator):
    language_mapping = {
        'en': 'eng',
        'es': 'spa',
        'de': 'deu',
        'ru': 'rus',
        'ja': 'jpn',
        'pt-BR': 'por',
        'ca': 'cat',
        'fr': 'fra',
        'pl': 'pol',
        'vi': 'vie',
        'zh': 'zho',
        'hu': 'hun',
        'ko': 'kor',
        'eu': 'eus',
        'it': 'ita',
        'uk-UA': 'ukr',
        'uk': 'ukr',
        'id': 'ind',
        'ar': 'arb',
        'fi': 'fin',
        'tr': 'tur',
        'da': 'dan',
        'th': 'tha',
        'sv': 'swe',
        'cs': 'ces'
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
            # Seamless is good for short messages/sentences,
            # so there is need to conduct sentence segmentation to have a
            # good quality translation of texts
            nlp_processors = {'tokenize': 'spacy'} if source_lang == 'en' else 'tokenize'
            nlp = stanza.Pipeline(
                lang=source_lang,
                download_method=DownloadMethod.REUSE_RESOURCES,
                processors=nlp_processors,
                use_gpu=True,
                verbose=False
            )
            sentence_segmented_texts = nlp.bulk_process(texts)
            translated_texts = []
            for document in sentence_segmented_texts:
                translated_text = ""
                for sentence in document.sentences:
                    decoder_input_ids = self.translate_text(target_lang, sentence.text)
                    translated_text += self.processor.decode(decoder_input_ids, skip_special_tokens=True) + " "
                translated_texts.append(translated_text.strip())
            return translated_texts

    def translate_text(self, target_lang, text):
        if self.max_length is None:
            encoded_batch = self.processor(text, return_tensors="pt", padding=True).to(self.device)
        else:
            encoded_batch = self.processor(text, return_tensors="pt", padding=True, truncation=True,
                                           max_length=self.max_length).to(self.device)
        decoder_input_ids = self.model.generate(**encoded_batch,
                                                tgt_lang=self.language_mapping[target_lang])[0].tolist()
        return decoder_input_ids
