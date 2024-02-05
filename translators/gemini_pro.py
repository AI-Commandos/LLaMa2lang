from translators.base import BaseTranslator
import google.generativeai as genai
import asyncio
class GeminiProTranslator(BaseTranslator):
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

    def __init__(self, access_token, max_length):
        if access_token is None:
            raise Exception("Access token is required!")
        super().__init__(None, None, None, None, max_length)
        genai.configure(api_key=access_token)
        self.printed_error_langs = {}
        self.model = genai.GenerativeModel('gemini-pro')

    async def translate_text(self, text, prompt):
        response = self.model.generate_content(f"{prompt}\n{text}")
        if response.prompt_feedback:
            print(text)
            print(response.prompt_feedback)
        else:
            print(response.text)
        return response.text

    async def translate_texts(self, texts, prompt):
        tasks = []
        for text in texts:
            tasks.append(self.translate_text(text, prompt))
            await asyncio.sleep(1)
        results = await asyncio.gather(*tasks)
        return results
    def translate(self, texts, source_lang, target_lang):
        if len(texts) > 60:
            raise Exception("Batch size cannot be more than 60 for this translator due ratelimit in 60 RPM!")
        if source_lang in self.language_mapping and target_lang in self.language_mapping:
            trgt_lang = self.language_mapping[target_lang]
            prompt = (f"Translate text below to {trgt_lang} language and preserve formatting and special characters. "
                  f"Respond with translated text ONLY. Here is text to translate: ")
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(self.translate_texts(texts, prompt))
            return result
        else:
            if not (source_lang in self.printed_error_langs):
                print(
                    f"[---- LLaMa2Lang ----] Tower Instruct cannot translate from source language {source_lang} or to your target language {target_lang}, returning originals")
                self.printed_error_langs[source_lang] = True
            return None

