# Change info
_v0.5_
* **[2024-02-08]** Added DPO training as an RLHF substitute.
* **[2024-02-08]** Added more translation methods like Tower Instruct (LLM) and Google Gemini via API (no GPU required)

_v0.4_
* **[2024-01-18]** The create threads script has been removed. We now directly use the chat template provided by the base model's tokenizer, thus supporting mutliple chat/instruct/prompt templates.

_v0.3_
* **[2024-01-12]** You can now benchmark different translation models using `benchmark.py`.
* **[2024-01-09]** We have significantly refactored the translation process. Please follow the readme carefully if you come from v0.2.
* **[2024-01-09]** We now support translation through M2M.
* **[2024-01-04]** We now support translation through MADLAD. Especially for models where Helsinki has a low BLEU score (less than 40), MADLAD (or the faster M2M) is preferred. Using MADLAD drastically slows down training time, especially if you quantize (4 bit is even slower than 8 bit).
* **[2024-01-04]** We now use argparser to parse command line arguments. Make sure you update your calls to our scripts accordingly. Use `-h` on all scripts to get help.

_v0.2_
* **[2023-12-29]** We now batch translations in `translate.py` for a 30-60% speed increase. If you have checkpoints from before this date, you can **not** continue using the main branch but instead must use the [v0.1 branch](https://github.com/UnderstandLingBV/LLaMa2lang/tree/v0.1).
