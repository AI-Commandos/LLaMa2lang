import argparse
import torch
from transformers import BitsAndBytesConfig
from datasets import load_dataset
import gc
from sacrebleu.metrics import BLEU, CHRF
from translators.m2m import M2MTranslator
from translators.madlad import MADLADTranslator
from translators.mbart import mBARTTranslator
from translators.nllb import NLLBTranslator
from translators.opus import OPUSTranslator

def main():
    allowed_models = ['opus', 'm2m_418m', 'm2m_1.2b', 'madlad_3b', 'madlad_7b', 'madlad_10b', 'madlad_7bbt', 'mbart', 'nllb_distilled600m', 'nllb_1.3b', 'nllb_distilled1.3b', 'nllb_3.3b']
    parser = argparse.ArgumentParser(description="Benchmark all the different translation models for a specific source and target language to find out which performs best. This uses 4bit quantization to limit GPU usage. Note: the outcomes are indicative - you cannot assume corretness of the BLEU and CHRF scores but you can compare models against each other relatively.")
    parser.add_argument('source_language', type=str,
                        help='The source language you want to test for. Check your dataset to see which occur most prevalent or use English as a good start.')
    parser.add_argument('target_language', type=str,
                        help='The source language you want to test for. This should be the language you want to apply the translate script on. Note: in benchmark, we use 2-character language codes, in constrast to translate.py where you need to specify whatever your model expects.')
    mdls = ', '.join(allowed_models)
    parser.add_argument('included_models', type=str,
                        help=f'Comma-separated list of models to include. Allowed values are: {mdls}')
    parser.add_argument('--cpu', action='store_true',
                        help="Forces usage of CPU. By default GPU is taken if available.")
    parser.add_argument('--start', type=int, default=0,
                        help="The starting offset to include sentences from the OPUS books dataset from. Defaults to 0.")
    parser.add_argument('--n', type=int, default=100,
                        help="The number of sentences to benchmark on. Defaults to 100.")
    parser.add_argument('--max_length', type=int, default=512,
                        help="How much tokens to generate at most. More tokens might be more accurate for lengthy input but creates a risk of running out of memory. Default is 512.")
    args = parser.parse_args()
    source_language = args.source_language
    target_language = args.target_language
    included_models = args.included_models
    force_cpu = args.cpu
    start = args.start
    n = args.n
    max_length = args.max_length

    # Initialize common parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() and not(force_cpu) else "cpu")
    # Set up quantization configs if required
    quant4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Initialize scorers
    bleu = BLEU()
    chrf = CHRF()

    # Handle the models
    models = [m for m in included_models.lower().split(",") if m in allowed_models]
    print(f"[---- LLaMa2Lang ----] Starting benchmarking from {source_language} to {target_language} for models {models} on {n} records on device {device}")

    # Load the OPUS dataset
    dataset = load_dataset("opus100", f'{source_language}-{target_language}', split=f'train[{start}:{start+n}]').shuffle().select(range(n))

    # Process each model one at a time
    translator = None
    for model in models:
        # Clear CUDA
        del translator
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
        gc.collect()

        # Handle the model naming
        model_target_language = target_language
        if model.startswith('madlad'):
            model_size = model.split('_')[1]
            if model_size == '7bbt':
                model_size = '7b-bt'
            translator = MADLADTranslator(device, True, quant4_config, False, max_length, model_size)
        elif model.startswith('m2m'):
            model_size = model.split('_')[1]
            translator = M2MTranslator(device, True, quant4_config, False, max_length, model_size)
        elif model.startswith('mbart'):
            translator = mBARTTranslator(device, True, quant4_config, False, max_length)
            model_target_language = translator.language_mapping[target_language]
        elif model.startswith('nllb'):
            model_size = model.split('_')[1]
            translator = NLLBTranslator(device, True, quant4_config, False, max_length, model_size)
            # TODO: Extend this later, there are far more languages
            model_target_language = translator.language_mapping[target_language]
        else:
            translator = OPUSTranslator(device, False, quant4_config, False, max_length)
        
        # Run the translations
        translated = []
        for s in dataset['translation']:
            translated += translator.translate([s[source_language]], source_language, model_target_language)

        # Compute scores, using max_length is not at all correct but it's better than not doing it at all
        b_score = bleu.corpus_score([s[:max_length] for s in translated], [[s[target_language][:max_length] for s in dataset['translation']]])
        c_score = chrf.corpus_score([s[:max_length] for s in translated], [[s[target_language][:max_length] for s in dataset['translation']]])
        # Report
        print(f"[---- LLaMa2Lang ----] [{model}] BLEU: {b_score.score}")
        print(f"[---- LLaMa2Lang ----] [{model}] CHRF: {c_score.score}")
        print("")

if __name__ == "__main__":
    main()