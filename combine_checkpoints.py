import json
import os
from datasets import Dataset, DatasetDict
import pandas as pd
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Combine checkpoint files from translation.")
    parser.add_argument('input_folder', type=str, 
                        help='The checkpoint folder used in translation, with the target language appended. Example: "./checkpoints/nl".')
    parser.add_argument('output_location', type=str,
                        help='Where to write the Huggingface Dataset. Can be a disk location or a Huggingface Dataset repository.')
    args = parser.parse_args()
    input_folder = args.input_folder
    output_location = args.output_location

    dataset = {}
    # Get the subdirectories which will become the keys of the Dataset
    folds = [name for name in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, name))]

    for fold in folds:
      all_data = []

      for lang_folder in os.listdir(os.path.join(input_folder, fold)):
          for filename in os.listdir(os.path.join(input_folder, fold, lang_folder)):
            if filename.endswith('.json'):
                file_path = os.path.join(input_folder, fold, lang_folder, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    all_data.extend(data)

      dataset[fold] = Dataset.from_pandas(pd.DataFrame(data=all_data))

    dataset = DatasetDict(dataset)
    # Check if output location is a valid directory
    if os.path.isdir(output_location):
        dataset.save_to_disk(output_location)
    else:
        # Try to push to hub, requires HF_TOKEN environment variable to be set, see https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hftoken
        dataset.push_to_hub(output_location)

if __name__ == "__main__":
    main()