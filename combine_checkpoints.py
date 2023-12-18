import json
import os
from datasets import Dataset, DatasetDict
import pandas as pd
import sys

input_folder = sys.argv[1]
output_location = sys.argv[2]
dataset = {}
# Get the subdirectories which will become the keys of the Dataset
folds = [name for name in os.listdir(input_folder) if os.path.isdir(name)]

for fold in folds:
  all_data = []

  for filename in os.listdir(fold):
      if filename.endswith('.json'):
          file_path = os.path.join(fold, filename)
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