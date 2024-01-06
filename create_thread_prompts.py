from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
import sys
import os
import pandas as pd
from tqdm import tqdm
import argparse

# Function to format a thread in LLaMa2 format
def format_thread(thread, system_instruction):
    formatted_thread = f"<s>[INST] <<SYS>>\n{system_instruction}\n<</SYS>>\n\n"

    for i in range(0, len(thread), 2):
        user_msg = thread[i]['text'] if i < len(thread) else ""
        model_answer = thread[i+1]['text'] if i+1 < len(thread) else ""
        formatted_thread += f"{user_msg} [/INST] {model_answer} </s>"
        if i+2 < len(thread):
            formatted_thread += f"<s>[INST] "

    return formatted_thread

# We only continue the thread with the highest ranked answer to each input
def find_highest_ranked_child(df, parent_id):
      children = df[df['parent_id'] == parent_id]
      if not children.empty:
          return children.loc[children['rank'].idxmax()]
      return None

def main():
    parser = argparse.ArgumentParser(description="Turn the translated dataset into threads in LLaMa2-chat format. We do this by always using the highest ranking answer following a given input prompt.")
    parser.add_argument('dataset_name', type=str, 
                        help='The input dataset, loaded from Huggingface datasets or disk. This should be the result of the previous step.')
    parser.add_argument('instruction_prompt', type=str, 
                        help='An instruction message added to every prompt given to the chatbot to force it to answer in the target language. Example: "You are a generic chatbot that always answers in English."')
    parser.add_argument('output_location', type=str,
                        help='Where to write the Huggingface Dataset to. Can be a disk location or a Huggingface Dataset repository. Be sure to set up HF_TOKEN.')
    parser.add_argument('--base_dataset_text_field', type=str, default="text",
                        help="The dataset's column name containing the actual text to translate. Defaults to text")
    parser.add_argument('--base_dataset_rank_field', type=str, default="rank",
                        help="The dataset's column name containing the rank of an answer given to a prompt. Defaults to rank")
    parser.add_argument('--base_dataset_id_field', type=str, default="message_id",
                        help="The dataset's column name containing the id of a text. Defaults to message_id")
    parser.add_argument('--base_dataset_parent_field', type=str, default="parent_id",
                        help="The dataset's column name containing the parent id of a text. Defaults to parent_id")
    args = parser.parse_args()
    dataset_name = args.dataset_name
    instruction_prompt = args.instruction_prompt
    output_location = args.output_location
    base_dataset_text_field = args.base_dataset_text_field
    base_dataset_rank_field = args.base_dataset_rank_field
    base_dataset_id_field = args.base_dataset_id_field
    base_dataset_parent_field = args.base_dataset_parent_field

    if os.path.isdir(dataset_name):
        dataset = load_from_disk(os.path.join(dataset_name))
    else:
        dataset = load_dataset(dataset_name)
    
    # Construct threads
    folds = dataset.keys()
    threads = {k: [] for k in folds}
    for fold in folds:
        print(f"Creating threads for fold {fold}")
        
        df = dataset[fold].to_pandas()

        # Replace NULLs in rank with a value lower than the lowest rank
        min_rank = df[base_dataset_rank_field].min()
        df[base_dataset_rank_field].fillna(min_rank - 1, inplace=True)

        # Identify root messages (those without a parent_id)
        root_messages = df[df[base_dataset_parent_field].isna()]

        with tqdm(total=len(root_messages)) as pbar:
            for _, root_message in root_messages.iterrows():
                # Create the thread
                thread = [{'text': root_message[base_dataset_text_field]}]
                next_message = find_highest_ranked_child(df, root_message[base_dataset_id_field])
            
                while next_message is not None:
                    thread.append({'text': next_message[base_dataset_text_field]})
                    next_message = find_highest_ranked_child(df, next_message[base_dataset_id_field])
            
                # Turn this into LLaMa2 format
                threads[fold].append(format_thread(thread, instruction_prompt))
                # Update progress
                pbar.update(1)

        threads[fold] = Dataset.from_pandas(pd.DataFrame(data=threads[fold])).rename_column('0', 'text')

    dataset = DatasetDict(threads)

    # Check if output location is a valid directory
    if os.path.isdir(output_location):
        dataset.save_to_disk(output_location)
    else:
        # Try to push to hub, requires HF_TOKEN environment variable to be set, see https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hftoken
        dataset.push_to_hub(output_location)

if __name__ == "__main__":
    main()