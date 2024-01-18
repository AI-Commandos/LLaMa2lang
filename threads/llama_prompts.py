from datasets import Dataset
import pandas as pd
from tqdm import tqdm

# We only continue the thread with the highest ranked answer to each input
def find_highest_ranked_child(df, parent_id):
      children = df[df['parent_id'] == parent_id]
      if not children.empty:
          return children.loc[children['rank'].idxmax()]
      return None

# Creates the prompts
def create_prompts(dataset, tokenizer, base_dataset_rank_field, base_dataset_parent_field, base_dataset_id_field, base_dataset_text_field, base_dataset_author_field, instruction_prompt):
    # Construct threads
    threads = []
    df = dataset.to_pandas()

    # Replace NULLs in rank with a value lower than the lowest rank
    min_rank = df[base_dataset_rank_field].min()
    df[base_dataset_rank_field].fillna(min_rank - 1, inplace=True)

    # Identify root messages (those without a parent_id)
    root_messages = df[df[base_dataset_parent_field].isna()]

    with tqdm(total=len(root_messages)) as pbar:
        for _, root_message in root_messages.iterrows():
            # Create the thread
            if root_message[base_dataset_author_field] == 'prompter':
                role = 'user'
            else:
                role = 'assistant'
            thread = [{
                'content': f" <<SYS>>\n{instruction_prompt.strip()}\n<</SYS>\n\n" + root_message[base_dataset_text_field],
                'role': role
            }]
            next_message = find_highest_ranked_child(df, root_message[base_dataset_id_field])
        
            while next_message is not None:
                thread.append({
                    'content': next_message[base_dataset_text_field],
                    'role': role
                })
                next_message = find_highest_ranked_child(df, next_message[base_dataset_id_field])
        
            # Turn this into LLaMa2 format
            threads.append(tokenizer.apply_chat_template(thread))
            # Update progress
            pbar.update(1)
    
    return threads