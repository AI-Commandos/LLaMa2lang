import argparse
import os

import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
from pandas import DataFrame
from tqdm import tqdm
from transformers import (
    AutoTokenizer, PreTrainedTokenizerBase,
)


def format_dpo(thread: list[str], system_instruction: str, bad_child: str, tokenizer: PreTrainedTokenizerBase) \
        -> dict[str, str]:
    chat = [
        {"role": "system", "content": system_instruction}
    ]

    formatted_thread: dict[str, str] = {}

    for i in range(0, len(thread) - 1):
        if i % 2 == 0:
            chat.append({"role": "user", "content": thread[i]})
        else:
            chat.append({"role": "assistant", "content": thread[i]})

    formatted_thread['prompt'] = tokenizer.apply_chat_template(chat, tokenize=False)[len(tokenizer.bos_token):]
    formatted_thread['chosen'] = thread[len(thread) - 1]
    formatted_thread['rejected'] = bad_child

    return formatted_thread


# We only continue the thread with the highest ranked answer to each input
def find_children_and_highest_ranked_child(df: DataFrame, parent_id: int) -> tuple[DataFrame, DataFrame]:
    children = df[df['parent_id'] == parent_id]
    max_rank = children['rank'].max()

    if not children.empty:
        return children[children['rank'] == max_rank], children[children['rank'] != max_rank]

    df_empty = children.iloc[:0, :].copy()

    return df_empty, df_empty


def main():
    parser = argparse.ArgumentParser(
        description="Turn the translated dataset into threads in LLaMa2-chat format. We do this by always using the highest ranking answer following a given input prompt.")
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
    parser.add_argument("--base_model", type=str, default="NousResearch/Llama-2-7b-chat-hf")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    instruction_prompt = args.instruction_prompt
    output_location = args.output_location
    base_dataset_text_field = args.base_dataset_text_field
    base_dataset_rank_field = args.base_dataset_rank_field
    base_dataset_id_field = args.base_dataset_id_field
    base_dataset_parent_field = args.base_dataset_parent_field
    base_model = args.base_model

    # Load base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    if os.path.isdir(dataset_name):
        dataset = load_from_disk(os.path.join(dataset_name))
    else:
        dataset = load_dataset(dataset_name)

    # Construct threads
    folds = dataset.keys()
    threads = {k: [] for k in folds}
    for fold in folds:
        print(f"Creating threads for fold {fold}")

        df: DataFrame = dataset[fold].to_pandas()

        # Replace NULLs in rank with a value lower than the lowest rank
        min_rank = df[base_dataset_rank_field].min()
        df[base_dataset_rank_field].fillna(min_rank - 1, inplace=True)

        # Identify root messages (those without a parent_id)
        root_messages = df[df[base_dataset_parent_field].isna()]

        with tqdm(total=len(root_messages)) as pbar:
            for _, root_message in root_messages.iterrows():
                # Create the thread
                thread: list[str] = [root_message[base_dataset_text_field]]

                good_child, bad_children = find_children_and_highest_ranked_child(df,
                                                                                  root_message[base_dataset_id_field])

                while not good_child.empty:
                    thread.append(good_child.iloc[0][base_dataset_text_field])

                    for bad_child in bad_children.iterrows():
                        formatted_dpo = format_dpo(thread, instruction_prompt, bad_child[1]['text'], tokenizer)
                        threads[fold].append(formatted_dpo)

                    good_child, bad_children = find_children_and_highest_ranked_child(df, good_child[
                        base_dataset_id_field].iloc[0])

                # Update progress
                pbar.update(1)

        threads[fold] = Dataset.from_pandas(pd.DataFrame(data=threads[fold]))

    dataset = DatasetDict(threads)

    # Check if output location is a valid directory
    if os.path.isdir(output_location):
        dataset.save_to_disk(output_location)
    else:
        # Try to push to hub, requires HF_TOKEN environment variable to be set, see https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hftoken
        dataset.push_to_hub(output_location)


if __name__ == "__main__":
    main()
