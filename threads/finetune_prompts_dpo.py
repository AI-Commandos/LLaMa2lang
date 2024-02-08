from pandas import DataFrame
from tqdm import tqdm

def format_dpo(
        thread: list[str],
        system_instruction: str,
        bad_child: str,
        tokenizer,
        chat_template) \
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

    # Run it untokenized so we can write it out
    formatted_thread['prompt'] = tokenizer.apply_chat_template(chat, tokenize=False, chat_template=chat_template)[len(tokenizer.bos_token):]
    formatted_thread['chosen'] = thread[-1]
    formatted_thread['rejected'] = bad_child

    return formatted_thread


# We only continue the thread with the highest ranked answer to each input
def find_children_and_highest_ranked_child(
        df: DataFrame,
        parent_id: int,
        base_dataset_parent_field: str,
        base_dataset_rank_field: str) -> tuple[DataFrame, DataFrame]:
    children = df[df[base_dataset_parent_field] == parent_id]
    min_rank = children[base_dataset_rank_field].min()

    if not children.empty:
        return children[children[base_dataset_rank_field] == min_rank], children[children[base_dataset_rank_field] != min_rank]

    df_empty = children.iloc[:0, :].copy()

    return df_empty, df_empty


def create_prompts(dataset, tokenizer, base_dataset_rank_field, base_dataset_parent_field, base_dataset_id_field, base_dataset_text_field, instruction_prompt, chat_template):
    # Construct threads
    threads = []
    df = dataset.to_pandas()

    # Replace NULLs in rank with a value highest than the highest rank
    max_rank = df[base_dataset_rank_field].max()
    df[base_dataset_rank_field].fillna(max_rank + 1, inplace=True)

    # Identify root messages (those without a parent_id)
    root_messages = df[df[base_dataset_parent_field].isna()]

    with tqdm(total=len(root_messages)) as pbar:
        for _, root_message in root_messages.iterrows():
            # Create the thread
            thread: list[str] = [root_message[base_dataset_text_field]]

            good_child, bad_children = find_children_and_highest_ranked_child(df,
                root_message[base_dataset_id_field], base_dataset_parent_field, base_dataset_rank_field)

            while not good_child.empty:
                thread.append(good_child.iloc[0][base_dataset_text_field])

                for bad_child in bad_children.iterrows():
                    formatted_dpo = format_dpo(thread, instruction_prompt, bad_child[1][base_dataset_text_field], tokenizer, chat_template)
                    threads.append(formatted_dpo)

                good_child, bad_children = find_children_and_highest_ranked_child(df, good_child[
                    base_dataset_id_field].iloc[0], base_dataset_parent_field, base_dataset_rank_field)

            # Update progress
            pbar.update(1)

    return threads
