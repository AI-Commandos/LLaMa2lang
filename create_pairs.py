from datasets import load_dataset
import json
import sys

output = sys.argv[1]

dataset = load_dataset("OpenAssistant/oasst1")

def get_instruction_output_pairs(data):
    # Create a dictionary to index assistant messages by their parent_id
    assistant_messages = {}
    for record in data:
        if record['role'] == 'assistant' and record['parent_id'] is not None:
            parent_id = record['parent_id']
            rank = record['rank'] if record['rank'] is not None else -1

            # Update the dictionary only if this record has a higher rank
            if parent_id not in assistant_messages or assistant_messages[parent_id]['rank'] is None or rank > assistant_messages[parent_id]['rank']:
                assistant_messages[parent_id] = record

    # Iterate through prompter messages and find the best assistant response
    for record in data:
        if record['role'] == 'prompter':
            best_response = assistant_messages.get(record['message_id'])
            if best_response:
                yield {
                    "instruction": record['text'],
                    "output": best_response['text'],
                    "i_lang": record['lang'],
                    "o_lang": best_response['lang']
                }

instruct_dataset = []
for instruct in get_instruction_output_pairs(dataset['train']):
    instruct_dataset.append(instruct)

with open(output, 'w', encoding='utf-8') as f:
  json.dump(instruct_dataset, f, ensure_ascii=False)