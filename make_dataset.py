from datasets import load_dataset

dataset = load_dataset('Salesforce/dialogstudio', 'MediaSum')

sub_sample = 100

json_output = []
for i,example in enumerate(dataset["train"]):
    prompt = "\n".join([seq for seq in eval(example["original dialog info"])["dialog history"]])
    instruction = f"Summarize the following dialogue: ### {prompt} ###"
    output = eval(example["original dialog info"])["summary"]
    json_output.append({"instruction": instruction, "input": "", "output": output})
    if i == sub_sample:
        break


# Define the file path where you want to save the JSON file
output_file_path = 'data/mediasum.json'

import json

# Write the `json_output` to a JSON file
with open(output_file_path, 'w') as json_file:
    json.dump(json_output, json_file, indent=4)

print(f'JSON data has been saved to {output_file_path}')

