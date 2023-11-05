from datasets import load_dataset
import numpy as np


print("Loading dataset... \n")
# load dataset avoiding timeout and print progress
#mediasum_dataset = load_dataset('Salesforce/dialogstudio', 'MediaSum', cache_dir='mnt/c/Users/pyanez/Desktop/pablo/universidad/memoria/src/data/downloads')
mediasum_dataset = load_dataset('Salesforce/dialogstudio', 'MediaSum', cache_dir='data')
#mediasum_dataset = load_dataset('Salesforce/dialogstudio', 'MediaSum', split='validation')
json_output = []
max_samples = 100

print("Generating samples... \n")
for j, example in enumerate(mediasum_dataset["validation"]):
    prompt = "\n".join([seq for seq in eval(example["original dialog info"])["dialog history"]])
    json_output.append({"instruction": prompt, "input": "", "output": "", "id": j})
    if j == max_samples:
        break

dataset_path= 'data'
output_dir= 'predictions'
target_dataset = "mediasum"
#
args = {
    "stage": "sft",
    "model_name_or_path": "meta-llama/Llama-2-7b-chat-hf",
    "do_predict": True,
    "dataset": target_dataset,
    "dataset_dir": dataset_path,
    "template": "llama2",
    "output_dir": output_dir,
    "per_device_eval_batch_size": 1,
    "max_samples": 1,
    "predict_with_generate": True,
    "cutoff_len": 4000
}

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args["model_name_or_path"], use_fast=True, token = "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",
                                          cache_dir="data", trust_remote_code=True)

lengths = []
# print("Counting inputs_ids... \n")
for i, example in enumerate(json_output):
    inputs_id = tokenizer.encode(example["instruction"], add_special_tokens=True)
    lengths.append(len(inputs_id))

# get max length
max_len = max(lengths)
print("*"*100)
print("max len: ")
print(max_len)
print("*"*100)
min_len = min(lengths)
print("min len: ")
print(min_len)
print("*"*100)
print("mean: ")
print(sum(lengths)/len(lengths))
print("*"*100)
print("std: ")
print(np.std(lengths))






