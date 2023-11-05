from datasets import load_dataset
import json

from llmtuner.tuner.core import get_train_args
from llmtuner.dsets import get_dataset, preprocess_dataset
from llmtuner.tuner.core import load_model_and_tokenizer


print("Loading dataset... \n")
# load dataset avoiding timeout and print progress
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



# print("Saving samples... \n")
# with open('data/mediasum.json', 'w') as outfile:
#     json.dump(json_output, outfile, indent=4)
#
#
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
model_args, data_args, training_args, finetuning_args, generating_args, general_args = get_train_args(args)
_, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="sft")

lengths = []
# print("Counting inputs_ids... \n")
for i, example in enumerate(json_output):
    inputs_id = tokenizer.encode(example["instruction"], add_special_tokens=True)
    lengths.append(len(inputs_id))


#
# print("Exploring dataset... \n")
# dataset = get_dataset(model_args, data_args)
# dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="sft")


# console:
# Dataset({
#     features: ['input_ids', 'attention_mask', 'labels'],
#     num_rows: 1
# })

# get all input_ids lengths
# lengths = []
# for i in range(len(dataset)):
#     lengths.append(len(dataset[i]['input_ids']))

# get max length
max_len = max(lengths)
print("*"*100)
print("max len: ")
print(max_len)
print("*"*100)

# get min length
print("*"*100)
min_len = min(lengths)
print("min len: ")
print(min_len)
print("*"*100)

# print mean and std
print("*"*100)
print("mean: ")
print(sum(lengths)/len(lengths))
print("*"*100)
print("std: ")
import numpy as np
print(np.std(lengths))
print("*"*100)






