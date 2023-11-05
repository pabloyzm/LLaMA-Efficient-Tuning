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
max_samples = 10

print("Generating samples... \n")
for j, example in enumerate(mediasum_dataset["validation"]):
    prompt = "\n".join([seq for seq in eval(example["original dialog info"])["dialog history"]])
    json_output.append({"instruction": prompt, "input": "", "output": "", "id": j})
    if j == max_samples:
        break

print("Saving samples... \n")
with open('data/mediasum.json', 'w') as outfile:
    json.dump(json_output, outfile, indent=4)


dataset_path= 'data'
output_dir= 'predictions'
target_dataset = "mediasum"

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

print("Exploring dataset... \n")
dataset = get_dataset(model_args, data_args)
dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="sft")

print(dataset)




