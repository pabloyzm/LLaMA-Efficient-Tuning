# from datasets import load_dataset
#
# dataset = load_dataset('Salesforce/dialogstudio', 'MediaSum')
#
# sub_sample = 0
#
# json_output = []
# for j, example in enumerate(dataset["train"]):
#     prompt = "\n".join([seq for seq in eval(example["original dialog info"])["dialog history"]])
#     dialog_sequences = prompt.split('\n')
#     context_size = 10
#     number_of_sequences = len(dialog_sequences)
#     total_windows = number_of_sequences//context_size
#     for i in range(total_windows):
#         if i > 0 and i < total_windows-1:
#             state = "continue"
#         elif i == 0:
#             state = "start"
#         else:
#             state = "end"
#         context = "" # for now
#         instruction = f"Summarize the following dialogue between '###', make a brief summary on what happens in it, " \
#                         f"DON'T complete it just summarize it. " \
#                         f"You'll be given the dialog state and the previous dialog history, which you have to adapt your summary to. " \
#                         f"Your answer MUST be in JSON format, with the following structure: " \
#                         f"{{'summary': 'your summary'}} " \
#                         f"where 'your summary' is a string with your summary. " \
#                         f"The summary MUST be at maximum 100 words long. " \
#                         f"\n\n Dialog state: {state} " \
#                         f"\n\n Context: {context} " \
#                         f"\n\n Conversation: " \
#                         f"###\n\n  {' '.join(dialog_sequences[i*context_size:(i+1)*context_size])} \n\n ###"
#         output = eval(example["original dialog info"])["summary"]
#         json_output.append({"instruction": instruction, "input": "", "output": output, "id": j, "state": state})
#     if j == sub_sample:
#         break
#
#
# # Define the file path where you want to save the JSON file
# output_file_path = 'data/mediasum-sampled.json'
#
# import json
#
# # Write the `json_output` to a JSON file
# with open(output_file_path, 'w') as json_file:
#     json.dump(json_output, json_file, indent=4)
#
# print(f'JSON data has been saved to {output_file_path}')
#

from datasets import load_dataset
import json
import os



from llmtuner.tuner.core import get_train_args
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments

from llmtuner.dsets import get_dataset, preprocess_dataset, split_dataset
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.misc import get_logits_processor
from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.tuner.sft.metric import ComputeMetrics
from llmtuner.tuner.sft.trainer import CustomSeq2SeqTrainer


dataset_path= 'data'
output_dir= 'predictions'

args = {
    "stage": "sft",
    "model_name_or_path": "meta-llama/Llama-2-7b-chat-hf",
    "do_predict": True,
    "dataset": "mediasum-aux",
    "dataset_dir": dataset_path,
    "template": "llama2",
    "output_dir": output_dir,
    "per_device_eval_batch_size": 1,
    "max_samples": 1,
    "predict_with_generate": True,
    "cutoff_len": 4000
}
model_args, data_args, training_args, finetuning_args, generating_args, general_args = get_train_args(args)
model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="sft")

if training_args.predict_with_generate:
    tokenizer.padding_side = "left"  # use left-padding in generation

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    pad_to_multiple_of=4,  # for shift short attention
    label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
)

# Override the decoding parameters of Seq2SeqTrainer
training_args_dict = training_args.to_dict()
training_args_dict.update(dict(
    generation_max_length=training_args.generation_max_length or data_args.cutoff_len,
    generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams
))
training_args = Seq2SeqTrainingArguments(**training_args_dict)

# Initialize our Trainer
trainer = CustomSeq2SeqTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=None,
    compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
    **{"eval_dataset": None}
)

# Keyword arguments for `model.generate`
gen_kwargs = generating_args.to_dict()
gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
gen_kwargs["logits_processor"] = get_logits_processor()


def generate_prediction(sample_data, model_script='src/train_bash.py', dataset_path='data', output_dir='predictions'):
    global model, tokenizer, model_args, data_args, training_args, generating_args
    """
    Generates a prediction for the current dialog context.
    """
    aux_data_path = os.path.join(dataset_path, 'mediasum-aux.json')

    # Save current sample to a json file
    with open(aux_data_path, 'w') as aux_file:
        json.dump(sample_data, aux_file, indent=4)

    dataset = get_dataset(model_args, data_args)
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="sft")

    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate: # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        result = trainer.get_predictions(predict_results)

    # return the prediction encoded in <>
    #start_index = result.find('<')
    # end_index = result.find('>')
    # result = result[0 : end_index]
    # print(result)
    return result.replace("Summary:", "").strip()

print("Loading dataset... \n")
# load dataset avoiding timeout and print progress
dataset = load_dataset('Salesforce/dialogstudio', 'MediaSum', cache_dir='data')
#dataset = load_dataset('Salesforce/dialogstudio', 'MediaSum')
sub_sample = 0
json_output = []

print("Generating samples... \n")
for j, example in enumerate(dataset["train"]):
    prompt = "\n".join([seq for seq in eval(example["original dialog info"])["dialog history"]])
    dialog_sequences = prompt.split('\n')
    context_size = 10
    number_of_sequences = len(dialog_sequences)
    total_windows = number_of_sequences // context_size
    for i in range(total_windows):
        # the state must be start, continue or end depending on the position of the window
        state = "continue" if i > 0 and i < total_windows - 1 else "start" if i == 0 else "end"
        instruction = f"Summarize the following dialogue between '###', make a brief summary on what happens in it, " \
                              f"DON'T complete it just summarize it. " \
                              f"You'll be given the dialog state. " \
                              f"Your answer MUST be in this format: " \
                              f"\n\n Summary: ### your summary ### " \
                              f"The Summary MUST be at maximum 100 words long. " \
                              f"Your answer MUST contain the summary and ONLY the summary. " \
                              f"Do NOT include any other information. " \
                              f"\n\n Dialog state: {state} " \
                              f"\n\n Conversation: " \
                              f"###\n\n  {' '.join(dialog_sequences[i * context_size:(i + 1) * context_size])} \n\n ###" \
                              # f"\n\n Previous summaries: ''' {' '.join(pre for pre in history) if len(history) > 0 else ''} ''' "
        # output = eval(example["original dialog info"])["summary"]
        res_ = generate_prediction({"instruction": instruction, "input": "", "output": ""})
        json_output.append({"instruction": instruction, "input": "", "id": j, "state": state, "output": res_})
    if j == sub_sample:
        break

# Define the file path where you want to save the JSON file
output_file_path = 'data/mediasum-sampled.json'

# Write the `json_output` to a JSON file
with open(output_file_path, 'w') as json_file:
    json.dump(json_output, json_file, indent=4)

print(f'JSON data has been saved to {output_file_path}')



