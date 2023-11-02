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
import subprocess
import os


def generate_prediction(sample_data, model_script='src/train_bash.py', dataset_path='data', output_dir='predictions'):
    """
    Generates a prediction for the current dialog context.
    """
    aux_data_path = os.path.join(dataset_path, 'mediasum-aux.json')

    # Save current sample to a json file
    with open(aux_data_path, 'w') as aux_file:
        json.dump(sample_data, aux_file, indent=4)

    # Run the prediction script
    subprocess.run([
        'CUDA_VISIBLE_DEVICES=0', 'python', model_script,
        '--stage', 'sft',
        '--model_name_or_path', 'meta-llama/Llama-2-7b-chat-hf',
        '--do_predict',
        '--dataset', 'mediasum-aux',
        '--dataset_dir', dataset_path,
        '--template', 'llama2',
        '--output_dir', output_dir,
        '--per_device_eval_batch_size', '1',
        '--max_samples', '1',
        '--predict_with_generate',
        '--cutoff_len', '4000'
    ], check=True)

    # Read the generated prediction
    prediction_file = os.path.join(output_dir, 'generated_predictions.jsonl')
    with open(prediction_file, 'r') as pred_file:
        prediction = json.loads(pred_file.readline())
    return prediction['predict']


dataset = load_dataset('Salesforce/dialogstudio', 'MediaSum')
sub_sample = 0
json_output = []

for j, example in enumerate(dataset["train"]):
    prompt = "\n".join([seq for seq in eval(example["original dialog info"])["dialog history"]])
    dialog_sequences = prompt.split('\n')
    context_size = 10
    number_of_sequences = len(dialog_sequences)
    total_windows = number_of_sequences // context_size
    history = []
    for i in range(total_windows):
        state = "continue" if i > 0 and i < total_windows - 1 else "end" if i == total_windows - 1 else "start"
        if i > 0:
            context = "\n".join(history)
        else:
            context = ""
        instruction = f"Summarize the following dialogue between '###', make a brief summary on what happens in it, " \
                              f"DON'T complete it just summarize it. " \
                              f"You'll be given the dialog state and the previous dialog context, which you have to adapt your summary to. " \
                              f"If the dialog state is 'start' you won't be given any context. " \
                              f"If the dialog state is 'continue' you'll be given the previous dialog contexts, your summary MUST be coherent, " \
                              f"consistent and MUST NOT repeat information in the previous contexts. " \
                              f"Your answer MUST be in JSON format, with the following structure: " \
                              f"{{'summary': 'your summary or continuation'}} " \
                              f"where 'your summary' is a string with your summary. " \
                              f"The summary MUST be at maximum 100 words long. " \
                              f"\n\n Dialog state: {state} " \
                              f"\n\n Context: {context} " \
                              f"\n\n Conversation: " \
                              f"###\n\n  {' '.join(dialog_sequences[i * context_size:(i + 1) * context_size])} \n\n ###"
        output = eval(example["original dialog info"])["summary"]
        json_output.append({"instruction": instruction, "input": "", "output": output, "id": j, "state": state})
        if state == "continue":
            history.append(generate_prediction(json_output[-1]))
        elif state == "end":
            history = []
        if j == sub_sample:
            break

# Define the file path where you want to save the JSON file
output_file_path = 'data/mediasum-sampled.json'

# Write the `json_output` to a JSON file
with open(output_file_path, 'w') as json_file:
    json.dump(json_output, json_file, indent=4)

print(f'JSON data has been saved to {output_file_path}')



