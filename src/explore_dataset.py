import os.path

import joblib
import tqdm
from datasets import load_dataset
import numpy as np


print("Loading dataset... \n")
# load dataset avoiding timeout and print progress
#mediasum_dataset = load_dataset('Salesforce/dialogstudio', 'MediaSum', cache_dir='mnt/c/Users/pyanez/Desktop/pablo/universidad/memoria/src/data/downloads')
mediasum_dataset = load_dataset('Salesforce/dialogstudio', 'MediaSum', cache_dir='data')
#mediasum_dataset = load_dataset('Salesforce/dialogstudio', 'MediaSum', split='validation')
json_output = []
max_samples = 10000
separator = "\n"

print("Generating samples... \n")
print(len(mediasum_dataset["validation"]))
for j, example in enumerate(mediasum_dataset["validation"]):
    prompt = "\n".join([seq for seq in eval(example["original dialog info"])["dialog history"]])
    dialog_sequences = prompt.split('\n')
    context_size = 10
    number_of_sequences = len(dialog_sequences)
    total_windows = number_of_sequences // context_size
    for i in range(total_windows):
        sequences = f"{separator.join(dialog_sequences[i * context_size:(i + 1) * context_size])}"
        json_output.append({"instruction": sequences, "input": "", "output": "", "id": j})
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

lengths = {j:0 for j in list(set([example["id"] for example in json_output]))}
import pandas as pd
# print("Counting inputs_ids... \n")
dialog_index = 0
for example in json_output:
    if example["id"] > dialog_index:
        dialog_index = example["id"]
    inputs_id = tokenizer.encode(example["instruction"], add_special_tokens=True)
    if len(inputs_id) > lengths[dialog_index]:
        lengths[dialog_index] = len(inputs_id)

lengths = [v for v in lengths.values()]
df = pd.DataFrame({"lenghts": lengths})
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
mean_ = sum(lengths)/len(lengths)
print(mean_)
print("*"*100)
print("std: ")
std_ = np.std(lengths)
print(std_)
print("*"*100)
print("skew: ")
print(df.skew())
print("*"*100)
print("kurtosis: ")
print(df.kurt())
print("*"*100)
print("samples: ", len(lengths))

# identify outliers
cut_off = std_ * 3
lower, upper = mean_ - cut_off, mean_ + cut_off
#print(lower, upper)

lengths = [x for x in lengths if lower < x < upper]

df = pd.DataFrame({"lenghts": lengths})
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
mean_ = sum(lengths)/len(lengths)
print(mean_)
print("*"*100)
print("std: ")
std_ = np.std(lengths)
print(std_)
print("*"*100)
print("skew: ")
print(df.skew())
print("*"*100)
print("kurtosis: ")
print(df.kurt())
print("*"*100)
print("filter samples: ", len(lengths))



from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

split = "train"

print("Evaluating clusters... \n")
if not os.path.exists(f"kmeans/summary_embeddings_{split}.csv"):
    df_cluster = []
    for example in tqdm.tqdm(mediasum_dataset[split]):
        summary = eval(example["original dialog info"])["summary"]
        embedding = model.encode(summary)
        df_aux = pd.DataFrame(np.reshape(embedding, (1,-1)), index=[0], columns=range(len(embedding)))
        df_cluster.append(df_aux)

    df_cluster = pd.concat(df_cluster, axis=0).reset_index(drop = True)
    df_cluster.to_csv(f"kmeans/summary_embeddings_{split}.csv", sep=";", index=False)
else:
    df_cluster = pd.read_csv(f"kmeans/summary_embeddings_{split}.csv", sep=";", )

from sklearn.cluster import MiniBatchKMeans, KMeans

find = True

if find:
    SSE = []
    numClusters = [i for i in range(2,30)]
    for k in tqdm.tqdm(numClusters):
        k_means = MiniBatchKMeans(n_clusters=k, n_init=10, random_state=1, batch_size=4096)
        chunk_iter = pd.read_csv(f"kmeans/summary_embeddings_{split}.csv", chunksize=50000)
        for chunk in chunk_iter:
            #chunk = chunk.drop(columns=['image_name', 'class'])
            k_means.partial_fit(chunk)
        SSE.append(k_means.inertia_)
    variation = [(SSE[i] - SSE[i+1])/ SSE[i] * 100 for i in range(len(SSE)-1)]
    n_clusters = numClusters[variation.index(max(variation)) + 1]
    print(f"El número óptimo de clusters es {n_clusters}")
    k_means = KMeans(n_clusters=n_clusters, n_init=10, random_state=1)
    clusters_ = k_means.fit_predict(df_cluster)
    joblib.dump(k_means, "kmeans/model.pkl")
else:
    k_means = joblib.load("kmeans/model.pkl")
    clusters_ = k_means.predict(df_cluster)

print(pd.DataFrame({"clusters": clusters_}).value_counts())


