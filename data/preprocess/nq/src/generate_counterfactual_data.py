# Generate counterfactual datasets
# Datasets is named by {name}-{closedbookfiltered}-{substitution_type}-substituted.json
# All samples in the dataset can be answered right in closedbook QA, and can be substituted by substitution_type

import sys
import json
import gzip
from tqdm import tqdm


root_dir = "datasets"
output_dir = "../../"

filename = sys.argv[1]

dataset_name = filename.split('.jsonl')[0]
dataset = dict()
with open(f"{root_dir}/substitution-sets/{filename}") as f:
    next(f)
    for line in f:
        sample = json.loads(line)
        dataset[sample["original_example"]] = {
            "qid": sample["original_example"],
            "question": sample["query"],
            "substituted_context": sample["context"],
            "substituted_answers": [x["text"] for x in sample["gold_answers"]]
        }

original_filename = filename.split("-")[0] + ".jsonl.gz"
with gzip.open(f"{root_dir}/original/{original_filename}", "rb") as f:
    next(f)
    for line in f:
        sample = json.loads(line)
        assert len(sample["qas"]) == 1, "len(sample['qas']) != 1"
        qid = sample["qas"][0]["qid"]
        if qid in dataset:
            assert qid == dataset[qid]["qid"]
            assert sample["qas"][0]["question"] == dataset[qid]["question"]
            dataset[qid]["original_context"] = sample["context"]
            dataset[qid]["original_answers"] = sample["qas"][0]["answers"]

with open(f"{output_dir}/{dataset_name}-counterfactual.json", "w") as f:
    for qid, sample in tqdm(dataset.items()):
        f.write(json.dumps(sample) + "\n")
