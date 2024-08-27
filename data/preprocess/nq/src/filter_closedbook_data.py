import os
import gzip
import json


original_data_dir = "datasets/original"
filenames = ["MRQANaturalQuestionsTrain.jsonl.gz", "MRQANaturalQuestionsDev.jsonl.gz"]
closedbook_data_dir = "datasets/closedbook_infer_output"

for filename in filenames:
    with open(os.path.join(closedbook_data_dir, filename.replace("jsonl.gz", "json")), "r") as f:
        answer_data = [json.loads(line) for line in f.readlines()]
        qids2correct = {sample["id"]: abs(sample["best_subspan_em"] - 1.0) < 1e-6 for sample in answer_data}

    output_list = []
    with gzip.open(os.path.join(original_data_dir, filename), "rb") as f:
        output_list.append(f.readline())
        for line in f:
            entry = json.loads(line)
            for qa in entry["qas"]:
                if qa["qid"] in qids2correct and qids2correct[qa["qid"]]:
                    output_list.append(line)

    with gzip.open(os.path.join(original_data_dir, filename.replace(".jsonl.gz", "-closedbookfiltered.jsonl.gz")), "wb") as f:
        f.writelines(output_list)