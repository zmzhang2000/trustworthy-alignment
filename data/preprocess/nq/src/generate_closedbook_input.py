import os
import json
import gzip
from tqdm import tqdm
from copy import deepcopy


root_dir = "datasets"
output_dir = "datasets/closedbook_infer_input"
os.makedirs(output_dir, exist_ok=True)

# generate closed-book original train/dev set
def generate_closed_book_prompt(question):
    prompt = f'Q: {question}?\nA:'
    return prompt

filenames = os.listdir(f"{root_dir}/original")
for filename in filenames:
    with gzip.open(f"{root_dir}/original/{filename}", "rb") as f:
        next(f)
        dataset = []
        for line in f:
            data = json.loads(line)
            dataset.append(data)

    # remove duplicate samples
    recorded = dict()
    deduplicated_dataset = []
    for data in dataset:
        data_wo_qid = deepcopy(data)
        assert len(data_wo_qid["qas"]) == 1, "len(data_wo_qid['qas']) != 1"
        data_wo_qid['qas'][0].pop('qid')
        data_str = json.dumps(data_wo_qid)
        if data_str not in recorded:
            recorded[data_str] = True
            deduplicated_dataset.append(data)
        else:
            print(f"Duplicate sample qid: {data['qas'][0]['qid']}")
    dataset = deduplicated_dataset
    del recorded, deduplicated_dataset

    dataset = [
        {"qid": data["qas"][0]["qid"],
         "question": data["qas"][0]["question"],
         "context": data["context"],
         "short_answers": data["qas"][0]["answers"]}
         for data in dataset
    ]

    dataset_name = filename.split('.jsonl.gz')[0]
    with open(f"{output_dir}/{dataset_name}.json", "w") as f:
        for idx, sample in enumerate(tqdm(dataset), start=1):
            prompt = generate_closed_book_prompt(sample["question"])
            response = " OR ".join(sample['short_answers'])
            sample = {"id": sample["qid"], "text": prompt, "right_answer": response}
            f.write(json.dumps(sample) + "\n")
