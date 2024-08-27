import os
import json
import argparse
from dschat.utils.metrics import memorization_ratio


parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_file', type=str, required=True,
    help="Path of the file to be inferred.")
args = parser.parse_args()

if "NaturalQuestions" in args.input_file:
    task = "question_answering"
elif "ConflictQA" in args.input_file:
    task = "multi_choice"
else:
    raise NotImplementedError(f"Metrics for dataset {args.input_file} is not implemented")

assert os.path.exists(args.input_file), f"{args.input_file} does not exist"
with open(args.input_file, "r") as f:
    samples = [json.loads(line) for line in f.readlines()]

metrics = memorization_ratio(samples, task=task)
metrics_str = json.dumps(metrics, indent=4)
print(metrics_str)

with open(args.input_file.replace(".json", ".meta"), "w") as f:
    f.write(metrics_str)