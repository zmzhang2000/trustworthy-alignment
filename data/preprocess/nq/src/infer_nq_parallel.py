import os
import subprocess
import argparse
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))


parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_name_or_path', type=str, required=True,
    help="Path to the model checkpoint or its name from huggingface.co/models")
parser.add_argument(
    '--infer_file', type=str, required=True,
    help="Path of the file to be inferred.")
parser.add_argument(
    '--output_file',
    type=str, default=None,
    help="Path of the output file.")
parser.add_argument(
    '--devices',
    type=str, default="0,1,2,3,4,5,6,7",
    help="Devices for each process to run the model on.")
args = parser.parse_args()


if __name__ == '__main__':
    if args.output_file is None:
        args.output_file = os.path.join("outputs", args.infer_file.split("/")[-1].split(".")[0],
                                        args.model_name_or_path.strip("/").split("/")[-2],
                                        args.model_name_or_path.strip("/").split("/")[-1] + ".json")
    output_tmp_dir = os.path.join(os.path.dirname(args.output_file), "tmp")
    os.makedirs(output_tmp_dir, exist_ok=True)

    # Fetch and split the prompts
    with open(args.infer_file) as fin:
        lines = list(fin)
    device_num = len(args.devices.split(","))
    split_ranges = [(indices[0], indices[-1] + 1) for indices in np.array_split(np.arange(len(lines)), device_num)]
    splits = [lines[start: end] for start, end in split_ranges]
    assert len(lines) == sum([len(split) for split in splits])
    for i, split in enumerate(splits):
        with open(os.path.join(output_tmp_dir, f"split_{i}.json"), "w") as fout:
            fout.writelines(split)

    # Run the inference
    processes = []
    try:
        for i, split in enumerate(splits):
            model_name_or_path = args.model_name_or_path
            infer_file = os.path.join(output_tmp_dir, f'split_{i}.json')
            output_file = os.path.join(output_tmp_dir, f'split_{i}_results.json')
            device = f"cuda:{args.devices.split(',')[i]}"
            cmd = f"python infer_nq.py " \
                  f"--model_name_or_path {args.model_name_or_path} " \
                  f"--infer_file {infer_file} " \
                  f"--output_file {output_file} " \
                  f"--device {device}"
            processes.append(subprocess.Popen(cmd, shell=True))
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        print("Ctrl+C pressed. Terminating all shell processes.")
        for process in processes:
            process.kill()
        exit(0)

    # Merge the results
    subprocess.run(f"cat {output_tmp_dir}/split_*_results.json > {args.output_file}", shell=True)
    subprocess.run(f"rm -rf {output_tmp_dir}", shell=True)
