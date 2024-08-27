import os
import subprocess
import argparse
os.chdir(os.path.dirname(os.path.abspath(__file__)))


parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_name_or_path', type=str, required=True,
    help="Path to the model checkpoint or its name from huggingface.co/models")
parser.add_argument(
    '--data_path', type=str, required=True,
    help="Name or path of the data to be inferred.")
parser.add_argument(
    '--output_file',
    type=str, default=None,
    help="Path of the output file.")
parser.add_argument(
    '--devices',
    type=str, default="0,1,2,3,4,5,6,7",
    help="Devices for each process to run the model on.")
parser.add_argument(
    "--prompt_template",
    type=str, default=None,
    help="Prompt template to use for inference")
args = parser.parse_args()


if __name__ == '__main__':
    if args.output_file is None:
        args.output_file = os.path.join("outputs", args.data_path.split("/")[-1].split(".")[0],
                                        args.model_name_or_path.strip("/").split("/")[-2],
                                        args.model_name_or_path.strip("/").split("/")[-1] + ".json")
    output_tmp_dir = os.path.join(os.path.dirname(args.output_file), "tmp")
    os.makedirs(output_tmp_dir, exist_ok=True)

    devices = [i for i in args.devices.split(",")]
    if "13b" in args.model_name_or_path.lower():
        devices = [",".join(devices[i:i+2]) for i in range(0, len(devices), 2)]
    num_shards = len(devices)
    processes = []
    try:
        for i in range(num_shards):
            cmd = f"CUDA_VISIBLE_DEVICES={devices[i]} " \
                  f"python infer.py " \
                  f"--model_name_or_path {args.model_name_or_path} " \
                  f"--data_path {args.data_path} " \
                  f"--output_file {output_tmp_dir}/split_{i}_results.json " \
                  f"--num_shards {num_shards} " \
                  f"--shard_id {i} " \
                  f"--prompt_template {args.prompt_template} " \
                  f"2>&1 | tee {output_tmp_dir}/{i}.log"
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
