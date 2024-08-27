import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import argparse
import json
from copy import deepcopy
from dschat.utils.metrics import best_subspan_em
from dschat.utils.utils import load_hf_tokenizer


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
    '--device',
    type=str, default="cuda",
    help="Device to run the model on.")
parser.add_argument(
    "--add_eot_token",
    action='store_true', default=True,
    help="Add <|endoftext|> as additional special token to tokenizer")
args = parser.parse_args()


generation_config = dict(
    do_sample=False,
    top_p=1.0,
    temperature=1.0,
    num_beams=1,
    repetition_penalty=1.2,
    max_new_tokens=64
)


if __name__ == '__main__':
    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None
    tokenizer = load_hf_tokenizer(args.model_name_or_path,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                 device_map="auto" if args.device == "cuda" else args.device,
                                                 torch_dtype="auto", use_flash_attention_2=True)
    model.eval()
    print("Load model successfully")
    
    if args.output_file is None:
        args.output_file = os.path.join("outputs", args.infer_file.split("/")[-1].split(".")[0],
                                        args.model_name_or_path.strip("/").split("/")[-2],
                                        args.model_name_or_path.strip("/").split("/")[-1] + ".json")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Fetch all of the prompts
    with open(args.infer_file) as fin:
        lines = list(fin)

    idx = 0
    num_correct = 0
    with open(args.output_file, "w") as f:
        for line in (tq:=tqdm(lines, desc="Best subspan EM:  0.0%")):
            input_example = json.loads(line)
            # Get the prediction for the input example
            prompt = input_example["text"]

            inputs = tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt"
            )
            generation_output = model.generate(
                input_ids = inputs["input_ids"].to(args.device), 
                **generation_config
            )[0]

            generate_text = tokenizer.decode(generation_output,skip_special_tokens=True)
            response = generate_text[len(prompt):].strip()

            output_example = deepcopy(input_example)
            output_example["response"] = response

            # NOTE: we take everything up to the first newline, since otherwise models could hack
            # the metric by simply copying te input context (as the gold answer is guaranteed
            # to occur in the input context).
            ground_truths = input_example["right_answer"].split(" OR ")
            responses = output_example["response"].split("\n")[0].strip().split(" OR ")
            example_metrics = max([best_subspan_em(prediction=response, ground_truths=ground_truths) for response in responses])
            output_example["best_subspan_em"] = example_metrics
            if abs(example_metrics - 1.0) < 1e-6:
                num_correct += 1
            idx += 1
            tq.set_description(f"Best subspan EM: {num_correct / idx * 100:4.1f}%")

            f.write(json.dumps(output_example) + "\n")
    
    with open(args.output_file.replace(".json", ".meta"), "w") as f:
        f.write(f"\nBest subspan EM: {num_correct / idx * 100:4.1f}%\n")
