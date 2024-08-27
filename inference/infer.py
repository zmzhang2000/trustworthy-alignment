import os
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import argparse
import json
from dschat.utils.utils import load_hf_tokenizer
from dschat.utils.data.raw_datasets import MRQANaturalQuestionsDataset, ConflictQADataset


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
    '--device',
    type=str, default="cuda",
    help="Device to run the model on.")
parser.add_argument(
    "--num_shards",
    type=int, default=1,
    help="Number of shards to split the dataset into.")
parser.add_argument(
    "--shard_id",
    type=int, default=0,
    help="ID of the shard to run the inference on.")
parser.add_argument(
    "--add_eot_token",
    action='store_true', default=True,
    help="Add <|endoftext|> as additional special token to tokenizer")
parser.add_argument(
    "--prompt_template",
    type=str, default=None,
    help="Prompt template to use for inference")
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
    if "NaturalQuestions" in args.data_path:
        dataset = MRQANaturalQuestionsDataset(None, None, None, args.data_path, task="question_answering", prompt_template=args.prompt_template).get_eval_data()
    elif "ConflictQA" in args.data_path:
        dataset = ConflictQADataset(None, 0, None, args.data_path, task="multi_choice", prompt_template=args.prompt_template).get_eval_data()
    else:
        raise NotImplementedError("Dataset not supported")
    
    # only infer on a shard of the dataset
    assert args.num_shards > 0 and args.shard_id < args.num_shards
    shard_indices = np.array_split(np.arange(len(dataset)), args.num_shards)[args.shard_id]
    dataset = dataset.select(shard_indices)

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None
    tokenizer = load_hf_tokenizer(args.model_name_or_path,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)

    use_flash_attention_2 = args.device != "cpu" and "llama" in args.model_name_or_path.lower()
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                 device_map="auto" if args.device == "cuda" else args.device,
                                                 torch_dtype="auto", use_flash_attention_2=use_flash_attention_2)
    model.eval()
    if args.device == "cpu":
        model = model.float()
    print("Load model successfully")
    
    if args.output_file is None:
        args.output_file = os.path.join("outputs", args.data_path.split("/")[-1].split(".")[0],
                                        args.model_name_or_path.strip("/").split("/")[-2],
                                        args.model_name_or_path.strip("/").split("/")[-1] + ".json")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, "w") as f:
        for sample in tqdm(dataset):
            prompt = sample["text"]

            inputs = tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt"
            )
            generation_output = model.generate(
                input_ids = inputs["input_ids"].to(args.device), 
                **generation_config
            )[0]

            sample["response"] = tokenizer.decode(generation_output,skip_special_tokens=True)[len(prompt):]
            f.write(json.dumps(sample) + "\n")
            f.flush()
