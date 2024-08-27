# Data Preprocessing for Natural Questions

This directory contains the scripts to preprocess the Natural Questions dataset.

## Steps

1. Setup the environment by running the following command:
```bash
conda create -n kc -y python==3.8
source activate kc
bash setup.sh
```

2. Preprocess the data by running the following command:
```bash
export PYTHONPATH=.
python src/download_datasets.py
python src/generate_closedbook_input.py

source activate base
python src/infer_nq_parallel.py --model_name_or_path ../../../ckpt/huggingface/meta-llama/Llama-2-7b-chat-hf --infer_file `pwd`/datasets/closedbook_infer_input/MRQANaturalQuestionsTrain.json --output_file `pwd`/datasets/closedbook_infer_output/MRQANaturalQuestionsTrain.json --devices=0,1,2,3,4,5,6,7
python src/infer_nq_parallel.py --model_name_or_path ../../../ckpt/huggingface/meta-llama/Llama-2-7b-chat-hf --infer_file `pwd`/datasets/closedbook_infer_input/MRQANaturalQuestionsDev.json --output_file `pwd`/datasets/closedbook_infer_output/MRQANaturalQuestionsDev.json --devices=0,1,2,3,4,5,6,7

source activate kc
python src/filter_closedbook_data.py
python src/normalize.py

SUB_TYPE=corpus  # alias, popularity, corpus, type-swap. See https://github.com/apple/ml-knowledge-conflicts
python src/generate_substitutions.py --inpath datasets/normalized/MRQANaturalQuestionsTrain-closedbookfiltered.jsonl.gz --outpath datasets/substitution-sets/MRQANaturalQuestionsTrain-closedbookfiltered-$SUB_TYPE.jsonl $SUB_TYPE-substitution
python src/generate_substitutions.py --inpath datasets/normalized/MRQANaturalQuestionsDev-closedbookfiltered.jsonl.gz --outpath datasets/substitution-sets/MRQANaturalQuestionsDev-closedbookfiltered-$SUB_TYPE.jsonl $SUB_TYPE-substitution

python src/generate_counterfactual_data.py MRQANaturalQuestionsTrain-closedbookfiltered-$SUB_TYPE.jsonl
python src/generate_counterfactual_data.py MRQANaturalQuestionsDev-closedbookfiltered-$SUB_TYPE.jsonl

mv MRQANaturalQuestions*-closedbookfiltered-$SUB_TYPE.jsonl ../..
```