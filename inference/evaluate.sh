set -e


MODEL_NAME_OR_PATH=../training/outputs/llama2_7b_main/actor
DATA_PATH=../data/MRQANaturalQuestionsSPLIT-closedbookfiltered-corpus-counterfactual.json
OUTPUT_FILE=outputs/results.json
PROMPT_TEMPLATE=instruction-based

python infer_parallel.py --model_name_or_path $MODEL_NAME_OR_PATH --data_path $DATA_PATH --output_file $OUTPUT_FILE --prompt_template $PROMPT_TEMPLATE
python evaluate.py --input_file $OUTPUT_FILE
