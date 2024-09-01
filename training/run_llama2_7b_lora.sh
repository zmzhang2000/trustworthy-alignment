ACTOR_MODEL_PATH=meta-llama/Llama-2-7b-chat-hf
CRITIC_MODEL_PATH=AdamG012/chat-opt-350m-reward-deepspeed
ACTOR_ZERO_STAGE=$3
CRITIC_ZERO_STAGE=$4
OUTPUT=$5
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./outputs/llama2_7b_main
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=2
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=2
fi
mkdir -p $OUTPUT

Actor_Lr=9.65e-6
Critic_Lr=5e-6

deepspeed --master_port 13680 --include "localhost:0,1,2,3,4,5,6,7" main.py \
   --data_path ../data/MRQANaturalQuestionsSPLIT-closedbookfiltered-corpus-counterfactual.json \
   --data_output_path /tmp/zhangzm_data_files \
   --data_split 0,0,1 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 1 \
   --per_device_training_batch_size 1 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --eval_steps 200 \
   --max_answer_seq_len 64 \
   --max_prompt_seq_len 1024 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --actor_dropout 0.0 \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --enable_hybrid_engine \
   --actor_lora_dim 64 \
   --critic_lora_dim 64 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir $OUTPUT \
   --dtype bf16 \
   --print_answers \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT/tensorboard \
    2>&1 | tee $OUTPUT/training.log
