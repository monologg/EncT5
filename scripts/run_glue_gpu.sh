#!/bin/bash
MODEL_NAME=google/t5-v1_1-base

TASK_NAME=cola

python3 run_glue.py \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 5e-5 \
    --adafactor \
    --num_train_epochs 20 \
    --logging_steps 100 \
    --output_dir eval_outputs/$TASK_NAME \
    --overwrite_output_dir \
    --cache_dir cache \
    --seed 42 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_matthews_correlation \
    --greater_is_better True \
    --early_stopping_patience 5

rm -rf eval_outputs/*/checkpoint*
    
TASK_NAME=sst2

python3 run_glue.py \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs 20 \
    --logging_steps 100 \
    --output_dir eval_outputs/$TASK_NAME \
    --overwrite_output_dir \
    --cache_dir cache \
    --seed 42 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_accuracy \
    --greater_is_better True \
    --early_stopping_patience 5

TASK_NAME=mrpc

python3 run_glue.py \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs 20 \
    --logging_steps 100 \
    --output_dir eval_outputs/$TASK_NAME \
    --overwrite_output_dir \
    --cache_dir cache \
    --seed 42 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_f1 \
    --greater_is_better True \
    --early_stopping_patience 5

rm -rf eval_outputs/*/checkpoint*

TASK_NAME=stsb

python3 run_glue.py \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs 20 \
    --logging_steps 100 \
    --output_dir eval_outputs/$TASK_NAME \
    --overwrite_output_dir \
    --cache_dir cache \
    --seed 42 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_combined_score \
    --greater_is_better True \
    --early_stopping_patience 5

rm -rf eval_outputs/*/checkpoint*

TASK_NAME=qqp

python3 run_glue.py \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs 20 \
    --logging_steps 100 \
    --output_dir eval_outputs/$TASK_NAME \
    --overwrite_output_dir \
    --cache_dir cache \
    --seed 42 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_accuracy \
    --greater_is_better True \
    --early_stopping_patience 5

rm -rf eval_outputs/*/checkpoint*

TASK_NAME=mnli

python3 run_glue.py \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs 20 \
    --logging_steps 100 \
    --output_dir eval_outputs/$TASK_NAME \
    --overwrite_output_dir \
    --cache_dir cache \
    --seed 42 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_accuracy \
    --greater_is_better True \
    --early_stopping_patience 5

rm -rf eval_outputs/*/checkpoint*

TASK_NAME=qnli

python3 run_glue.py \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs 20 \
    --logging_steps 100 \
    --output_dir eval_outputs/$TASK_NAME \
    --overwrite_output_dir \
    --cache_dir cache \
    --seed 42 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_accuracy \
    --greater_is_better True \
    --early_stopping_patience 5

rm -rf eval_outputs/*/checkpoint*

TASK_NAME=rte

python3 run_glue.py \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs 20 \
    --logging_steps 100 \
    --output_dir eval_outputs/$TASK_NAME \
    --overwrite_output_dir \
    --cache_dir cache \
    --seed 42 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_accuracy \
    --greater_is_better True \
    --early_stopping_patience 5

rm -rf eval_outputs/*/checkpoint*
