#!/bin/bash

set -e
set -x

size=$1
dataset=$2
flair_batch_size=$3
SEED=$4
generations=$5

directory="../datasets-precompute/${dataset}/${size}"

attn_train="train_processed"
attn_dev="dev_processed"

run="${generations}-${size}-${SEED}-${shouldLinearizeAllWords}-tokenfix"

python flair_train.py \
--input_folder $directory \
--output_folder "${directory}/consistency" \
--gpu cuda:0 \
--train_file "${attn_train}.txt" \
--batch_size $flair_batch_size \
--lr 0.01 \
--epochs 100 \
--seed $SEED \
--model dmis-lab/biobert-large-cased-v1.1

python pretrain_dynamic.py \
--directory $directory \
--train_file $attn_train \
--dev_file $attn_dev \
--epochs 10 \
--batch_size 16 \
--file_name $run \
--seed $SEED \

best_model="${directory}/${attn_train}-${run}-final"

inference_file="train_processed"

python test-dynamic.py \
--model $best_model \
--input_file $inference_file \
--sample_generation_mode dynamic \
--directory $directory \
--topk 10 \
--num_of_sequences $generations \
--max_length 200 \
--do_sample True \
--num_beams 5 \
--file_name $run \
--root_dir $directory \
--remove_repetitions False \
--seed $SEED

generated_file="${inference_file}-${run}"

python flair_eval_equal.py \
--input_folder $directory \
--output_folder "${directory}/${generated_file}" \
--gpu cuda:0 \
--input_file $generated_file \
--need_consistency False \
--file_name $run \
--seed $SEED \
--gold_file $inference_file \
--checkpoint "${directory}/consistency/best-model.pt"

consistent_file="${generated_file}-aug+gold.txt"

python flair_train.py \
--input_folder $directory \
--output_folder "${directory}/${generated_file}-flair" \
--gpu cuda:0 \
--train_file $consistent_file \
--batch_size $flair_batch_size \
--lr 0.01 \
--epochs 100 \
--seed $SEED \
--model dmis-lab/biobert-large-cased-v1.1
