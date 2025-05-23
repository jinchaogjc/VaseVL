#!/bin/bash

INFER_FILE="results/VaseVL_Qwen2.5-VL-7B-Instruct/VaseVL_Qwen2.5-VL-7B-Instruct_inference_answers.jsonl"
python evaluation/vqa_inference.py \
    --model-path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --question-file "data/VaseVLDataset/vasevl_single_questions.json" \
    --output-file $INFER_FILE \
    --batch-size 80
saves/qwen2_5vl_7b_full_sft_vasevl_dataset/full/sft

# --annotation-file use_default_annotation_file
python evaluation/vqa_evaluation.py \
    --infer-file  $INFER_FILE \
    --result-file "results/VaseVL_Qwen2.5-VL-7B-Instruct/VaseVL_Qwen2.5-VL-7B-Instruct_evaluation.txt"