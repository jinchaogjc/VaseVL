#!/bin/bash

INFER_FILE="results/VaseVL_Qwen2-VL-2B-Instruct/VaseVL_Qwen2-VL-2B-Instruct_inference_answers.jsonl"

python evaluation/vqa_inference.py \
    --model-path "Qwen/Qwen2-VL-2B-Instruct" \
    --question-file "data/VaseVLDataset/vasevl_single_questions.json" \
    --output-file $INFER_FILE \
    --batch-size 80

# --annotation-file use_default_annotation_file
python evaluation/vqa_evaluation.py \
    --infer-file  $INFER_FILE \
    --result-file "results/VaseVL_Qwen2-VL-2B-Instruct/VaseVL_Qwen2-VL-2B-Instruct_evaluation.txt"
