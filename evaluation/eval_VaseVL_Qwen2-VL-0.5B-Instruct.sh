#!/bin/bash

INFER_FILE="results/VaseVL_Qwen2-VL-0.5B/VaseVL_Qwen2-VL-0.5B_inference_answers.jsonl"

python evaluation/vqa_inference.py \
    --model-path "/inspire/hdd/ws-ba572160-47f8-4ca1-984e-d6bcdeb95dbb/a100-maybe/wangbaode/NIPS_2025/Codes/models/llava-onevision-qwen2-0.5b-ov-hf" \
    --question-file "data/VaseVLDataset/vasevl_single_questions_no_tag.json" \
    --output-file $INFER_FILE \
    --batch-size 80

# --annotation-file use_default_annotation_file
python evaluation/vqa_evaluation.py \
    --annotation-file data/VaseVLDataset/vasevl_single_gt_answers_no_tag.json \
    --infer-file  $INFER_FILE \
    --result-file "results/VaseVL_Qwen2-VL-0.5B/VaseVL_Qwen2-VL-0.5B_evaluation.txt"
