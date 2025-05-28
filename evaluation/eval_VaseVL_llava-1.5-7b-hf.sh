#!/bin/bash

INFER_FILE="results/VaseVL_llava-1.5-7b-hf/VaseVL_llava-1.5-7b-hf_inference_answers.jsonl"

python evaluation/vqa_inference.py \
    --model-path "llava-hf/llava-1.5-7b-hf" \
    --question-file "data/VaseVLDataset/vasevl_single_questions_no_tag.json" \
    --output-file $INFER_FILE \
    --batch-size 1

# --annotation-file use_default_annotation_file
python evaluation/vqa_evaluation.py \
    --annotation-file data/VaseVLDataset/vasevl_single_gt_answers_no_tag.json \
    --infer-file  $INFER_FILE \
    --result-file "results/VaseVL_llava-1.5-7b-hf_evaluation.txt"
