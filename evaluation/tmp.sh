#!/bin/bash

INFER_FILE="data/VaseVLDataset/VaseVL_Qwen2-VL-2B-Instruct_inference_answers.jsonl"

# --annotation-file use_default_annotation_file
python evaluation/vqa_evaluation.py \
    --infer-file  $INFER_FILE \
    --result-file "results/VaseVL_Qwen2-VL-2B-Instruct/VaseVL_Qwen2-VL-2B-Instruct_evaluation.txt"



INFER_FILE="data/VaseVLDataset/VaseVL_Qwen2.5-VL-3B-Instruct_inference_answers.jsonl"

# --annotation-file use_default_annotation_file
python evaluation/vqa_evaluation.py \
    --infer-file  $INFER_FILE \
    --result-file "results/VaseVL_Qwen2.5-VL-3B-Instruct_SFT/VaseVL_Qwen2.5-VL-3B-Instruct_SFT_evaluation.txt"


#!/bin/bash


INFER_FILE="data/VaseVLDataset/VaseVL_Qwen2.5-VL-3B-Instruct_inference_answers.jsonl"
# --annotation-file use_default_annotation_file
python evaluation/vqa_evaluation.py \
    --infer-file  $INFER_FILE \
    --result-file "results/VaseVL_Qwen2.5-VL-3B-Instruct/VaseVL_Qwen2.5-VL-3B-Instruct_evaluation.txt"
