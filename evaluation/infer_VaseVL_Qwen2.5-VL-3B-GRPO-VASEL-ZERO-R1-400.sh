
python evaluation/vqa_inference_r1.py \
    --model-path "/inspire/hdd/ws-ba572160-47f8-4ca1-984e-d6bcdeb95dbb/a100-maybe/wangbaode/NIPS_2025/Agent/VLM-R1-Safety/src/open-r1-multimodal/output/Qwen2.5-VL-3B-GRPO-VASEL-ZERO-R1-400" \
    --image-dir "data/VaseVLDataset/images" \
    --question-file "data/VaseVLDataset/vasevl_single_questions.json" \
    --output-file "data/VaseVLDataset/VaseVL_Qwen2.5-VL-3B-GRPO-VASEL-ZERO-R1-400_inference_answers.jsonl" \
    --batch-size 1
