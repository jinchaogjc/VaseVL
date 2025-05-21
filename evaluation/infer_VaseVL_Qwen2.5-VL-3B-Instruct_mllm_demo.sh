
python vqa_inference.py \
    --model-path "/inspire/hdd/ws-ba572160-47f8-4ca1-984e-d6bcdeb95dbb/a100-maybe/wangbaode/Codes/LLaMA-Factory/output/qwen2_5vl_3b_lora_sft" \
    --image-dir "../data/VaseVLDataset/images" \
    --question-file "../data/VaseVLDataset/VaseVL_question_50.json" \
    --output-file "../data/VaseVLDataset/VaseVL_Qwen2.5-VL-3B-Instruct_mllm_demo_inference_answers.jsonl" \
    --batch-size 1
