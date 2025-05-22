
python evaluation/vqa_inference.py \
    --model-path "/inspire/hdd/ws-ba572160-47f8-4ca1-984e-d6bcdeb95dbb/a100-maybe/wangbaode/Codes/LLaMA-Factory/saves/qwen2_5vl-3b-vasevl_dataset/full/sft/checkpoint-18/" \
    --image-dir "data/VaseVLDataset/images" \
    --question-file "data/VaseVLDataset/vasevl_single_questions.json" \
    --output-file "data/VaseVLDataset/VaseVL_Qwen2.5-VL-3B-Instruct_vasevl_dataset_inference_answers.jsonl" \
    --batch-size 80
