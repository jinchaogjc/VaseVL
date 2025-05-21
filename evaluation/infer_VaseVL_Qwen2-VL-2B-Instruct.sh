
python evaluation/vqa_inference.py \
    --model-path "Qwen/Qwen2-VL-2B-Instruct" \
    --image-dir "data/VaseVLDataset/images" \
    --question-file "data/VaseVLDataset/vasevl_single_questions.json" \
    --output-file "data/VaseVLDataset/VaseVL_Qwen2-VL-2B-Instruct_inference_answers.jsonl" \
    --batch-size 1