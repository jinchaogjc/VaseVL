
python vqa_inference.py \
    --model-path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --image-dir "../data/VaseVLDataset/images" \
    --question-file "../data/VaseVLDataset/VaseVL_question_50.json" \
    --output-file "../data/VaseVLDataset/VaseVL_Qwen2.5-VL-3B-Instruct_inference_answers.jsonl" \
    --batch-size 1
