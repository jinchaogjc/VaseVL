
python vqa_inference.py \
    --model-path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --image-dir "../data/VaseVLDataset_sub/images" \
    --question-file "../data/VaseVLDataset_sub/VaseVL_question.json" \
    --output-file "../data/VaseVLDataset_sub/VaseVL_Qwen2.5-VL-3B-Instruct_inference_answers.jsonl" \
    --batch-size 1