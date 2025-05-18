import os
import json
import re
from accuracy_valuator import TextMatchEvaluator, TextCapsBleu4Evaluator


def eval_single(annotation_file, result_file):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)
    annotations = json.load(open(annotation_file))
    
    annotations = {(annotation['id'], annotation['instruction'].lower()): annotation for annotation in annotations}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    for result in results:
        annotation = annotations[(result['question_id'], result['instruction'].lower())]
        pred_list.append({
            "pred_answer": result['output'],
            "gt_answers": annotation['output'],
            "question": annotation['instruction'],
        })
    # print(pred_list)
    print(len(pred_list))
    evaluator = TextMatchEvaluator()
    print('Samples: {}\nAccuracy: '.format(len(pred_list)))
    acc_dict, acc_list = evaluator.eval_pred_list(pred_list)
    print(acc_dict)
    print(acc_list)


    Q7 = "What is the decoration of the vase?"
    Q8 = "What is the overall of the vase?"
    evaluator_bleu = TextCapsBleu4Evaluator()

    evaluator_bleu.set_q(Q7)
    bleu1_Q7 = evaluator_bleu.eval_pred_list(pred_list)
    print(bleu1_Q7)

    evaluator_bleu.set_q(Q8)
    bleu1_Q8 = evaluator_bleu.eval_pred_list(pred_list)
    print(bleu1_Q8)

    print("Accuracy from Q1 to Q6: ", acc_list[0:6])
    print("Bleu of Q7: ", bleu1_Q7)
    print("Bleu_of Q8: ", bleu1_Q8)


if __name__ == "__main__":
    annotation_file = "../data/VaseVLDataset_sub/VaseVL_gt_answers.json"
    # result_file = "../data/VaseVLDataset_sub/VaseVL_inference_answers.jsonl"
    result_file = "../data/VaseVLDataset_sub/VaseVL_Qwen2.5-VL-3B-Instruct_inference_answers.jsonl"

    
    eval_single(annotation_file, result_file)

    print("FINISHED.........")