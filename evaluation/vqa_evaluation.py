import os
import json
from accuracy_valuator import TextCapsBleu4Evaluator, DateAccuracyEvaluator, STVQAANLSEvaluator
import argparse
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import remove_tags


def eval_single(annotation_file, infer_file, result_file):

    Q1 = "<image>\n What is the fabric of the vase?"
    Q2 = "<image>\n What is the technique of the vase?"
    Q3 = "<image>\n What is the shape name of the vase?"
    Q4 = "<image>\n What is the provenance of the vase?"
    Q5 = "<image>\n What is the date of the vase?"
    Q6 = "<image>\n What is the attribution of the vase?"
    Q7 = "<image>\n What is the decoration of the vase?"
    # Q8 = "<image>\n What is the overall of the vase?"


    experiment_name = os.path.splitext(os.path.basename(infer_file))[0]
    print(experiment_name)
    annotations = json.load(open(annotation_file))
    
    annotations = {(annotation['id'], annotation['instruction'].lower()): annotation for annotation in annotations}
    results = [json.loads(line) for line in open(infer_file)]

    pred_list = []
    for result in results:
        # try:
        #     if "<image>\n " in result['instruction'].lower():
        #         annotation = annotations[(result['question_id'], result['instruction'].lower())]
        #     else:
        #         annotation = annotations[(result['question_id'], "<image>\n "+result['instruction'].lower())]
        try:
            annotation = annotations[(result['question_id'], result['instruction'].lower())]
        except Exception:
            import pdb
            pdb.set_trace()
            pass
        pred_list.append({
            "pred_answer": result['output'],
            "gt_answers": annotation['output'],
            "question": annotation['instruction'],
        })
    # print(pred_list)
    # print(len(pred_list))
    # 
    # evaluator = TextMatchEvaluator()
    # print('Samples: {}\nAccuracy: '.format(len(pred_list)))
    acc_list = [0.0 for i in range(0, 5)]
    # print(acc_list)
    # acc_dict, acc_list = evaluator.eval_pred_list(pred_list)

    # import pdb
    # pdb.set_trace()
    evaluator = STVQAANLSEvaluator()
    evaluator.set_q(Q1)
    acc_list[0] = evaluator.eval_pred_list(pred_list)
    # print('Samples: {}, Q1 Accuracy: {:.2f}%\n'.format(len(pred_list), 100. * acc_list[0]))

    evaluator.set_q(Q2)
    acc_list[1] = evaluator.eval_pred_list(pred_list)
    # print('Samples: {}, Q2 Accuracy: {:.2f}%\n'.format(len(pred_list), 100. * acc_list[1]))
    evaluator.set_q(Q3)
    acc_list[2] = evaluator.eval_pred_list(pred_list)
    # print('Samples: {}, Q3 Accuracy: {:.2f}%\n'.format(len(pred_list), 100. * acc_list[2]))
    evaluator.set_q(Q4)
    acc_list[3] = evaluator.eval_pred_list(pred_list)
    # print('Samples: {}, Q4 Accuracy: {:.2f}%\n'.format(len(pred_list), 100. * acc_list[3]))
    evaluator.set_q(Q6)
    acc_list[4] = evaluator.eval_pred_list(pred_list)
    # print('Samples: {}, Q6 Accuracy: {:.2f}%\n'.format(len(pred_list), 100. * acc_list[4]))

    # print(acc_dict)
    # print(acc_list)
   

    evaluator_date = DateAccuracyEvaluator()
    evaluator_date.set_q(Q5)
    acc_Q5 = evaluator_date.eval_pred_list(pred_list)

  
    evaluator_bleu = TextCapsBleu4Evaluator()
    evaluator_bleu.set_q(Q7)
    bleu1_Q7 = evaluator_bleu.eval_pred_list(pred_list)
    print(bleu1_Q7)

    # evaluator_bleu.set_q(Q8)
    # bleu1_Q8 = evaluator_bleu.eval_pred_list(pred_list)
    # print(bleu1_Q8)

    # save results to result_file

    # print("--------Q1 to Q1 results:----------")
    # print(f"Q1:{Q1}, Accuracy: \t\t {acc_list[0]:.2%}")
    # print(f"Q2:{Q2}, Accuracy: \t {acc_list[1]:.2%}")
    # print(f"Q3:{Q3}, Accuracy: \t {acc_list[2]:.2%}")
    # print(f"Q4:{Q4}, Accuracy: \t {acc_list[3]:.2%}")
    # print(f"Q5:{Q5}, Accuracy: \t\t {acc_Q5:.2%}")
    # print(f"Q6:{Q6}, Accuracy: \t {acc_list[4]:.2%}")
    # print(f"Q7:{Q7}, Bleu: \t\t {bleu1_Q7:.2%}")
    # print(f"Q8:{Q8}, Bleu: \t\t {bleu1_Q8:.2%}")

    Q1 = "What is the fabric of the vase?"
    Q2 = "What is the technique of the vase?"
    Q3 = "What is the shape name of the vase?"
    Q4 = "What is the provenance of the vase?"
    Q5 = "What is the date of the vase?"
    Q6 = "What is the attribution of the vase?"
    Q7 = "What is the decoration of the vase?"

    results = [
        ("Q1", Q1, "Accuracy", acc_list[0]),
        ("Q2", Q2, "Accuracy", acc_list[1]),
        ("Q3", Q3, "Accuracy", acc_list[2]),
        ("Q4", Q4, "Accuracy", acc_list[3]),
        ("Q5", Q5, "Accuracy", acc_Q5),
        ("Q6", Q6, "Accuracy", acc_list[4]),
        ("Q7", Q7, "Bleu", bleu1_Q7),
        # ("Q8", Q8, "Bleu", bleu1_Q8)
    ]

    # 定义表格格式
    header = f"{'Question':<50}\t{'Metric':<10}\t{'Value':>8}"
    separator = "-"*100

    # 生成表格内容
    table_content = []
    for qid, question, metric, value in results:
        formatted_line = f"{qid}:{question:<45}\t{metric + ':':<10}\t{value*100:>7.2f}%"
        table_content.append(formatted_line)

    # 控制台输出
    print(infer_file + "\n")
    print("\nVASE ANALYSIS RESULTS:")
    print(header)
    print(separator)
    print("\n".join(table_content))

    # 文件输出
    with open(result_file, "w") as f:
        f.write(infer_file + "\n")
        f.write("VASE ANALYSIS RESULTS\n")
        f.write(header + "\n")
        f.write(separator + "\n")
        f.write("\n".join(table_content))


if __name__ == "__main__":
    # annotation_file = "data/VaseVLDataset/vasevl_single_gt_answers.json"
    # infer_file = "data/VaseVLDataset_sub/VaseVL_Qwen2.5-VL-3B-Instruct_inference_answers.jsonl"
    # result_file = "results/VaseVL_Qwen2.5-VL-3B-Instruct/VaseVL_Qwen2.5-VL-3B-Instruct_evaluation.txt"     


    """Configure argument parser for inference parameters"""
    infer_parser = argparse.ArgumentParser(description="Evaluation Script")
   
    # Required parameters
    infer_parser.add_argument(
        "--annotation-file",
        type=str,
        default="data/VaseVLDataset/vasevl_single_gt_answers.json",
        help="Path to annotation file (default: %(default)s)"
    )
    
    # Data configuration
    infer_parser.add_argument(
        "--infer-file",
        type=str,
        default="data/VaseVLDataset/VaseVL_Qwen2.5-VL-3B-Instruct_inference_answers.jsonl",
        help="Path to inference file (default: %(default)s)"
    )

    infer_parser.add_argument(
        "--result-file",
        type=str,
        default="results/VaseVL_Qwen2.5-VL-3B-Instruct/VaseVL_Qwen2.5-VL-3B-Instruct_evaluation.txt",
        help="Path to result file (default: %(default)s)"
    )
    args = infer_parser.parse_args()
    print(args)

    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)                  

    eval_single(args.annotation_file, args.infer_file, args.result_file)

    print("FINISHED.........")
