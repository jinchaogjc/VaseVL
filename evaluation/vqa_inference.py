from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
# from evaluate import load
import json
import os
from tqdm import tqdm
from PIL import Image
import sys, os
import platform
import torch
from qwen_vl_utils import process_vision_info
from pathlib import Path
import argparse


if platform.system() == 'Darwin' and torch.backends.mps.is_available():
    current_file_abs = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_abs)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    sys.path.append(parent_dir)

    from device import get_device
    device_map, device = get_device()
else:
    if torch.cuda.is_available():
        device_map, device = "auto", "cuda"
    else:
        device_map, device = "auto", "cpu"


# def load_image(image_file):
#     if image_file.startswith("http") or image_file.startswith("https"):
#         response = requests.get(image_file)
#         image = Image.open(BytesIO(response.content)).convert("RGB")
#     else:
#         image = Image.open(image_file).convert("RGB")
#     return image

def read_list_from_json(file_path: str, list_key: str = None):
    """
    从JSON文件读取列表数据
    
    :param file_path: JSON文件路径
    :param list_key: 可选参数，当列表嵌套在字典中时指定键名
    :return: Python列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 情况1：JSON文件直接存储列表 [3,6](@ref)
            if isinstance(data, list):
                return data
            
            # 情况2：列表嵌套在字典中 [1,4](@ref)
            if list_key and isinstance(data, dict):
                return data.get(list_key, [])
            
            raise ValueError("Unsupported JSON format")
            
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
    except json.JSONDecodeError:
        print("错误：JSON格式无效")
    except Exception as e:
        print(f"未知错误：{str(e)}")


def prepare_message_example():
    messages2 = [
    {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    "question_id": 34602
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    messages3 = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "../example/vase.png",
                    "question_id": 34602
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    # Combine messages for batch processing
    import json

    # Define a Python list
    data_list = [messages2, messages3]
    return data_list


def prepare_message(question_file):
     # 读取数据
    # with open(DATASET_JSONL) as f:
    #     samples = [json.loads(line) for line in f]
    data_list = read_list_from_json(question_file)
    print("Eval ", len(data_list), " samples...")
    return data_list


def batch_process(images, texts, device, model, processor, batch_size=2):
    all_outputs = []
    for i in tqdm(range(0, len(images), batch_size)):
        batch_images = images[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]
        
        inputs = processor(
            text=batch_texts,
            images=batch_images,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # generated_ids = model.generate(**inputs, max_new_tokens=128)
        # outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Batch Inference
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        all_outputs.extend(output_texts)
    return all_outputs


def convert_to_jsonl(questions, answers, output_path):
    """
    Convert question list and predictions to JSONL format
    
    :param questions: List of question items in original format
    :param answers: List of predicted answers
    :param output_path: Path to save JSONL file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, (q_item, answer) in enumerate(zip(questions, answers)):
            # Extract image filename from path
            img_path = q_item[0]['content'][0]['image']
            img_file = Path(img_path).name
            
            # Extract question text
            question_text = q_item[0]['content'][1]['text']
            question_id = q_item[0]['content'][0]['question_id']
            
            # Create output record (ID starts from 0)
            record = {
                "question_id": question_id,
                "image": img_file,
                "instruction": question_text,
                "output": answer,
                "type": "qa"
            }
            
            # Write as JSON line
            f.write(json.dumps(record) + '\n')


if __name__=="__main__":
    # device_map, device = get_device()


    """Configure argument parser for inference parameters"""
    infer_parser = argparse.ArgumentParser(description="Evaluation Script")
   
    # Required parameters
    infer_parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Path to pretrained model (default: %(default)s)"
    )
    
    # Data configuration
    infer_parser.add_argument(
        "--image-dir",
        type=str,
        default="../data/TextVQA/images",
        help="Path to image directory (default: %(default)s)"
    )
    
    # infer_parser.add_argument(
    #     "--dataset-file",
    #     type=str,
    #     default="../data/TextVQA/TextVQA.jsonl",
    #     help="Path to dataset JSONL file (default: %(default)s)"
    # )
    
    infer_parser.add_argument(
        "--question-file",
        type=str,
        default="../data/TextVQA/question.json",
        help="Path to question file (default: %(default)s)"
    )
    
    # Output configuration
    infer_parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save inference results"
    )
    
    # Runtime parameters
    infer_parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Inference batch size (default: %(default)s)"
    )
    
    # infer_parser.add_argument(
    #     "--use-cot",
    #     action="store_true",
    #     help="Enable chain-of-thought reasoning"
    # )

    # Example usage:
    
    args = infer_parser.parse_args()
    print(args)

    if "Qwen2.5" in args.model_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype="auto", device_map=device_map
        )
    elif args.model_path.startswith("Qwen/Qwen2-"):
        # default: Load the model on the available device(s)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype="auto", device_map="auto"
        )
    else:
        print(args.model_path, " model not supported.")
    

    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    # processor = AutoProcessor.from_pretrained(args.model_path)

    # messages = prepare_message_example()
    messages = prepare_message(args.question_file)

    
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
   
    image_inputs, video_inputs = process_vision_info(messages)
    print("Eval batch size: ", args.batch_size)
    pred_answer = batch_process(image_inputs, texts, device, model, processor, batch_size=args.batch_size)
    print(pred_answer)

    # save results to .jsonl
    print("SAVING JSON FILE...")
    
    print(messages, pred_answer)
    convert_to_jsonl(messages, pred_answer, args.output_file)
    print("Json file saved to ", args.output_file, " FINISHED.")
