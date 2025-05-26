import json
from pathlib import Path
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import remove_tags


def convert_json_to_question_list(input_data):
    """
    Convert conversation-based JSON to structured question list format
    
    Parameters:
    input_data (list): Original JSON data containing conversations
    
    Returns:
    list: Processed question list in target format
    """
    question_list = []
    
    
    # Iterate through each vase entry
    for vase in input_data:
        image_path = vase["images"]
        image_path = Path(f"data/{DATASET}/{image_path}").as_posix()
        
        question_id = vase["id"]
        question = vase["instruction"]
        question_entry = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "question_id": question_id
                    },
                    {
                        "type": "text",
                        "text": remove_tags(question)
                    }
                ]
            }
        ]
        question_list.append(question_entry)
               
                
    return question_list

def save_question_list(output_path, question_list):
    """
    Save processed question list to JSON file
    
    Parameters:
    output_path (str): Path to save output file
    question_list (list): Processed data to save
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(question_list, f, 
                     indent=4,
                     ensure_ascii=False)  # Preserve special characters
        print(f"Successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving file: {str(e)}")


if __name__ == "__main__":
    # Sample input data (use your actual JSON data)
    DATASET = "VaseVLDataset"
    input_path = "data/VaseVLDataset/vasevl_single_gt_answers.json"
    output_path = "data/VaseVLDataset/vasevl_single_questions.json"
    print(output_path)


    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            input_json = json.load(f)  # 提取data数组[6,7](@ref)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading file: {str(e)}")
        exit()

    # Process conversion
    processed_questions = convert_json_to_question_list(input_json)

    # Save to file
    save_question_list(output_path, processed_questions)
