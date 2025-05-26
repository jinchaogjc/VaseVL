import json

def convert_conversations_to_qa(input_data):
    qa_list = []
    global_id = 0  # 全局ID计数器
    
    for item in input_data:
        vase_id = item["id"]
        image_path = item["images"]
        conversations = item["conversations"]
        
        # 遍历对话对（每两个元素为一组）
        for i in range(0, len(conversations), 2):
            if i+1 >= len(conversations):
                break  # 跳过不完整的对话对
            
            question = conversations[i]["value"]
            answer = conversations[i+1]["value"]
            
            qa_entry = {
                "id": global_id,
                "images": image_path,
                "instruction": question,
                "output": answer,
                "type": "qa"
            }
            
            qa_list.append(qa_entry)
            global_id += 1  # ID递增
    
    return qa_list


if __name__ == "__main__":
    DATASET = "VaseVLDataset"
    # input_path = os.path.join("../data/", DATASET, DATASET + "_val_annotation_data_test_single_vasevl.json")
    # output_path = os.path.join("../data/", DATASET, DATASET + "_question.json")

    input_path = "data/VaseVLDataset/data_test_single_llava_vasevl_v8.json"
    output_path = "data/VaseVLDataset/vasevl_single_gt_answers.json"
    print(output_path)


    # 加载原始数据
    with open(input_path, "r") as f:
        original_data = json.load(f)

    # 执行转换
    converted_data = convert_conversations_to_qa(original_data)

    # 保存结果
    with open(output_path, "w") as f:
        json.dump(converted_data, f, indent=2)

    print(f"成功转换并保存了{len(converted_data)}条QA数据")
