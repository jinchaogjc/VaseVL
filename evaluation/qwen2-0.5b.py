import requests
from PIL import Image
from qwen_vl_utils import process_vision_info
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import os, platform, sys

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


model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)

processor = AutoProcessor.from_pretrained(model_id,
                                          use_fast=True,
                                          padding_side='left')

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
# conversation = [
#     {

#       "role": "user",
#       "content": [
#           {"type": "text", "text": "What are these?"},
#           {"type": "image"},
#         ],
#     },
# ]

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "example/vase.png",
                "question_id": 34602
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
# conversation = [conversation, conversation]
conversation = [conversation, conversation]
# prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
prompt = [
    processor.apply_chat_template(msg, add_generation_prompt=True)
    for msg in conversation
]

image_inputs, video_inputs = process_vision_info(conversation)
# image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
# raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(images=image_inputs, text=prompt,  padding="max_length",            
            max_length=128, return_tensors='pt').to(device, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
