from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
import time
# default: Load the model on the available device(s)
# model = AutoModelForImageTextToText.from_pretrained(
#     "/mynvme0/models/Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = AutoModelForImageTextToText.from_pretrained(
    "/mynvme0/models/Qwen/Qwen3-VL-8B-Instruct",
    dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="cuda:0",
)

processor = AutoProcessor.from_pretrained("/mynvme0/models/Qwen/Qwen3-VL-8B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output

t = time.time()
generated_ids = model.generate(**inputs, max_new_tokens=128)
print("Generation time:", time.time() - t)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)