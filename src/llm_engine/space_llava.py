import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
import requests
from io import BytesIO
from typing import AnyStr
import re

spatial_prompt = '''

Use the following 4 steps sequentially to answer the question:

Step 1 **Analyze the question**

Step 2 **Identify up to 10 reference scales in the image, ranging from large to small sizes, and list them in the specified format**
- A reference scale must be typical in size.
- A reference scale can be the dimensions of an object or an object part.
- A reference scale must NOT be floor tiles or floor planks.
- Formulate the reference scales using the format: """The [choose from front-to-back, side-to-side, left-to-right, diameter, height (top to bottom edge), or mounting height (bottom edge to floor)] of [object or object part] is approximately [dimension estimate]."""

Step 3 **Propose a robust step-by-step plan to answer the question by using the reference scales in Step 2**
- A robust step-by-step plan performs the estimation in a coarse-to-fine manner.
    - First, use a reliable and large-sized reference scale as the primary reference for estimation.
    - Then, gradually use a reliable and smaller-sized reference scale for adjustment.
    - Repeat until the estimation is precise enough.
- When performing visual comparison, be aware of perspective distortion.
- Do NOT rely on pixel measurements from the images.

Step 4 **Focus on the image and follow the plan in Step 3 to answer the question**

'''

system_message = (
    #"You are VL-Thinking 🤔, a helpful assistant with excellent reasoning ability. "
    "You should first think about the reasoning process and then provide the answer. "
    "Use <think>...</think> and <answer>...</answer> tags."
    "Answer in the format: \\n\\n\\\\scalar{VALUE} \\\\distance_unit{UNIT}\n"
)

class VLM:
    def __init__(self, model_name='/mynvme0/models/SpatialVLM/spacellava-1.5-7b/', device='cuda:3'):
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, device_map=device, torch_dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = device

    def extract_answer(self, text):
        """
        Extracts the text content within the last pair of <answer>...</answer> tags from a given text.

        Args:
            text: The input text string.

        Returns:
            The extracted text content within the last <answer> tags, or None if no such tags are found.
        """
        matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return last_match.group(1).strip()
        return None

    def get_response(self, prompt: AnyStr, image: Image.Image):
        chat = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": spatial_prompt + prompt}]}
        ]

        text_input = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        # Tokenize and generate
        inputs = self.processor(text=[text_input], images=[image], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
        output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip("\n")
        return output


if __name__ == "__main__":
    # Example usage
    model_id = "/mynvme0/models/SpatialVLM/spacellava-1.5-7b/"
    image_path = "test/samples/111.jpg"  # or local path
    prompt = "What can you infer from this image about the environment?"

    # Load model and processor
    vlm = VLM(model_name=model_id)
    image = Image.open(image_path).convert("RGB")

    response = vlm.get_response(prompt, image)
    print(response)
