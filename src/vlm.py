import time
import logging
import torch
import numpy as np

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from src.request_api import RequestAPI

class VLM:
    def __init__(self, cfg, device="cuda"):
        start_time = time.time()
        # self.model = load(cfg.model_id, hf_token=cfg.hf_token)
        model_id = cfg.model_name_or_path.split("/")[-1]
        self.model = None
        self.processor = None
        logging.info(f"Loading VLM model {model_id}")

        if model_id in ["Qwen2-VL-2B-Instruct", "Qwen2-VL-72B-Instruct", "Qwen2-VL-7B-Instruct", "Qwen2-VL-72B-Instruct-GPTQ-Int4", "Qwen2-VL-72B-Instruct-GPTQ-Int8"]:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                cfg.model_name_or_path,
                torch_dtype="auto",
                attn_implementation="flash_attention_2",
                device_map=device,
            )
            self.processor = AutoProcessor.from_pretrained(cfg.model_name_or_path)

        elif model_id in ["GPT-4o"]:
            self.model = RequestAPI()
            
        else:
            raise ValueError(f"Unknown model_id: {model_id}")

        logging.info(f"Loaded VLM in {time.time() - start_time:.3f}s")

        self.input_token_usage = 0
        self.output_token_usage = 0

    def generate(self, prompt, image, T=0.4, max_tokens=512):
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()
        generated_text = self.model.generate(
            image,
            prompt_text,
            do_sample=True,
            temperature=T,
            max_new_tokens=max_tokens,
            min_length=1,
        )
        return generated_text

    def get_response(self, image=None, prompt=None, kb=None, device="cuda"):
        if isinstance(self.model, RequestAPI):
            return self.get_response_api(image, prompt, kb)
        return self.get_response_local(image, prompt, kb, device)

    def get_response_api(self, image, prompt, kb):
        return self.model.request_with_retry(image, prompt, kb)

    def get_response_local(self, image=None, prompt=None, kb=None, device="cuda"):
        # 创建对话信息
        message = {
                "role": "user",
                "content": [],
        }
        # 添加知识库信息
        context = []
        for item in kb:
            message["content"].append({
                    "type": "image",
                    "image": item['image'],
                })
            message["content"].append({
                    "type": "text",
                    "text": item['text'],
                })

        # 添加图像和提示信息
        if image is not None:
            message["content"].append({
                "type": "image",
                "image": image,
            })
        if prompt is not None:
            message["content"].append({
                "type": "text",
                "text": prompt,
            })
        messages = [message]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        self.input_token_usage += inputs.input_ids.shape[1]
        self.output_token_usage += np.array([gen_ids.shape[0] for gen_ids in generated_ids_trimmed]).sum()

        return output_text

    def get_loss(self, image, prompt, tokens, get_smx=True, T=1):
        "Get unnormalized losses (negative logits) of the tokens"
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()
        losses = self.model.get_loss(
            image,
            prompt_text,
            return_string_probabilities=tokens,
        )[0]
        losses = np.array(losses)
        if get_smx:
            return np.exp(-losses / T) / np.sum(np.exp(-losses / T))
        return losses


