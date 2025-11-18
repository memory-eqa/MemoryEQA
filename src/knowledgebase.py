import torchvision
import faiss
import numpy as np
import torch.nn.functional as F
import json
import math
import logging
from PIL import Image
import torch

class DynamicKnowledgeBase:
    def __init__(self, cfg, device="cuda"):
        self.text_embedder, self.image_model, self.preprocess = None, None, None
        text_model_name = cfg.text.split("/")[-1]
        visual_model_name = cfg.visual.split("/")[-1]
        if text_model_name == visual_model_name and visual_model_name in ["clip-vit-large-patch14"]:
            from transformers import CLIPProcessor, CLIPModel
            self.text_embedder = self.image_model = CLIPModel.from_pretrained(cfg.text).to(device)
            self.preprocess = CLIPProcessor.from_pretrained(cfg.text)
        elif text_model_name == visual_model_name and visual_model_name in ["blip2-opt-2.7b"]:
            from transformers import Blip2Processor, Blip2Model
            self.text_embedder = self.image_model = Blip2Model.from_pretrained(cfg.text, torch_dtype=torch.float16).to(device)
            self.preprocess = Blip2Processor.from_pretrained(cfg.text)
        else:
            from sentence_transformers import SentenceTransformer
            import clip
            self.text_embedder = SentenceTransformer(cfg.text, device=device)
            self.image_model, self.preprocess = clip.load(cfg.visual, device=device)

        self.index = faiss.IndexFlatL2(cfg.dim)
        self.data = []  # 存储文本和图像的信息

    def update_memory(self, text, image, topk=5, lambda_sim=0.5, device="cuda"):
        memories, indice = self.search(text, image, topk, device)
        if len(memories) == 0:
            return 
        
        # 得到当前观察的特征用于与历史观察计算相似度
        image_inputs = self.preprocess(images=image, return_tensors="pt", padding=True, truncation=True).to(self.image_model.device)
        current_obs = self.image_model.get_image_features(**image_inputs).cpu().detach().numpy()

        current_objs = json.loads(text.split("Objects: ")[-1])

        for memory, ind in zip(memories, indice):
            history_obs = memory["image_vector"]
            descript = memory["text"]

            # 计算当前观察和历史观察的相似度
            current_obs = F.normalize(torch.tensor(current_obs), p=2, dim=-1).numpy()
            history_obs = F.normalize(torch.tensor(history_obs), p=2, dim=-1).numpy()
            obs_similarity = np.dot(current_obs, history_obs.T)[0][0]

            objs = json.loads(descript.split("Objects: ")[-1])
            cap_similarity = []
            for obj in objs:
                cls = obj["cls"]
                cap = obj["caption"]
                pos = obj["pos"]
                for cur_obj in current_objs:
                    cur_cls = cur_obj["cls"]
                    cur_cap = cur_obj["caption"]
                    cur_pos = cur_obj["pos"]
                    # 如果类别相同，则计算距离和语义相似度
                    if cls == cur_cls:
                        # 计算位置的欧氏距离
                        pos_distance = math.exp(-np.linalg.norm(np.array(pos) - np.array(cur_pos)))

                        # 计算描述的相似度
                        cap_inputs = self.preprocess(text=[cap, cur_cap], return_tensors="pt", padding=True, truncation=True).to(self.text_embedder.device)
                        cap_vector = self.text_embedder.get_text_features(**cap_inputs).cpu().detach().numpy()
                        cap_similarity.append(np.dot(cap_vector[0], cap_vector[1]) * pos_distance)

            if len(cap_similarity) > 0:
                cap_similarity = sum(cap_similarity) / len(cap_similarity)
                similarity = (1 - lambda_sim) * cap_similarity + lambda_sim * obs_similarity
            else:
                similarity = 0
                
            if similarity > 0.9:
                logging.info(f"Updating knowledge base...")
                self.index.remove_ids(np.array([ind]))
        # self.add_to_knowledge_base(text, image, device)

    def add_to_knowledge_base(self, text=None, image=None, device='cuda'):
        # 更新记忆
        self.update_memory(text, image, 5, 0.5, device)

        image_vector = text_vector = None
        if text:
            text_inputs = self.preprocess(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
            text_vector = self.text_embedder.get_text_features(**text_inputs).cpu().detach().numpy()
        if image:
            image_inputs = self.preprocess(images=image, return_tensors="pt", padding=True, truncation=True).to(device)
            image_vector = self.image_model.get_image_features(**image_inputs).cpu().detach().numpy()

        if image is not None and text is not None:
            combined_vector = np.concatenate([image_vector, text_vector], axis=1)
        elif image is not None:
            combined_vector = image_vector
        else:
            combined_vector = text_vector
        
        # 添加到FAISS索引
        self.index.add(combined_vector)
        # 存储原始数据
        self.data.append({
            "image": image,  # 原始图像
            "text": text,    # 原始文本
            "image_vector": image_vector,  # 图像向量
            "text_vector": text_vector,    # 文本向量
            "combined_vector": combined_vector  # 拼接后的向量
        })

    def search(self, query_text, image, top_k=5, device='cuda'):
        # 编码文本查询
        text_inputs = self.preprocess(text=query_text, return_tensors="pt", padding=True, truncation=True).to(self.text_embedder.device)
        text_vector = self.text_embedder.get_text_features(**text_inputs).cpu().detach().numpy()

        image_inputs = self.preprocess(images=image, return_tensors="pt", padding=True).to(self.image_model.device)
        image_vector = self.image_model.get_image_features(**image_inputs).cpu().detach().numpy()

        combined_vector = np.concatenate([image_vector, text_vector], axis=1)

        # 在FAISS索引中检索最相似的向量
        distances, indices = self.index.search(combined_vector, top_k)

        # 返回检索到的图像文本对
        retrieved_pairs = [self.data[i] for i in indices[0] if i != -1]
        return retrieved_pairs, indices[0]

    def clear(self):
        # 清空FAISS索引和数据
        self.index.reset()
        self.data = []