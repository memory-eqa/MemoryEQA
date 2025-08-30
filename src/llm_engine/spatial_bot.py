import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import numpy as np

class VLM:
    def __init__(self, model_name='/mynvme0/models/SpatialBot-3B/', device='cuda:3'):
        # disable some warnings
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        warnings.filterwarnings('ignore')
        self.device = device
        self.model_name = model_name
        
        # create model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # float32 for cpu
            device_map=self.device,
            trust_remote_code=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True)
        
    def get_response(self, prompt, image1, image2, kb=None, offset_bos=0):
        text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:"
        text_chunks = [self.tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]
        input_ids = torch.tensor(text_chunks[0] + [-201] + [-202] + text_chunks[1][offset_bos:], dtype=torch.long).unsqueeze(0).to(self.device)
        
        channels = len(image2.getbands())
        if channels == 1:
            img = np.array(image2)
            height, width = img.shape
            three_channel_array = np.zeros((height, width, 3), dtype=np.uint8)
            three_channel_array[:, :, 0] = (img // 1024) * 4
            three_channel_array[:, :, 1] = (img // 32) * 8
            three_channel_array[:, :, 2] = (img % 32) * 8
            image2 = Image.fromarray(three_channel_array, 'RGB')
        image_tensor = self.model.process_images([image1,image2], self.model.config).to(dtype=self.model.dtype, device=self.device)

        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=100,
            use_cache=True,
            repetition_penalty=1.0 # increase this to avoid chattering
        )[0]

        response = self.tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
        return  response

if __name__=='__main__':
    vlm = VLM()
    image1 = Image.open('test/111.jpg')
    image2 = np.load("test/depth.npy")  # Load depth map as numpy array
    image2 = Image.fromarray(image2)
    prompt = 'What is the depth value of point <0,0>? Answer directly from depth map.'
    
    response = vlm.get_response(prompt, image1, image2)
    print(response)