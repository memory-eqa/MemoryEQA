import torch

from VideoMind.videomind.constants import GROUNDER_PROMPT, PLANNER_PROMPT, VERIFIER_PROMPT
from VideoMind.videomind.dataset.utils import process_vision_info
from VideoMind.videomind.model.builder import build_model
from VideoMind.videomind.utils.io import get_duration
from VideoMind.videomind.utils.parser import parse_span

MODEL_PATH = '/mynvme0/models/VideoMind/VideoMind-7B/'

video_path = 'test/samples/2IXHD25Gvsc_0190789_0192589.mp4'
question = 'How many people appear in the video?'

# initialize role *grounder*
model, processor = build_model(MODEL_PATH)
device = next(model.parameters()).device

# initialize role *planner*
model.load_adapter(f'{MODEL_PATH}/planner', adapter_name='planner')

# initialize role *verifier*
model.load_adapter(f'{MODEL_PATH}/verifier', adapter_name='verifier')

# ==================== Planner ====================

messages = [{
    'role':
    'user',
    'content': [{
        'type': 'video',
        'video': video_path,
        'min_pixels': 36 * 28 * 28,
        'max_pixels': 64 * 28 * 28,
        'max_frames': 100,
        'fps': 1.0
    }, {
        'type': 'text',
        'text': PLANNER_PROMPT.format(question)
    }]
}]

# preprocess inputs
text = processor.apply_chat_template(messages, add_generation_prompt=True)
images, videos = process_vision_info(messages)
data = processor(text=[text], images=images, videos=videos, return_tensors='pt').to(device)

# switch adapter to *planner*
model.base_model.disable_adapter_layers()
model.base_model.enable_adapter_layers()
model.set_adapter('planner')

# run inference
output_ids = model.generate(**data, do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=256)

# decode output ids
output_ids = output_ids[0, data.input_ids.size(1):-1]
response = processor.decode(output_ids, clean_up_tokenization_spaces=False)

print(f'Planner Response: {response}')

# ==================== Grounder ====================

messages = [{
    'role':
    'user',
    'content': [{
        'type': 'video',
        'video': video_path,
        'min_pixels': 36 * 28 * 28,
        'max_pixels': 64 * 28 * 28,
        'max_frames': 150,
        'fps': 1.0
    }, {
        'type': 'text',
        'text': GROUNDER_PROMPT.format(question)
    }]
}]

# preprocess inputs
text = processor.apply_chat_template(messages, add_generation_prompt=True)
images, videos = process_vision_info(messages)
data = processor(text=[text], images=images, videos=videos, return_tensors='pt').to(device)

# switch adapter to *grounder*
model.base_model.disable_adapter_layers()
model.base_model.enable_adapter_layers()
model.set_adapter('grounder')

# run inference
output_ids = model.generate(**data, do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=256)

# decode output ids
output_ids = output_ids[0, data.input_ids.size(1):-1]
response = processor.decode(output_ids, clean_up_tokenization_spaces=False)

print(f'Grounder Response: {response}')

duration = get_duration(video_path)

# 1. extract timestamps and confidences
blob = model.reg[0].cpu().float()
pred, conf = blob[:, :2] * duration, blob[:, -1].tolist()

# 2. clamp timestamps
pred = pred.clamp(min=0, max=duration)

# 3. sort timestamps
inds = (pred[:, 1] - pred[:, 0] < 0).nonzero()[:, 0]
pred[inds] = pred[inds].roll(1)

# 4. convert timestamps to list
pred = pred.tolist()

print(f'Grounder Regressed Timestamps: {pred}')

# ==================== Verifier ====================

# using top-5 predictions
probs = []
for cand in pred[:5]:
    s0, e0 = parse_span(cand, duration, 2)
    offset = (e0 - s0) / 2
    s1, e1 = parse_span([s0 - offset, e0 + offset], duration)

    # percentage of s0, e0 within s1, e1
    s = (s0 - s1) / (e1 - s1)
    e = (e0 - s1) / (e1 - s1)

    messages = [{
        'role':
        'user',
        'content': [{
            'type': 'video',
            'video': video_path,
            'video_start': s1,
            'video_end': e1,
            'min_pixels': 36 * 28 * 28,
            'max_pixels': 64 * 28 * 28,
            'max_frames': 64,
            'fps': 2.0
        }, {
            'type': 'text',
            'text': VERIFIER_PROMPT.format(question)
        }]
    }]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    data = processor(text=[text], images=images, videos=videos, return_tensors='pt')

    # ===== insert segment start/end tokens =====
    video_grid_thw = data['video_grid_thw'][0]
    num_frames, window = int(video_grid_thw[0]), int(video_grid_thw[1] * video_grid_thw[2] / 4)
    assert num_frames * window * 4 == data['pixel_values_videos'].size(0)

    pos_s, pos_e = round(s * num_frames), round(e * num_frames)
    pos_s, pos_e = min(max(0, pos_s), num_frames), min(max(0, pos_e), num_frames)
    assert pos_s <= pos_e, (num_frames, s, e)

    base_idx = torch.nonzero(data['input_ids'][0] == model.config.vision_start_token_id).item()
    pos_s, pos_e = pos_s * window + base_idx + 1, pos_e * window + base_idx + 2

    input_ids = data['input_ids'][0].tolist()
    input_ids.insert(pos_s, model.config.seg_s_token_id)
    input_ids.insert(pos_e, model.config.seg_e_token_id)
    data['input_ids'] = torch.LongTensor([input_ids])
    data['attention_mask'] = torch.ones_like(data['input_ids'])
    # ===========================================

    data = data.to(device)

    # switch adapter to *verifier*
    model.base_model.disable_adapter_layers()
    model.base_model.enable_adapter_layers()
    model.set_adapter('verifier')

    # run inference
    with torch.inference_mode():
        logits = model(**data).logits[0, -1].softmax(dim=-1)

    # NOTE: magic numbers here
    # In Qwen2-VL vocab: 9454 -> Yes, 2753 -> No
    score = (logits[9454] - logits[2753]).sigmoid().item()
    probs.append(score)

# sort predictions by verifier's confidence
ranks = torch.Tensor(probs).argsort(descending=True).tolist()

pred = [pred[idx] for idx in ranks]
conf = [conf[idx] for idx in ranks]

print(f'Verifier Re-ranked Timestamps: {pred}')

# ==================== Answerer ====================

# select the best candidate moment
s, e = parse_span(pred[0], duration, 32)

messages = [{
    'role':
    'user',
    'content': [{
        'type': 'video',
        'video': video_path,
        'video_start': s,
        'video_end': e,
        'min_pixels': 128 * 28 * 28,
        'max_pixels': 256 * 28 * 28,
        'max_frames': 32,
        'fps': 2.0
    }, {
        'type': 'text',
        'text': question
    }]
}]

text = processor.apply_chat_template(messages, add_generation_prompt=True)
images, videos = process_vision_info(messages)
data = processor(text=[text], images=images, videos=videos, return_tensors='pt').to(device)

# remove all adapters as *answerer* is the base model itself
with model.disable_adapter():
    output_ids = model.generate(**data, do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=256)

# decode output ids
output_ids = output_ids[0, data.input_ids.size(1):-1]
response = processor.decode(output_ids, clean_up_tokenization_spaces=False)

print(f'Answerer Response: {response}')
