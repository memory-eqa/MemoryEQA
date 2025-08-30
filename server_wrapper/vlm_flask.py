import argparse
import json
import numpy as np
from src.vlm import VLM
from omegaconf import OmegaConf
from flask import Flask, request, jsonify
from PIL import Image

# get config path
parser = argparse.ArgumentParser()
parser.add_argument("-cfg", "--cfg_file", help="cfg file path", default="", type=str)
args = parser.parse_args()
cfg = OmegaConf.load(args.cfg_file)
OmegaConf.resolve(cfg)

vlm = VLM(cfg, cfg.device)


# 创建Flask应用
app = Flask(__name__)

@app.route('/get_loss', methods=['POST'])
def get_loss():
    file = request.files['image']
    img = Image.open(file).convert("RGB")
    
    text = request.form['text']
    str_list = json.loads(request.form['str_list'])
    
    # try:
    result = vlm.get_loss(img, text, str_list)
    return jsonify({"result": result.tolist()}), 200

@app.route('/get_response', methods=['POST'])
def get_response():
    if 'image' in request.files.keys():
        file = request.files['image']
        img = Image.open(file).convert("RGB")
    else:
        img = None
    text = request.form['text']

    result = vlm.get_response(img, text)
    return jsonify({"result": result}), 200

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001, threaded=True)
