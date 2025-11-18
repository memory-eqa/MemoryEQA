# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import base64
import os
import random
import socket
import time
from typing import Any, Dict

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request


class ServerMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def process_payload(self, payload: dict) -> dict:
        raise NotImplementedError


def host_model(model: Any, name: str, port: int = 5000) -> None:
    """
    Hosts a model as a REST API using Flask.
    """
    app = Flask(__name__)

    @app.route(f"/{name}", methods=["POST"])
    def process_request() -> Dict[str, Any]:
        payload = request.json
        print(f"payload{payload}")
        return jsonify(model.process_payload(payload))

    # app.run(host="localhost", port=port)
    app.run(host="0.0.0.0", port=port)



def bool_arr_to_str(arr: np.ndarray) -> str:
    """Converts a boolean array to a string."""
    packed_str = base64.b64encode(arr.tobytes()).decode()
    return packed_str


def str_to_bool_arr(s: str, shape: tuple) -> np.ndarray:
    """Converts a string to a boolean array."""
    # Convert the string back into bytes using base64 decoding
    bytes_ = base64.b64decode(s)

    # Convert bytes to np.uint8 array
    bytes_array = np.frombuffer(bytes_, dtype=np.uint8)

    # Reshape the data back into a boolean array
    unpacked = bytes_array.reshape(shape)
    return unpacked

def numpy_to_string(array):
    # 将 numpy 数组转换为 bytes
    array_bytes = array.tobytes()
    # 将 bytes 编码为 base64 string
    array_string = base64.b64encode(array_bytes).decode('utf-8')
    # 将数组的形状信息添加到字符串的开头
    shape_str = ','.join(map(str, array.shape))
    dtype_str = str(array.dtype)
    array_string = f"{shape_str}|{dtype_str}|{array_string}"
    return array_string

def string_to_numpy(array_string):
    # 从字符串中提取形状信息和数据类型
    shape_str, dtype_str, array_data = array_string.split('|')
    shape = tuple(map(int, shape_str.split(',')))
    dtype = np.dtype(dtype_str)
    # 解码 base64 string 为 bytes
    array_bytes = base64.b64decode(array_data.encode('utf-8'))
    # 将 bytes 转换回 numpy 数组
    array = np.frombuffer(array_bytes, dtype=dtype).reshape(shape)
    return array


def image_to_str(img_np: np.ndarray, quality: int = 90) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    retval, buffer = cv2.imencode(".jpg", img_np, encode_param)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str


def str_to_image(img_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(img_str)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_np = cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)
    return img_np

import json
import json
import numpy as np

def json_to_pointcloud2(pc2_msg_json: str) -> np.ndarray:
    """
    Convert a JSON string representation of PointCloud2 back to a numpy array.

    :param pc2_msg_json: JSON string representation of the PointCloud2 message.
    :return: A numpy array representation of the PointCloud2 message.
    """
    # Parse the JSON string
    pc2_msg_dict = json.loads(pc2_msg_json)
    
    # Extract necessary information from the JSON dictionary
    height = pc2_msg_dict["height"]
    width = pc2_msg_dict["width"]
    is_dense = pc2_msg_dict["is_dense"]
    point_step = int(pc2_msg_dict["point_step"])  # Ensure this is an integer
    row_step = int(pc2_msg_dict["row_step"])  # Ensure this is an integer
    fields = pc2_msg_dict["fields"]
    data = pc2_msg_dict["data"]

    # Determine dtype based on fields
    dtype = []
    for field in fields:
        name = field["name"]
        offset = field["offset"]
        datatype = field["datatype"]
        count = field["count"]
        if datatype == 7:  # FLOAT32
            dtype.append((name, np.float32))
        elif datatype == 1:  # UINT8
            dtype.append((name, np.uint8))
        # Add more types as needed

    # Convert the data from list to bytes
    data_bytes = bytes(data)  # Ensure data is in bytes

    # Create numpy array from bytes
    num_points = width * height
    num_fields = len(dtype)
    
    # Assuming data consists of float32
    points = np.frombuffer(data_bytes, dtype=np.float32)
    
    # Reshape array based on width, height, and number of fields
    try:
        points_reshaped = points.reshape((height, width, num_fields))
    except ValueError as e:
        raise ValueError(f"Cannot reshape array of size {points.size} into shape ({height}, {width}, {num_fields}): {e}")
    
    return points_reshaped

def send_request(url: str, **kwargs: Any) -> dict:
    response = {}
    for attempt in range(10):
        try:
            response = _send_request(url, **kwargs)
            break
        except Exception as e:
            if attempt == 9:
                print(e)
                exit()
            else:
                print(f"Error: {e}. Retrying in 20-30 seconds...")
                time.sleep(20 + random.random() * 10)

    return response


def _send_request(url: str, **kwargs: Any) -> dict:
    lockfiles_dir = "lockfiles"
    if not os.path.exists(lockfiles_dir):
        os.makedirs(lockfiles_dir)
    filename = url.replace("/", "_").replace(":", "_") + ".lock"
    filename = filename.replace("localhost", socket.gethostname())
    filename = os.path.join(lockfiles_dir, filename)
    try:
        while True:
            # Use a while loop to wait until this filename does not exist
            while os.path.exists(filename):
                # If the file exists, wait 50ms and try again
                time.sleep(0.05)

                try:
                    # If the file was last modified more than 120 seconds ago, delete it
                    if time.time() - os.path.getmtime(filename) > 120:
                        os.remove(filename)
                except FileNotFoundError:
                    pass

            rand_str = str(random.randint(0, 1000000))

            with open(filename, "w") as f:
                f.write(rand_str)
            time.sleep(0.05)
            try:
                with open(filename, "r") as f:
                    if f.read() == rand_str:
                        break
            except FileNotFoundError:
                pass

        # Create a payload dict which is a clone of kwargs but all np.array values are
        # converted to strings
        payload = {}
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                payload[k] = image_to_str(v, quality=kwargs.get("quality", 90))
            else:
                payload[k] = v

        # Set the headers
        headers = {"Content-Type": "application/json"}

        start_time = time.time()
        while True:
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=1)
                if resp.status_code == 200:
                    result = resp.json()
                    break
                else:
                    raise Exception("Request failed")
            except (
                requests.exceptions.Timeout,
                requests.exceptions.RequestException,
            ) as e:
                print(e)
                if time.time() - start_time > 20:
                    raise Exception("Request timed out after 20 seconds")

        try:
            # Delete the lock file
            os.remove(filename)
        except FileNotFoundError:
            pass

    except Exception as e:
        try:
            # Delete the lock file
            os.remove(filename)
        except FileNotFoundError:
            pass
        raise e

    return result
