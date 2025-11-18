import json

import os
import jsonlines
import numpy as np

# {
#     "meta": {
#         "question": "What is the relative volume of the SMX in the SMX?",
#         "scene": "FloorPlan1",
#         "floor": "1",
#         "max_steps": 4
#     },
#     "question_ind": 0,
#     "step_0": {
#         "step": 0,
#         "pts": [
#             1.551065,
#             0.13908827,
#             -3.0018978
#         ],
#         "angle": 0.0953831335597118,
#         "smx_vlm_rel": "No",
#         "smx_vlm_pred": "C",
#         "is_succeess": false
#     },
# }

def load_jsons(files_path):
    """
    读取 JSON 文件。

    :param file_path: 文件路径
    :return: JSON 数据
    """
    all_data = []
    for file_path in files_path:
        print(file_path)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            # raise FileNotFoundError(f"File not found: {file_path}")
        else:
            file_ext = os.path.splitext(file_path)[-1]
            if file_ext == ".json":
                with open(file_path, 'r', encoding='utf-8') as file:
                    all_data.extend(json.load(file))
            elif file_ext == ".jsonl":
                with jsonlines.open(file_path, mode="r") as reader:
                    for item in reader:
                        all_data.append(item)
    
    return all_data

def evaluate(files_path):
    samples = load_jsons(files_path)
    results = []
    choices = ['a', 'b', 'c', 'd']

    path_length = 0
    time_comsume = 0

    success_count = 0
    early_success_count = 0
    final_success_count = 0
    sample_count = len(samples)
    for data in samples:

        path_length += data["summary"]["path_length"]
        time_comsume += data["summary"]["all_time_comsume"]
        success = data["summary"]["is_success"]
        if success:
            success_count += 1

        result = {}

        meta_data = data['meta']
        step_data = data['step']
        max_step = meta_data['max_steps']
        answer = meta_data['answer']

        memory_time = 0
        planner_time = 0
        stop_time = 0
        answering_time = 0

        early_succ = False
        for step in step_data:

            memory_time += step.get('memory_time', 0)
            planner_time += step.get('planner_time', 0)
            stop_time += step.get('stop_time', 0)
            answering_time += step.get('answering_time', 0)

            relevent = step.get("smx_vlm_rel", "")
            if relevent.lower() in ["a","b","c","d","e","yes"]:
                if not early_succ:
                    early_succ = step.get("is_success", False)
                    if early_succ:
                        early_success_count += 1
            else:
                if data["summary"]["is_success"]:
                    final_success_count += 1
                    break

        results.append(result)

    path_length /= len(samples)
    time_comsume /= len(samples)

    memory_time /= len(samples)
    planner_time /= len(samples)
    stop_time /= len(samples)
    answering_time /= len(samples)

    print(f"Average path length: {path_length:.2f}")
    print(f"Average time comsume: {time_comsume:.2f}")
    print(f"Average memory time: {memory_time:.2f}")
    print(f"Average planner time: {planner_time:.2f}")
    print(f"Average stop time: {stop_time:.2f}")
    print(f"Average answering time: {answering_time:.2f}")

    # if success_count == 0:
    #     print("No successful results found.")
    #     return
    
    print(f"总共有{sample_count}个结果，其中成功的有{success_count}个。")
    print(f"成功率为{success_count/sample_count:.2%}。")
    print(f"成功率为{(early_success_count+final_success_count)/sample_count:.2%}。")

if __name__ == '__main__':
    files_path = [
        "results/Qwen3VL/MT-HM3D-new/MT-HM3D-new_gpu0/results.json",
        "results/Qwen3VL/MT-HM3D-new/MT-HM3D-new_gpu1/results.json",
        "results/Qwen3VL/MT-HM3D-new/MT-HM3D-new_gpu2/results.json",
        "results/Qwen3VL/MT-HM3D-new/MT-HM3D-new_gpu3/results.json",
        "results/Qwen3VL/MT-HM3D-new/MT-HM3D-new_gpu4/results.json",
        "results/Qwen3VL/MT-HM3D-new/MT-HM3D-new_gpu5/results.json",
        "results/Qwen3VL/MT-HM3D-new/MT-HM3D-new_gpu6/results.json",
        "results/Qwen3VL/MT-HM3D-new/MT-HM3D-new_gpu7/results.json",
    ]
    evaluate(files_path)
