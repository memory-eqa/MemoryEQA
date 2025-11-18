import os
import json
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
            raise FileNotFoundError(f"File not found: {file_path}")
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                all_data.extend(json.load(file))
    return all_data

def evaluate(files_path):
    samples = load_jsons(files_path)
    results = []
    choices = ['a', 'b', 'c', 'd']

    path_length = 0
    time_comsume = 0
    input_token_usage = 0
    output_token_usage = 0

    success_count = 0

    for data in samples:

        path_length += data["summary"]["path_length"]
        time_comsume += data["summary"]["all_time_comsume"]

        # input_token_usage += data["summary"]["input_token_usage"]
        # output_token_usage += data["summary"]["output_token_usage"]

        result = {}

        meta_data = data['meta']
        step_data = data['step']
        max_step = meta_data['max_steps']
        answer = meta_data['answer']
        response = data["summary"]["smx_vlm_pred"][0]
        if response == answer:
            success_count += 1

        memory_time = 0
        planner_time = 0
        stop_time = 0
        answering_time = 0

        for step in step_data:
            memory_time += step.get('memory_time', 0)
            planner_time += step.get('planner_time', 0)
            stop_time += step.get('stop_time', 0)
            answering_time += step.get('answering_time', 0)
            
            step_num = step['step'] + 1
            res = step['smx_vlm_pred']
            if isinstance(res, str):
                res = step['smx_vlm_pred'].split('. ')[0]
                if step['smx_vlm_pred'].split('. ')[0].lower() == "y":
                    res = "yes"
                elif step['smx_vlm_pred'].split('. ')[0].lower() == "n":
                    res = "no"
            elif isinstance(res, list):
                index = np.array(np.argmax(res))
                res = choices[index]
            is_success = answer.lower() == res
            if is_success and "norm_early_success_step" not in result.keys():
                result["norm_early_success_step"] = step_num / max_step
            result["norm_success_step"] = step_num / max_step
            result["is_success"] = is_success
        results.append(result)

    path_length /= len(samples)
    time_comsume /= len(samples)
    # input_token_usage /= len(samples)
    # output_token_usage /= len(samples)

    memory_time /= len(samples)
    planner_time /= len(samples)
    stop_time /= len(samples)
    answering_time /= len(samples)

    print(f"Average path length: {path_length:.2f}")
    # print(f"Average input token usage: {input_token_usage:.2f}")
    # print(f"Average output token usage: {output_token_usage:.2f}")
    print(f"Average time comsume: {time_comsume:.2f}")
    print(f"Average memory time: {memory_time:.2f}")
    print(f"Average planner time: {planner_time:.2f}")
    print(f"Average stop time: {stop_time:.2f}")
    print(f"Average answering time: {answering_time:.2f}")

    results_num = len(results)
    norm_steps = 0
    norm_early_steps = 0
    early_count = 0
    # success_count = 0
    for result in results:
        if result.get("is_success"):
            success_count += 1
            norm_steps += result.get("norm_success_step")
        if result.get("norm_early_success_step"):
            norm_early_steps += result.get("norm_early_success_step")
            early_count += 1

    if success_count == 0:
        print("No successful results found.")
        return
    
    print(f"总共有{results_num}个结果，其中成功的有{success_count}个。")
    print(f"成功率为{success_count/results_num:.2%}。")
    print(f"平均归一化成功步数为{norm_steps/success_count:.2}。")

    if early_count > 0:
        print(f"总共有{early_count}个结果提成功。")
        print(f"提早成功率为{early_count/results_num:.2%}")
        print(f"平均归一化提早成功步数为{norm_early_steps/early_count:.2}。")

if __name__ == '__main__':
    files_path = [
        "results/Qwen3VL/MT-HM3D-new.bak/MT-HM3D-new_gpu0/results.json",
        "results/Qwen3VL/MT-HM3D-new.bak/MT-HM3D-new_gpu1/results.json",
        "results/Qwen3VL/MT-HM3D-new.bak/MT-HM3D-new_gpu2/results.json",
        "results/Qwen3VL/MT-HM3D-new.bak/MT-HM3D-new_gpu3/results.json",
        "results/Qwen3VL/MT-HM3D-new.bak/MT-HM3D-new_gpu4/results.json",
        "results/Qwen3VL/MT-HM3D-new.bak/MT-HM3D-new_gpu5/results.json",
        "results/Qwen3VL/MT-HM3D-new.bak/MT-HM3D-new_gpu6/results.json",
        "results/Qwen3VL/MT-HM3D-new.bak/MT-HM3D-new_gpu7/results.json",
    ]
    evaluate(files_path)
