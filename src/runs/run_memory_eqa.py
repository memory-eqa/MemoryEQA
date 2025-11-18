"""
Run EQA in Habitat-Sim with VLM exploration.

"""
import os
import numpy as np
import logging
import csv
import json
import jsonlines

from tqdm import tqdm
from src.modeling.memory_eqa import MemoryEQA

np.set_printoptions(precision=3)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"

def run_on_gpu(gpu_id, gpu_index, gpu_count, cfg_file):
    from omegaconf import OmegaConf
    """在指定 GPU 上运行 main(cfg)，并传递 GPU 信息"""
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # 设置可见的 GPU
    cfg = OmegaConf.load(cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, f"{cfg.exp_name}/{cfg.exp_name}_gpu{gpu_id}")
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(cfg.output_dir, f"log_{gpu_id}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # 将 GPU 信息传递给 main 函数
    logging.info(f"***** Running {cfg.exp_name} on GPU {gpu_id}/{gpu_count} *****")

    memory_eqa = MemoryEQA(cfg, gpu_id)

    with open(cfg.question_data_path) as f:
        questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]

    results_all = []
    part_data = len(questions_data) / gpu_count
    start_idx = int(part_data * gpu_index)
    end_idx = int(part_data * (gpu_index + 1))
    for question_ind in tqdm(range(start_idx, end_idx)):
        data = questions_data[question_ind]
        result = memory_eqa.run(data, question_ind)
        results_all.append(result)
        
        # with open(os.path.join(cfg.output_dir, f"results.jsonl"), "w") as f:
        #     json.dump(results_all, f, indent=4)
        with jsonlines.open(os.path.join(cfg.output_dir, f"results.jsonl"), mode="a") as writer:
            writer.write(result)

    with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
        json.dump(results_all, f, indent=4)


if __name__ == "__main__":
    import argparse
    import os
    import logging
    from multiprocessing import Process, set_start_method

    # 设置多进程启动方式为 spawn
    set_start_method("spawn", force=True)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--cfg_file", help="cfg file path", default="cfg/vlm_exp_ov.yaml", type=str)
    parser.add_argument("-gpus", "--gpu_ids", help="Comma-separated GPU IDs to use (e.g., '0,1,2')", type=str, default="0")
    args = parser.parse_args()

    # Get list of GPUs
    gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(",")]
    gpu_count = len(gpu_ids)  # 计算 GPU 数量

    # Launch processes for each GPU
    processes = []
    for gpu_id in gpu_ids:
        gpu_index = gpu_ids.index(gpu_id)
        p = Process(target=run_on_gpu, args=(gpu_id, gpu_index, gpu_count, args.cfg_file))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()
