"""
Run EQA in Habitat-Sim with VLM exploration.

"""

import os
import json
import torch
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
import numpy as np

np.set_printoptions(precision=3)
import csv
import pickle
import logging
import math
import quaternion
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
from src.habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
)
from src.geom import get_cam_intr, get_scene_bnds
from src.tsdf import TSDFPlanner
from src.utils import interpolate_position_and_rotation, get_vlm_loss, get_vlm_response

from src.vlm import VLM
import clip
from src.knowledgebase import DynamicKnowledgeBase
from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt

def display_sample(rgb, depth, save_path="sample.png"):
    # 创建一个包含3列的子图
    fig, axes = plt.subplots(2, 1, figsize=(5, 8))

    # 显示RGB图像
    axes[0].imshow(rgb)
    axes[0].set_title("RGB Image")
    axes[0].axis('off')  # 关闭坐标轴

    # 显示深度图像
    axes[1].imshow(depth, cmap='jet')  # 使用 'jet' 配色方案
    axes[1].set_title("Depth Image")
    axes[1].axis('off')  # 关闭坐标轴

    # 调整子图布局
    plt.tight_layout()

    # 保存图像为PNG文件
    plt.savefig(save_path, format="png")

    plt.close()

def save_rgbd(rgb, depth, save_path="rgbd.png"):
    depth_image = (depth.astype(np.float32) / depth.max()) * 255
    depth_image = np.clip(depth_image, 0, 255).astype(np.uint8)
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGRA)

    rgbd = np.concatenate((rgb, depth_image), axis=0)
    plt.imsave(save_path, rgbd)

def draw_letters(rgb_im, prompt_points_pix, letters, circle_radius, fnt, save_path):
    rgb_im_draw = rgb_im.copy()
    draw = ImageDraw.Draw(rgb_im_draw)
    for prompt_point_ind, point_pix in enumerate(prompt_points_pix):
        draw.ellipse(
            (
                point_pix[0] - circle_radius,
                point_pix[1] - circle_radius,
                point_pix[0] + circle_radius,
                point_pix[1] + circle_radius,
            ),
            fill=(200, 200, 200, 255),
            outline=(0, 0, 0, 255),
            width=3,
        )
        draw.text(
            tuple(point_pix.astype(int).tolist()),
            letters[prompt_point_ind],
            font=fnt,
            fill=(0, 0, 0, 255),
            anchor="mm",
            font_size=12,
        )
    rgb_im_draw.save(save_path)
    return rgb_im_draw

def main(cfg, gpu_id, gpu_index, gpu_count):
    camera_tilt = cfg.camera_tilt_deg * np.pi / 180
    img_height = cfg.img_height
    img_width = cfg.img_width

    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    logging.info(cfg)

    prompt = cfg.prompt
    prompt_caption = prompt.caption
    prompt_rel = prompt.relevent
    prompt_question = prompt.question
    prompt_lsv = prompt.local_sem
    prompt_gsv = prompt.global_sem

    # Load dataset
    with open(cfg.question_data_path) as f:
        questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]
    with open(cfg.init_pose_data_path) as f:
        init_pose_data = {}
        for row in csv.DictReader(f, skipinitialspace=True):
            init_pose_data[row["scene_floor"]] = {
                "init_pts": [
                    float(row["init_x"]),
                    float(row["init_y"]),
                    float(row["init_z"]),
                ],
                "init_angle": float(row["init_angle"]),
            }

    device = f"cuda:{gpu_id}"
    vlm = VLM(cfg.vlm, device=device)

    # init memory module
    rag_cfg = cfg.rag
    if rag_cfg.use_rag:
        knowledge_base = DynamicKnowledgeBase(rag_cfg, device=device)

    letters = ["A", "B", "C", "D"]  # always four
    fnt = ImageFont.truetype("data/Open_Sans/static/OpenSans-Regular.ttf", 30,)

    # Run all questions
    cnt_data = 0
    results_all = []
    part_data = len(questions_data) / gpu_count
    start_idx = int(part_data * gpu_index)
    end_idx = int(part_data * (gpu_index + 1))

    logging.info(f"Loaded {start_idx} - {end_idx} questions.")

    for question_ind in tqdm(range(start_idx, end_idx)):
        if rag_cfg.use_rag:
            knowledge_base.clear()
        kb = []
        # Extract question
        question_data = questions_data[question_ind]
        scene = question_data["scene"]
        floor = question_data["floor"]
        scene_floor = scene + "_" + floor
        question = question_data["question"]
        # choices = [c.split("'")[1] for c in question_data["choices"].split("',")]
        choices = [c.strip("'\"") for c in question_data["choices"].strip("[]").split(", ")]
        answer = question_data["answer"]

        init_pts = init_pose_data[scene_floor]["init_pts"]
        init_angle = init_pose_data[scene_floor]["init_angle"]
        
        logging.info(f"\n========\nIndex: {question_ind} Scene: {scene} Floor: {floor}")

        # Re-format the question to follow LLaMA style
        vlm_question = question
        vlm_pred_candidates = ["A", "B", "C", "D"]
        # open or close vocab
        is_open_vocab = True
        if is_open_vocab:
            answer = choices[vlm_pred_candidates.index(answer)]
        else:
            for token, choice in zip(vlm_pred_candidates, choices):
                vlm_question += "\n" + token + ". " + choice
        logging.info(f"Question:\n{vlm_question}\nAnswer: {answer}")

        # Set data dir for this question - set initial data to be saved
        episode_data_dir = os.path.join(cfg.output_dir, str(question_ind))
        os.makedirs(episode_data_dir, exist_ok=True)

        # Set up scene in Habitat
        try:
            simulator.close()
        except:
            pass
        scene_data_path = ""
        for scene_path in cfg.scene_data_path:
            if os.path.exists(os.path.join(scene_path, scene)):
                scene_data_path = scene_path
                break
        scene_mesh_dir = os.path.join(
            scene_data_path, scene, scene[6:] + ".basis" + ".glb"
        )
        navmesh_file = os.path.join(
            scene_data_path, scene, scene[6:] + ".basis" + ".navmesh"
        )
        sim_settings = {
            "scene": scene_mesh_dir,
            "default_agent": 0,
            "sensor_height": cfg.camera_height,
            "width": img_width,
            "height": img_height,
            "hfov": cfg.hfov,
        }
        sim_cfg = make_simple_cfg(sim_settings)
        simulator = habitat_sim.Simulator(sim_cfg)
        pathfinder = simulator.pathfinder
        pathfinder.seed(cfg.seed)
        pathfinder.load_nav_mesh(navmesh_file)
        if not pathfinder.is_loaded:
            print("Not loaded .navmesh file yet. Please check file path {}.".format(navmesh_file))

        agent = simulator.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        pts = np.array(init_pts)
        angle = init_angle

        # Floor - use pts height as floor height
        rotation = quat_to_coeffs(
            quat_from_angle_axis(angle, np.array([0, 1, 0]))
        ).tolist()
        pts_normal = pos_habitat_to_normal(pts)
        floor_height = pts_normal[-1]
        tsdf_bnds, scene_size = get_scene_bnds(pathfinder, floor_height)
        num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
        logging.info(
            f"Scene size: {scene_size} Floor height: {floor_height} Steps: {num_step}"
        )

        result = {
            "meta": {
                "question_ind": question_ind,
                "question": vlm_question,
                "answer": answer,
                "scene": scene,
                "floor": floor,
                "max_steps": num_step,
            },
            "step": [],
            "summary": {},
        }

        # Initialize TSDF
        tsdf_planner = TSDFPlanner(
            vol_bnds=tsdf_bnds,
            voxel_size=cfg.tsdf_grid_size,
            floor_height_offset=0,
            pts_init=pos_habitat_to_normal(pts),
            init_clearance=cfg.init_clearance * 2,
        )

        # Run steps
        pts_pixs = np.empty((0, 2))  # for plotting path on the image
        smx_vlm_pred = None
        for cnt_step in range(num_step):
            logging.info(f"\n== step: {cnt_step}")

            # Save step info and set current pose
            step_name = f"step_{cnt_step}"
            logging.info(f"Current pts: {pts}")

            agent_state.position = pts
            agent_state.rotation = rotation
            agent.set_state(agent_state)

            pts_normal = pos_habitat_to_normal(pts)
            
            result["step"].append({"step": cnt_step, "pts": pts.tolist(), "angle": angle})

            # Update camera info
            sensor = agent.get_state().sensor_states["depth_sensor"]
            quaternion_0 = sensor.rotation
            translation_0 = sensor.position

            cam_pose = np.eye(4)
            cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
            cam_pose[:3, 3] = translation_0
            cam_pose_normal = pose_habitat_to_normal(cam_pose)
            cam_pose_tsdf = pose_normal_to_tsdf(cam_pose_normal)

            # Get observation at current pose - skip black image, meaning robot is outside the floor
            obs = simulator.get_sensor_observations()
            rgb = obs["color_sensor"]
            depth = obs["depth_sensor"]

            rgb_im = Image.fromarray(rgb, mode="RGBA").convert("RGB")
            if rag_cfg.use_rag:
                caption = vlm.get_response(rgb_im, prompt_caption, [], device=device)

            if cfg.save_obs:
                save_rgbd(rgb, depth, os.path.join(episode_data_dir, f"{cnt_step}_rgbd.png"))
                if rag_cfg.use_rag:
                    rgb_path = os.path.join(episode_data_dir, "{}.png".format(cnt_step))
                    plt.imsave(rgb_path, rgb)
                    # 当前帧加入知识库
                    # knowledge_base.add_text_data(f"{step_name}: position is {pts}, {caption}", device=device)
                    # knowledge_base.add_image_data(rgb_path, device=device)
                    knowledge_base.add_to_knowledge_base(f"{step_name}: position is {pts}, {caption}", rgb_im, device=device)

            num_black_pixels = np.sum(
                np.sum(rgb, axis=-1) == 0
            )  # sum over channel first
            if num_black_pixels < cfg.black_pixel_ratio * img_width * img_height:
                # TSDF fusion
                tsdf_planner.integrate(
                    color_im=rgb,
                    depth_im=depth,
                    cam_intr=cam_intr,
                    cam_pose=cam_pose_tsdf,
                    obs_weight=1.0,
                    margin_h=int(cfg.margin_h_ratio * img_height),
                    margin_w=int(cfg.margin_w_ratio * img_width),
                )

                # tsdf_planner.get_mesh(f"results/scenes/scene_{question_ind}.ply")

                # 模型判断是否有信心回答当前问题
                if rag_cfg.use_rag:
                    kb = knowledge_base.search(prompt_rel.format(question), 
                                               rgb_im, 
                                               top_k=rag_cfg.max_retrieval_num if cnt_step > rag_cfg.max_retrieval_num else cnt_step,
                                               device=device)
                smx_vlm_rel = vlm.get_response(rgb_im, prompt_rel.format(question), kb, device=device)[0].strip(".")
                logging.info(f"Rel - Prob: {smx_vlm_rel}")

                logging.info(f"Prompt Pred: {prompt_question.format(vlm_question)}")
                if rag_cfg.use_rag:
                    kb = knowledge_base.search(prompt_question.format(vlm_question), 
                                               rgb_im, 
                                               top_k=rag_cfg.max_retrieval_num if cnt_step > rag_cfg.max_retrieval_num else cnt_step,
                                               device=device)
                smx_vlm_pred = vlm.get_response(rgb_im, prompt_question.format(vlm_question), kb, device=device)[0].strip(".")
                logging.info(f"Pred - Prob: {smx_vlm_pred}")

                # save data
                result["step"][cnt_step]["smx_vlm_rel"] = smx_vlm_rel[0]
                result["step"][cnt_step]["smx_vlm_pred"] = smx_vlm_pred[0]
                result["step"][cnt_step]["is_success"] = smx_vlm_pred[0] == answer

                # 如果有信心回答，则直接获取答案
                if smx_vlm_rel.lower() in ["c", "d", "e", "yes"]:
                    break

                # Get frontier candidates
                prompt_points_pix = []
                if cfg.use_active:
                    prompt_points_pix, fig = (
                        tsdf_planner.find_prompt_points_within_view(
                            pts_normal,
                            img_width,
                            img_height,
                            cam_intr,
                            cam_pose_tsdf,
                            **cfg.visual_prompt,
                        )
                    )
                    fig.tight_layout()
                    plt.savefig(os.path.join(episode_data_dir, "prompt_points.png".format(cnt_step)))
                    plt.close()

                # Visual prompting
                actual_num_prompt_points = len(prompt_points_pix)
                if actual_num_prompt_points >= cfg.visual_prompt.min_num_prompt_points:
                    rgb_im_draw = draw_letters(rgb_im, 
                                               prompt_points_pix, 
                                               letters, 
                                               cfg.visual_prompt.circle_radius, 
                                               fnt, 
                                               os.path.join(episode_data_dir, f"{cnt_step}_draw.png"))

                    # get VLM reasoning for exploring
                    if cfg.use_lsv:
                        if rag_cfg.use_rag:
                            kb = knowledge_base.search(prompt_lsv.format(question), 
                                                       rgb_im, 
                                                       top_k=rag_cfg.max_retrieval_num if cnt_step > rag_cfg.max_retrieval_num else cnt_step,
                                                       device=device)
                        response = vlm.get_response(rgb_im_draw, prompt_lsv.format(question), kb, device=device)[0]
                        lsv = np.zeros(actual_num_prompt_points)
                        for i in range(actual_num_prompt_points):
                            if response == letters[i]:
                                lsv[i] = 1
                        lsv *= actual_num_prompt_points / 3
                    else:
                        lsv = (
                            np.ones(actual_num_prompt_points) / actual_num_prompt_points
                        )

                    # base - use image without label
                    if cfg.use_gsv:
                        if rag_cfg.use_rag:
                            kb = knowledge_base.search(prompt_gsv.format(question), 
                                                       rgb_im, 
                                                       top_k=rag_cfg.max_retrieval_num if cnt_step > rag_cfg.max_retrieval_num else cnt_step,
                                                       device=device)
                        response = vlm.get_response(rgb_im, prompt_gsv.format(question), kb, device=device)[0].strip(".")
                        gsv = np.zeros(2)
                        if response == "Yes":
                            gsv[0] = 1
                        else:
                            gsv[1] = 1
                        gsv = (
                            np.exp(gsv[0] / cfg.gsv_T) / cfg.gsv_F
                        )  # scale before combined with lsv
                    else:
                        gsv = 1
                    sv = lsv * gsv
                    logging.info(f"Exp - LSV: {lsv} GSV: {gsv} SV: {sv}")

                    # Integrate semantics only if there is any prompted point
                    tsdf_planner.integrate_sem(
                        sem_pix=sv,
                        radius=1.0,
                        obs_weight=1.0,
                    )  # voxel locations already saved in tsdf class

            else:
                logging.info("Skipping black image!")

            # Determine next point
            if cnt_step < num_step:
                pts_normal, angle, cur_angle, pts_pix, fig = tsdf_planner.find_next_pose(
                    pts=pts_normal,
                    angle=angle,
                    cam_pose=cam_pose_tsdf,
                    flag_no_val_weight=cnt_step < cfg.min_random_init_steps,
                    **cfg.planner,
                )
                pts_pixs = np.vstack((pts_pixs, pts_pix))
                pts_normal = np.append(pts_normal, floor_height)
                pts = pos_normal_to_habitat(pts_normal)

                # Add path to ax5, with colormap to indicate order
                ax5 = fig.axes[4]
                ax5.plot(pts_pixs[:, 1], pts_pixs[:, 0], linewidth=5, color="black")
                ax5.scatter(pts_pixs[0, 1], pts_pixs[0, 0], c="white", s=50)
                fig.tight_layout()

                plt.savefig(
                    os.path.join(episode_data_dir, "map.png".format(cnt_step + 1))
                )
                plt.close()
                
            rotation = quat_to_coeffs(
                quat_from_angle_axis(angle, np.array([0, 1, 0]))
            ).tolist()

        # Final prediction
        if cnt_step == num_step - 1:
            logging.info("Max step reached!")
            if rag_cfg.use_rag:
                kb = knowledge_base.search(prompt_question.format(vlm_question), 
                                           rgb_im, 
                                           top_k=rag_cfg.max_retrieval_num if cnt_step > rag_cfg.max_retrieval_num else cnt_step,
                                           device=device)
            smx_vlm_pred = vlm.get_response(rgb_im, prompt_question.format(vlm_question), kb, device=device)[0].strip(".")
            logging.info(f"Pred - Prob: {smx_vlm_pred}")

        if smx_vlm_pred is not None:
            is_success = smx_vlm_pred == answer
        else:
            is_success = False

        # Episode summary
        logging.info(f"\n== Episode Summary")
        logging.info(f"Scene: {scene}, Floor: {floor}")
        logging.info(f"Question:\n{vlm_question}\nAnswer: {answer}")
        logging.info(f"Success (max): {is_success}")
        result["summary"]["smx_vlm_pred"] = smx_vlm_pred
        result["summary"]["smx_vlm_pred"] = smx_vlm_pred
        result["summary"]["is_success"] = is_success
        # Save data
        results_all.append(result)
        cnt_data += 1
        if cnt_data % cfg.save_freq == 0:
            with open(os.path.join(cfg.output_dir, f"results_{gpu_id}_{cnt_data}.json"), "w") as f:
                json.dump(results_all, f, indent=4)

    # Save all data again
    with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
        json.dump(results_all, f, indent=4)
    logging.info(f"\n== All Summary")
    logging.info(f"Number of data collected: {cnt_data}")


def run_on_gpu(gpu_id, gpu_index, gpu_count, cfg_file):
    from omegaconf import OmegaConf
    """在指定 GPU 上运行 main(cfg)，并传递 GPU 信息"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # 设置可见的 GPU
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
    main(cfg, gpu_id, gpu_index, gpu_count)


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
