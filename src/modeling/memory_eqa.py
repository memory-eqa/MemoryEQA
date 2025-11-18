"""
Run EQA in Habitat-Sim with VLM exploration.

"""

import os
import json

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
logging.basicConfig(format='%(message)s',level=logging.INFO)
import math
import quaternion
import cv2
import matplotlib.pyplot as plt
import ast
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
from src.utils import (
    draw_letters,
    save_rgbd,
    display_sample,
    pixel2world
)

from src.vlm import VLM
# from src.llm_engine.spatial_bot import VLM
from src.knowledgebase import DynamicKnowledgeBase
from ultralytics import YOLO

import matplotlib.pyplot as plt
import time

class MemoryEQA():
    def __init__(self, cfg, gpu_id):
        self.cfg = cfg
        self.device = f"cuda:{gpu_id}"
        logging.info(cfg)

        self.camera_tilt = cfg.camera_tilt_deg * np.pi / 180
        self.cam_intr = get_cam_intr(cfg.hfov, cfg.img_height, cfg.img_width)
        self.img_height = cfg.img_height
        self.img_width = cfg.img_width

        self.simulator = None

        # init prompts
        prompt = cfg.prompt
        self.prompt_caption = prompt.caption
        self.prompt_rel = prompt.relevent
        self.prompt_question = prompt.question
        self.prompt_lsv = prompt.local_sem
        self.prompt_gsv = prompt.global_sem

        # load init pose data
        with open(cfg.init_pose_data_path) as f:
            self.init_pose_data = {}
            for row in csv.DictReader(f, skipinitialspace=True):
                self.init_pose_data[row["scene_floor"]] = {
                    "init_pts": [
                        float(row["init_x"]),
                        float(row["init_y"]),
                        float(row["init_z"]),
                    ],
                    "init_angle": float(row["init_angle"]),
                }

        # init VLM model
        self.vlm = VLM(cfg.vlm, device=self.device)
        # self.vlm = VLM(device=self.device)

        # init memory module
        if cfg.rag.use_rag:
            self.knowledge_base = DynamicKnowledgeBase(cfg.rag, device=self.device)
        # init detector 'yolov12{n/s/m/l/x}.pt'
        self.detector = YOLO(cfg.detector)

        # init drawing
        self.letters = ["A", "B", "C", "D"]  # always four
        self.fnt = ImageFont.truetype("data/Open_Sans/static/OpenSans-Regular.ttf", 30,)

        self.confident_threshold = ["b", "c", "d", "e", "yes"]

    def init_sim(self, scene):
        # Set up scene in Habitat
        try:
            self.simulator.close()
        except:
            pass
        scene_data_path = ""
        for scene_path in self.cfg.scene_data_path:
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
            "sensor_height": self.cfg.camera_height,
            "width": self.img_width,
            "height": self.img_height,
            "hfov": self.cfg.hfov,
        }
        sim_cfg = make_simple_cfg(sim_settings)
        self.simulator = habitat_sim.Simulator(sim_cfg)
        pathfinder = self.simulator.pathfinder
        pathfinder.seed(self.cfg.seed)
        pathfinder.load_nav_mesh(navmesh_file)
        if not pathfinder.is_loaded:
            print("Not loaded .navmesh file yet. Please check file path {}.".format(navmesh_file))

        agent = self.simulator.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()

        return agent, agent_state, self.simulator, pathfinder

    def init_planner(self, tsdf_bnds, pts):
        # Initialize TSDF Planner
        tsdf_planner = TSDFPlanner(
            vol_bnds=tsdf_bnds,
            voxel_size=self.cfg.tsdf_grid_size,
            floor_height_offset=0,
            pts_init=pts,
            init_clearance=self.cfg.init_clearance * 2,
        )
        return tsdf_planner

    def prepare_data(self, question_data, question_ind):
        if self.cfg.rag.use_rag:
            self.knowledge_base.clear()

        # Extract question
        scene = question_data["scene"]
        floor = question_data["floor"]
        scene_floor = scene + "_" + "0"
        question = question_data["question"]
        choices = [c.strip("'\"") for c in question_data["choices"].strip("[]").split(", ")]
        answer = question_data["answer"]

        init_pts = self.init_pose_data[scene_floor]["init_pts"]
        init_angle = self.init_pose_data[scene_floor]["init_angle"]
        # init_pts = ast.literal_eval(question_data["init_pos"])
        # init_angle = float(question_data["init_angle"])
        
        logging.info(f"\n========\nIndex: {question_ind} Scene: {scene} Floor: {floor}")

        # Re-format the question to follow LLaMA style
        vlm_question = question
        vlm_pred_candidates = ["A", "B", "C", "D"]

        # open or close vocab
        is_open_vocab = False
        if is_open_vocab:
            answer = choices[vlm_pred_candidates.index(answer)]
        else:
            for token, choice in zip(vlm_pred_candidates, choices):
                vlm_question += "\n" + token + ". " + choice
        logging.info(f"Question:\n{vlm_question}\nAnswer: {answer}")

        # Set data dir for this question - set initial data to be saved
        episode_data_dir = os.path.join(self.cfg.output_dir, str(question_ind))
        os.makedirs(episode_data_dir, exist_ok=True)

        agent, agent_state, self.simulator, pathfinder = self.init_sim(scene)
        
        pts = np.array(init_pts)
        angle = init_angle

        # Floor - use pts height as floor height
        rotation = quat_to_coeffs(
            quat_from_angle_axis(angle, np.array([0, 1, 0]))
        ).tolist()
        pts_normal = pos_habitat_to_normal(pts)
        floor_height = pts_normal[-1]
        tsdf_bnds, scene_size = get_scene_bnds(pathfinder, floor_height)
        num_step = int(math.sqrt(scene_size) * self.cfg.max_step_room_size_ratio)
        logging.info(
            f"Scene size: {scene_size} Floor height: {floor_height} Steps: {num_step}"
        )

        # init planner
        tsdf_planner = self.init_planner(tsdf_bnds, pts_normal)

        # path_length_gt = float(question_data["trajectory_lenth"])
        # related_objs = question_data["related_objs"]
        metadata = {
            # "related_objs": related_objs,
            # "shortest_length": path_length_gt,
            "question_ind": question_ind,
            "org_question": question,
            "question": vlm_question,
            "answer": answer,
            "scene": scene,
            "floor": floor,
            "max_steps": num_step,
            "angle": angle,
            "init_pts": pts.tolist(),
            "init_rotation": rotation,
            "floor_height": floor_height,
            "scene_size": scene_size,
        }

        return metadata, agent, agent_state, tsdf_planner, episode_data_dir

    def get_mesh(self, planner, save_path):
        # f"results/scenes/scene_{question_ind}.ply"
        return planner.get_mesh(save_path)

    def run(self, question_data, question_ind):
        meta, agent, agent_state, tsdf_planner, episode_data_dir = self.prepare_data(question_data, question_ind)

        result = {
            "meta": meta,
            "step": [],
            "summary": {},
        }

        # Extract metadata
        question = meta["org_question"]
        vlm_question = meta["question"]
        answer = meta["answer"]
        scene = meta["scene"]
        floor = meta["floor"]
        num_step = meta["max_steps"]
        angle = meta["angle"]
        pts = np.array(meta["init_pts"])
        rotation = meta["init_rotation"]
        floor_height = meta["floor_height"]
        scene_size = meta["scene_size"]

        # Run steps
        pts_pixs = np.empty((0, 2))  # for plotting path on the image
        smx_vlm_pred = None

        shortest_path = habitat_sim.ShortestPath()
        path_length = 0
        time_comsume = 0

        all_time = time.time()
        for cnt_step in range(num_step):
            logging.info(f"\n== step: {cnt_step}")
            # Save step info and set current pose
            step_name = f"step_{cnt_step}"
            logging.info(f"Current pts: {pts}")
            
            planner_time = 0

            shortest_path.requested_start = agent_state.position
            shortest_path.requested_end = pts
            found = self.simulator.pathfinder.find_path(shortest_path)
            if found:
                path_length += shortest_path.geodesic_distance

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
            obs = self.simulator.get_sensor_observations()
            rgb = obs["color_sensor"]
            depth = obs["depth_sensor"]
            depth_im = Image.fromarray(depth)
            rgb_im = Image.fromarray(rgb, mode="RGBA").convert("RGB")

            room = self.vlm.get_response(rgb_im, "What room are you most likely to be in at the moment? Answer with a phrase", [], device=self.device)[0]
            # room = self.vlm.get_response("What room are you most likely to be in at the moment? Answer with a phrase", rgb_im, depth_im)
            
            t = time.time()
            objects = self.detector(rgb_im)[0]
            objs_info = []
            for box in objects.boxes:
                cls = objects.names[box.cls.item()]
                box = box.xyxy[0].cpu()
                # 裁剪目标区域进行描述
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                obj_im = rgb_im.crop((x1, y1, x2, y2))
                obj_caption = self.vlm.get_response(obj_im, f"Describe this {cls} in a short sentence.", [], device=self.device)[0]
                # obj_caption = self.vlm.get_response(self.prompt_caption, obj_im, depth_im)

                # 中心点转换世界坐标
                x, y = (x1 + x2) / 2, (y1 + y2) / 2
                world_pos = pixel2world(x, y, depth[int(y), int(x)], cam_pose)
                world_pos = pos_normal_to_habitat(world_pos)
                # 保存目标信息
                obj_dict = {"room": room, "cls": cls ,"caption": obj_caption, "pos": world_pos.tolist()}
                objs_info.append(obj_dict)

            if self.cfg.save_obs:
                save_rgbd(rgb, depth, os.path.join(episode_data_dir, f"{cnt_step}_rgbd.png"))

            if self.cfg.rag.use_rag:
                caption = self.vlm.get_response(rgb_im, self.prompt_caption, [], device=self.device)[0]
                rgb_path = os.path.join(episode_data_dir, "{}.png".format(cnt_step))
                plt.imsave(rgb_path, rgb)
                # 构建目标信息
                objs_str = json.dumps(objs_info)
                memory_text = f"{step_name}: agent position is {pts}. {caption} Objects: {objs_str}"
                # self.knowledge_base.update_memory(f"{step_name}: agent position is {pts}. {caption}. Objects: {objs_str}", rgb_im, device=self.device)
                self.knowledge_base.add_to_knowledge_base(memory_text, rgb_im, device=self.device)
                # self.knowledge_base.add_to_knowledge_base(f"{step_name}: agent position is {pts}. Objects: {objs_str}", rgb_im, device=self.device)
                # self.knowledge_base.add_to_knowledge_base(f"{step_name}: agent position is {pts}. {caption}", rgb_im, device=self.device)

            memory_time = time.time() - t
            result["step"][cnt_step]["memory_time"] = float(memory_time)

            num_black_pixels = np.sum(
                np.sum(rgb, axis=-1) == 0
            )  # sum over channel first
            if num_black_pixels < self.cfg.black_pixel_ratio * self.img_width * self.img_height:
                # TSDF fusion
                t = time.time()
                tsdf_planner.integrate(
                    color_im=rgb,
                    depth_im=depth,
                    cam_intr=self.cam_intr,
                    cam_pose=cam_pose_tsdf,
                    obs_weight=1.0,
                    margin_h=int(self.cfg.margin_h_ratio * self.img_height),
                    margin_w=int(self.cfg.margin_w_ratio * self.img_width),
                )
                planner_time += (time.time() - t)
                result["step"][cnt_step]["planner_time"] = float(planner_time)
                
                # 模型判断是否有信心回答当前问题
                t = time.time()
                kb = []
                if self.cfg.rag.use_rag:
                    kb, _ = self.knowledge_base.search(self.prompt_rel.format(question), 
                                               rgb_im, 
                                               top_k=self.cfg.rag.max_retrieval_num if cnt_step > self.cfg.rag.max_retrieval_num else cnt_step+1,
                                               device=self.device)
                smx_vlm_rel = self.vlm.get_response(rgb_im, self.prompt_rel.format(question), kb, device=self.device)[0].strip(".")
                # smx_vlm_rel = self.vlm.get_response(self.prompt_rel.format(question), rgb_im, depth_im).strip(".")
                logging.info(self.prompt_rel.format(question))
                logging.info(f"Rel - Prob: {smx_vlm_rel}")
                stop_time = time.time() - t
                result["step"][cnt_step]["stop_time"] = float(stop_time)

                t = time.time()
                logging.info(f"Prompt Pred: {self.prompt_question.format(vlm_question)}")
                if self.cfg.rag.use_rag:
                    kb, _ = self.knowledge_base.search(self.prompt_question.format(vlm_question), 
                                               rgb_im, 
                                               top_k=self.cfg.rag.max_retrieval_num if cnt_step > self.cfg.rag.max_retrieval_num else cnt_step+1,
                                               device=self.device)
                
                smx_vlm_pred = self.vlm.get_response(rgb_im, self.prompt_question.format(vlm_question), kb, device=self.device)[0].strip(".")
                # smx_vlm_pred = self.vlm.get_response(self.prompt_question.format(vlm_question), rgb_im, depth_im).strip(".")

                logging.info(f"Pred - Prob: {smx_vlm_pred}")
                answering_time = time.time() - t
                result["step"][cnt_step]["answering_time"] = float(answering_time)

                # save data
                result["step"][cnt_step]["smx_vlm_rel"] = smx_vlm_rel
                result["step"][cnt_step]["smx_vlm_pred"] = smx_vlm_pred
                result["step"][cnt_step]["is_success"] = smx_vlm_pred == answer
                
                # 如果有信心回答，则直接获取答案
                if smx_vlm_rel.lower() in self.confident_threshold:
                    break

                # Get frontier candidates
                prompt_points_pix = []
                if self.cfg.use_active:
                    t = time.time()
                    prompt_points_pix, fig = (
                        tsdf_planner.find_prompt_points_within_view(
                            pts_normal,
                            self.img_width,
                            self.img_height,
                            self.cam_intr,
                            cam_pose_tsdf,
                            **self.cfg.visual_prompt,
                        )
                    )
                    planner_time = (time.time() - t)
                    result["step"][cnt_step]["planner_time"] = float(planner_time)
                    fig.tight_layout()
                    plt.savefig(os.path.join(episode_data_dir, "prompt_points.png".format(cnt_step)))
                    plt.close()

                # Visual prompting
                actual_num_prompt_points = len(prompt_points_pix)
                if actual_num_prompt_points >= self.cfg.visual_prompt.min_num_prompt_points:
                    rgb_im_draw = draw_letters(rgb_im, 
                                               prompt_points_pix, 
                                               self.letters, 
                                               self.cfg.visual_prompt.circle_radius, 
                                               self.fnt, 
                                               os.path.join(episode_data_dir, f"{cnt_step}_draw.png"))

                    # get VLM reasoning for exploring
                    t = time.time()
                    if self.cfg.use_lsv:
                        if self.cfg.rag.use_rag:
                            kb, _ = self.knowledge_base.search(self.prompt_lsv.format(question), 
                                                       rgb_im, 
                                                       top_k=self.cfg.rag.max_retrieval_num if cnt_step > self.cfg.rag.max_retrieval_num else cnt_step+1,
                                                       device=self.device)
                        response = self.vlm.get_response(rgb_im_draw, self.prompt_lsv.format(question), kb, device=self.device)[0]
                        # response = self.vlm.get_response(self.prompt_lsv.format(question), rgb_im_draw, depth_im)
                        lsv = np.zeros(actual_num_prompt_points)
                        for i in range(actual_num_prompt_points):
                            if response == self.letters[i]:
                                lsv[i] = 1
                        lsv *= actual_num_prompt_points / 3
                    else:
                        lsv = (
                            np.ones(actual_num_prompt_points) / actual_num_prompt_points
                        )

                    # base - use image without label
                    if self.cfg.use_gsv:
                        if self.cfg.rag.use_rag:
                            kb, _ = self.knowledge_base.search(self.prompt_gsv.format(question), 
                                                       rgb_im, 
                                                       top_k=self.cfg.rag.max_retrieval_num if cnt_step > self.cfg.rag.max_retrieval_num else cnt_step+1,
                                                       device=self.device)
                        response = self.vlm.get_response(rgb_im, self.prompt_gsv.format(question), kb, device=self.device)[0].strip(".")
                        # response = self.vlm.get_response(self.prompt_gsv.format(question), rgb_im, depth_im).strip(".")
                        gsv = np.zeros(2)
                        if response.lower() == "yes":
                            gsv[0] = 1
                        else:
                            gsv[1] = 1
                        gsv = (np.exp(gsv[0] / self.cfg.gsv_T) / self.cfg.gsv_F)  # scale before combined with lsv
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
                    planner_time += (time.time() - t)
                    result["step"][cnt_step]["planner_time"] = float(planner_time)

            else:
                logging.info("Skipping black image!")

            # Determine next point
            if cnt_step < num_step:
                pts_normal, angle, cur_angle, pts_pix, fig = tsdf_planner.find_next_pose(
                    pts=pts_normal,
                    angle=angle,
                    cam_pose=cam_pose_tsdf,
                    flag_no_val_weight=cnt_step < self.cfg.min_random_init_steps,
                    **self.cfg.planner,
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

        time_comsume += (time.time() - all_time)

        # Final prediction
        if cnt_step == num_step - 1:
            logging.info("Max step reached!")
            if self.cfg.rag.use_rag:
                kb, _ = self.knowledge_base.search(self.prompt_question.format(vlm_question), 
                                           rgb_im, 
                                           top_k=self.cfg.rag.max_retrieval_num if cnt_step > self.cfg.rag.max_retrieval_num else cnt_step+1,
                                           device=self.device)
            smx_vlm_pred = self.vlm.get_response(rgb_im, self.prompt_question.format(vlm_question), kb, device=self.device)[0].strip(".")
            # smx_vlm_pred = self.vlm.get_response(self.prompt_question.format(vlm_question), rgb_im, depth_im).strip(".")
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
        result["summary"]["is_success"] = is_success
        # result["summary"]["input_token_usage"] = int(self.vlm.input_token_usage)
        # result["summary"]["output_token_usage"] = int(self.vlm.output_token_usage)
        result["summary"]["path_length"] = float(path_length)
        result["summary"]["all_time_comsume"] = float(time_comsume)

        return result


if __name__ == "__main__":
    import argparse
    import os
    import logging
    from omegaconf import OmegaConf

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--cfg_file", help="cfg file path", default="cfg/vlm_exp_ov.yaml", type=str)
    parser.add_argument("-gpus", "--gpu_ids", help="Comma-separated GPU IDs to use (e.g., '0,1,2')", type=str, default="0")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    cfg.output_dir = os.path.join(cfg.output_parent_dir, f"{cfg.exp_name}/{cfg.exp_name}")
    memory_eqa = MemoryEQA(cfg, 2)

    with open(cfg.question_data_path) as f:
        questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]

    all_results = []
    for question_ind, data in enumerate(questions_data):
        result = memory_eqa.run(data, question_ind)
        all_results.append(result)

    with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=4)