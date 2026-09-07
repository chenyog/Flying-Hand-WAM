import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import fcntl
import h5py
import subprocess
import time
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name, task_namespace=None):
    module_name = (
        f"{task_namespace}/{task_name}"
        if task_namespace is not None
        else task_name
    )
    try:
        env_class = load_task_class(module_name)
        env_instance = env_class()
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        raise SystemExit(f"No such task: {module_name}") from exc
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(
    task_name=None,
    task_config=None,
    episode_num=None,
    save_path=None,
    data_worker_index=0,
    data_worker_count=1,
):
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    embodiment_type = args.get("embodiment")
    task_namespace = "flying_hand" if "flying-hand" in embodiment_type else None
    task = class_decorator(task_name, task_namespace=task_namespace)
    args["task_namespace"] = task_namespace
    if episode_num is not None:
        if episode_num <= 0:
            raise ValueError("episode_num must be positive")
        args["episode_num"] = episode_num
    if save_path is not None:
        args["save_path"] = save_path
    if data_worker_count <= 0:
        raise ValueError("data_worker_count must be positive")
    if not 0 <= data_worker_index < data_worker_count:
        raise ValueError(
            "data_worker_index must be in [0, data_worker_count)"
        )
    args["data_worker_index"] = data_worker_index
    args["data_worker_count"] = data_worker_count

    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    if "flying-hand" in embodiment_type:
        print("\033[95mCluttered Board:\033[0m " + str(args["domain_randomization"]["cluttered_board"]))
    else:
        print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    if "random_table_height" in args["domain_randomization"]:
        print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["data_root"] = args["save_path"]
    args["save_path"] = os.path.join(
        args["data_root"],
        str(args["task_name"]),
        args["task_config"],
    )
    run(task, args)


def run(TASK_ENV, args):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        while suc_num < args["episode_num"]:
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                TASK_ENV.play_once()

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    seed_list.append(epid)
                    TASK_ENV.save_traj_data(suc_num)
                    suc_num += 1
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    fail_num += 1

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                # stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            epid += 1

            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        data_worker_index = args.get("data_worker_index", 0)
        data_worker_count = args.get("data_worker_count", 1)

        def episode_data_complete(idx):
            file_path = os.path.join(
                args["save_path"],
                "data",
                f"episode{idx}.hdf5",
            )
            if not os.path.exists(file_path):
                return False
            if args.get("task_namespace") != "flying_hand":
                return True

            try:
                with h5py.File(file_path, "r") as file:
                    dataset_paths = [
                        "flying_hand/actual_state",
                        "flying_hand/target_state",
                    ]
                    dataset_paths.extend(
                        f"observation/{camera_name}/rgb"
                        for camera_name in args["camera"]["video_cameras"]
                    )
                    frame_counts = [
                        file[path].shape[0]
                        for path in dataset_paths
                    ]
            except (KeyError, OSError):
                return False

            if min(frame_counts, default=0) <= 0:
                return False
            if len(set(frame_counts)) != 1:
                return False

            for camera_name in args["camera"]["video_cameras"]:
                video_path = os.path.join(
                    args["save_path"],
                    "video",
                    camera_name,
                    f"episode{idx}.mp4",
                )
                if not os.path.exists(video_path):
                    return False
                if os.path.getsize(video_path) == 0:
                    return False
            return True

        for episode_idx in range(
            data_worker_index,
            args["episode_num"],
            data_worker_count,
        ):
            if episode_data_complete(episode_idx):
                continue
            print(f"\033[34mTask name: {args['task_name']}\033[0m")

            TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

            traj_data = TASK_ENV.load_tran_data(episode_idx)
            args["left_joint_path"] = traj_data["left_joint_path"]
            args["right_joint_path"] = traj_data["right_joint_path"]
            TASK_ENV.set_path_lst(args)

            info_file_path = os.path.join(args["save_path"], "scene_info.json")

            info = TASK_ENV.play_once()
            os.makedirs(os.path.dirname(info_file_path), exist_ok=True)
            with open(info_file_path, "a+", encoding="utf-8") as file:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX)
                file.seek(0)
                contents = file.read()
                info_db = json.loads(contents) if contents.strip() else {}
                info_db[f"episode_{episode_idx}"] = info
                info_db = {
                    key: info_db[key]
                    for key in sorted(
                        info_db,
                        key=lambda value: int(
                            value.removeprefix("episode_")
                        ),
                    )
                }
                file.seek(0)
                file.truncate()
                json.dump(info_db, file, ensure_ascii=False, indent=4)
                file.flush()
                os.fsync(file.fileno())
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)

            TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
            TASK_ENV.merge_pkl_to_hdf5_video()
            TASK_ENV.remove_data_cache()
            assert TASK_ENV.check_success(), "Collect Error"

        if data_worker_count > 1:
            print(
                "Data shard complete; generate instructions after all "
                "workers finish."
            )
            return

        command = [
            sys.executable,
            os.path.join(
                parent_directory,
                "../description/utils/generate_episode_instructions.py",
            ),
            args["task_name"],
            args["task_config"],
            str(args["language_num"]),
            "--save-path",
            args["data_root"],
        ]
        if args["task_namespace"] is not None:
            command.extend([
                "--instruction-namespace",
                args["task_namespace"],
            ])
        subprocess.run(command, check=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(parent_directory, "..")))
    from tests.test_render import SapienRenderSmokeTest
    SapienRenderSmokeTest()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--episode-num", type=int)
    parser.add_argument("--save-path", type=str)
    parser.add_argument("--data-worker-index", type=int, default=0)
    parser.add_argument("--data-worker-count", type=int, default=1)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(
        task_name=task_name,
        task_config=task_config,
        episode_num=parser.episode_num,
        save_path=parser.save_path,
        data_worker_index=parser.data_worker_index,
        data_worker_count=parser.data_worker_count,
    )
