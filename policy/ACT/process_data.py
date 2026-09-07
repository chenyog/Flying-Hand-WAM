import sys

sys.path.append("./policy/ACT/")

import os
import h5py
import numpy as np
import pickle
import cv2
import argparse
import pdb
import json
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        if "flying_hand" in root:
            return {
                "robot_type": "flying_hand",
                "state": root["flying_hand/actual_state"][()].astype(np.float32),
                "action": root["flying_hand/target_state"][()].astype(np.float32),
                "head_camera": root["observation/head_camera/rgb"][()],
                "wrist_camera": root["observation/wrist_camera/rgb"][()],
            }
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def _episode_number(path):
    match = re.fullmatch(r"episode(\d+)\.hdf5", path.name)
    if match is None:
        raise ValueError(f"Unexpected episode filename: {path}")
    return int(match.group(1))


def discover_flying_hand_episodes(data_root, task_config):
    """Return every episode in a deterministic, global index space."""

    task_roots = sorted(data_root.glob(f"*/{task_config}/data"))
    if not task_roots:
        raise FileNotFoundError(
            f"No Flying-Hand task data found below {data_root} for config {task_config!r}"
        )

    all_episodes = []
    for task_root in task_roots:
        episodes = sorted(
            task_root.glob("episode*.hdf5"), key=_episode_number
        )
        actual = [_episode_number(path) for path in episodes]
        expected = list(range(len(episodes)))
        if actual != expected:
            raise FileNotFoundError(
                f"{task_root}: expected contiguous episodes 0..{len(episodes) - 1}, "
                f"got {actual}"
            )
        all_episodes.extend(episodes)

    return all_episodes


def write_flying_hand_episode(source, hdf5path, source_label):
    """Write one Flying-Hand source episode in the ACT HDF5 layout."""
    actual = source["state"]
    target = source["action"]
    if actual.shape != target.shape or actual.ndim != 2 or actual.shape[1] != 5:
        raise ValueError(
            f"Invalid Flying-Hand state/action shape in {source_label}: "
            f"{actual.shape}, {target.shape}"
        )
    length = actual.shape[0]
    if len(source["head_camera"]) != length or len(source["wrist_camera"]) != length:
        raise ValueError(f"Flying-Hand image/state length mismatch in {source_label}")
    if length < 2:
        raise ValueError(f"Flying-Hand episode has no transition: {source_label}")

    cam_high = []
    cam_wrist = []
    for index in range(length - 1):
        head = cv2.imdecode(
            np.frombuffer(source["head_camera"][index], np.uint8), cv2.IMREAD_COLOR
        )
        wrist = cv2.imdecode(
            np.frombuffer(source["wrist_camera"][index], np.uint8), cv2.IMREAD_COLOR
        )
        if head is None or wrist is None:
            raise ValueError(
                f"Failed to decode Flying-Hand image in {source_label}, frame {index}"
            )
        # ACT batches the camera tensors together, so both views use one shape.
        head = cv2.resize(head, (640, 480), interpolation=cv2.INTER_LINEAR)
        wrist = cv2.resize(wrist, (640, 480), interpolation=cv2.INTER_LINEAR)
        cam_high.append(cv2.cvtColor(head, cv2.COLOR_BGR2RGB))
        cam_wrist.append(cv2.cvtColor(wrist, cv2.COLOR_BGR2RGB))

    with h5py.File(hdf5path, "w") as f:
        f.create_dataset("action", data=target[1:].astype(np.float32))
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=actual[:-1].astype(np.float32))
        image = obs.create_group("images")
        image.create_dataset("cam_high", data=np.stack(cam_high), dtype=np.uint8)
        image.create_dataset("cam_wrist", data=np.stack(cam_wrist), dtype=np.uint8)


def _convert_one_flying_hand_episode(task):
    output_index, source_path, save_path = task
    source = load_hdf5(str(source_path))
    if not isinstance(source, dict) or source.get("robot_type") != "flying_hand":
        raise ValueError(f"Not a Flying-Hand episode: {source_path}")
    output_path = os.path.join(save_path, f"episode_{output_index}.hdf5")
    write_flying_hand_episode(source, output_path, str(source_path))
    return output_index, source_path


def data_transform_flying_hand_paths(paths, save_path, num_workers=1):
    """Merge multiple Flying-Hand task directories into one ACT dataset."""
    if not paths:
        raise ValueError("No Flying-Hand episodes selected")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    tasks = [(index, path, save_path) for index, path in enumerate(paths)]
    if num_workers == 1:
        results = map(_convert_one_flying_hand_episode, tasks)
        for output_index, source_path in results:
            print(
                f"process Flying-Hand episode {output_index + 1}/{len(paths)} "
                f"success: {source_path}"
            )
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # executor.map preserves input order while work runs in parallel.
            for output_index, source_path in executor.map(
                _convert_one_flying_hand_episode, tasks
            ):
                print(
                    f"process Flying-Hand episode {output_index + 1}/{len(paths)} "
                    f"success: {source_path}"
                )
    return len(paths)


def data_transform(path, episode_num, save_path):
    begin = 0
    is_flying_hand = False
    floders = os.listdir(path)
    assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):
        source = load_hdf5(os.path.join(path, f"episode{i}.hdf5"))
        if isinstance(source, dict) and source.get("robot_type") == "flying_hand":
            is_flying_hand = True
            hdf5path = os.path.join(save_path, f"episode_{i}.hdf5")
            write_flying_hand_episode(source, hdf5path, f"episode{i}")
            begin += 1
            print(f"process Flying-Hand episode {i} success!")
            continue

        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = (load_hdf5(
            os.path.join(path, f"episode{i}.hdf5")))
        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        last_state = None
        for j in range(0, left_gripper_all.shape[0]):

            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )

            if j != left_gripper_all.shape[0] - 1:
                state = np.concatenate((left_arm, [left_gripper], right_arm, [right_gripper]), axis=0)  # joint

                state = state.astype(np.float32)
                qpos.append(state)

                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_high_resized = cv2.resize(camera_high, (640, 480))
                cam_high.append(camera_high_resized)

                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_resized = cv2.resize(camera_right_wrist, (640, 480))
                cam_right_wrist.append(camera_right_wrist_resized)

                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_resized = cv2.resize(camera_left_wrist, (640, 480))
                cam_left_wrist.append(camera_left_wrist_resized)

            if j != 0:
                action = state
                actions.append(action)
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])

        hdf5path = os.path.join(save_path, f"episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            image = obs.create_group("images")
            # cam_high_enc, len_high = images_encoding(cam_high)
            # cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
            # cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
            image.create_dataset("cam_high", data=np.stack(cam_high), dtype=np.uint8)
            image.create_dataset("cam_right_wrist", data=np.stack(cam_right_wrist), dtype=np.uint8)
            image.create_dataset("cam_left_wrist", data=np.stack(cam_left_wrist), dtype=np.uint8)

        begin += 1
        print(f"proccess {i} success!")

    return begin, is_flying_hand


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument("task_name", type=str, nargs="?")
    parser.add_argument("task_config", type=str, nargs="?")
    parser.add_argument("expert_data_num", type=int, nargs="?")
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Merge episodes from every Flying-Hand task",
    )
    parser.add_argument(
        "--task-config",
        dest="all_task_config",
        default="flying_hand_clean",
        help="Task config to select for --all-tasks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for --all-tasks",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel episode conversion workers (default: 1)",
    )

    args = parser.parse_args()

    if args.all_tasks:
        if args.task_name or args.task_config or args.expert_data_num:
            parser.error("--all-tasks cannot be combined with positional task arguments")

        repo_root = Path(__file__).resolve().parents[2]
        data_root = repo_root / "data" / "flying_hand"
        output = args.output or (
            Path("processed_data")
            / "sim-flying_hand"
            / "all-tasks"
        )
        paths = discover_flying_hand_episodes(data_root, args.all_task_config)
        begin = data_transform_flying_hand_paths(
            paths, str(output), num_workers=args.workers
        )
        sim_task_name = "sim-flying_hand-all-tasks"
        sim_config = {
            "dataset_dir": str(output),
            "num_episodes": begin,
            "episode_len": 1000,
            "camera_names": ["cam_high", "cam_wrist"],
            "robot_type": "flying_hand",
            "action_dim": 5,
            "source_task_config": args.all_task_config,
        }
        SIM_TASK_CONFIGS_PATH = "./SIM_TASK_CONFIGS.json"
        try:
            with open(SIM_TASK_CONFIGS_PATH, "r") as f:
                SIM_TASK_CONFIGS = json.load(f)
        except Exception:
            SIM_TASK_CONFIGS = {}
        SIM_TASK_CONFIGS[sim_task_name] = sim_config
        with open(SIM_TASK_CONFIGS_PATH, "w") as f:
            json.dump(SIM_TASK_CONFIGS, f, indent=4)
        with open(output / "conversion_complete.json", "w") as f:
            json.dump({"num_episodes": begin, "task_config": args.all_task_config}, f, indent=2)
        print(f"processed {begin} episodes into {output}")
        raise SystemExit(0)

    if not args.task_name or not args.task_config or args.expert_data_num is None:
        parser.error(
            "single-task mode requires TASK_NAME TASK_CONFIG EXPERT_DATA_NUM, "
            "or use --all-tasks"
        )

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num

    repo_root = Path(__file__).resolve().parents[2]
    flying_hand_data = repo_root / "data" / "flying_hand" / task_name / task_config / "data"
    generic_data = repo_root / "data" / task_name / task_config / "data"
    source_data = flying_hand_data if flying_hand_data.is_dir() else generic_data

    begin, is_flying_hand = data_transform(
        str(source_data),
        expert_data_num,
        f"processed_data/sim-{task_name}/{task_config}-{expert_data_num}",
    )

    SIM_TASK_CONFIGS_PATH = "./SIM_TASK_CONFIGS.json"

    try:
        with open(SIM_TASK_CONFIGS_PATH, "r") as f:
            SIM_TASK_CONFIGS = json.load(f)
    except Exception:
        SIM_TASK_CONFIGS = {}

    sim_task_name = f"sim-{task_name}-{task_config}-{expert_data_num}"
    SIM_TASK_CONFIGS[sim_task_name] = {
        "dataset_dir": f"./processed_data/sim-{task_name}/{task_config}-{expert_data_num}",
        "num_episodes": expert_data_num,
        "episode_len": 1000,
        "camera_names": ["cam_high", "cam_wrist"] if is_flying_hand else [
            "cam_high", "cam_right_wrist", "cam_left_wrist"
        ],
    }
    if is_flying_hand:
        SIM_TASK_CONFIGS[sim_task_name]["robot_type"] = "flying_hand"
        SIM_TASK_CONFIGS[sim_task_name]["action_dim"] = 5

    with open(SIM_TASK_CONFIGS_PATH, "w") as f:
        json.dump(SIM_TASK_CONFIGS, f, indent=4)
