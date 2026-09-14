import sys
import os
import subprocess
import json
import hashlib
import gc
import random

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH, load_task_class
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

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


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def get_eval_video_sizes(args):
    camera_args = args["camera"]
    video_cameras = camera_args.get("video_cameras")
    if not video_cameras:
        video_cameras = [
            key.removeprefix("collect_")
            for key, value in camera_args.items()
            if key.startswith("collect_") and value
        ]
    if not video_cameras:
        video_cameras = [
            key.removesuffix("_type")
            for key in camera_args
            if key.endswith("_camera_type")
        ]

    video_sizes = {}
    for camera_name in video_cameras:
        type_key = f"{camera_name}_type"
        if type_key not in camera_args:
            continue
        camera_config = get_camera_config(camera_args[type_key])
        video_sizes[camera_name] = str(camera_config["w"]) + "x" + str(camera_config["h"])
    return video_sizes


def get_eval_video_fps(args):
    timestep = float(args["timestep"])
    save_freq = float(args["save_freq"])
    if timestep <= 0 or save_freq <= 0:
        raise ValueError(f"`timestep` and `save_freq` must be positive, got {timestep=} {save_freq=}")
    fps = 1.0 / (timestep * save_freq)
    return int(round(fps)) if abs(fps - round(fps)) < 1e-6 else fps


def save_flying_hand_eval_history(
    task_env,
    *,
    task_name,
    task_config,
    episode,
    seed,
    instruction,
    success,
    video_sizes,
    fps,
):
    """Save lightweight 20 Hz replay data without encoding RGB during eval."""
    chunks = list(task_env.flying_hand_eval_action_chunks)
    actions = (
        np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        if chunks
        else np.empty((0, 5), dtype=np.float32)
    )
    chunk_lengths = np.asarray([len(chunk) for chunk in chunks], dtype=np.int32)
    history_stem = Path(task_env.eval_video_path) / f"episode{episode}_history"
    np.savez_compressed(
        history_stem.with_suffix(".npz"),
        actions=actions,
        chunk_lengths=chunk_lengths,
        actual_states=np.asarray(task_env.flying_hand_eval_actual_states, dtype=np.float32),
        target_states=np.asarray(task_env.flying_hand_eval_target_states, dtype=np.float32),
    )
    metadata = {
        "format_version": 2,
        "task_name": str(task_name),
        "task_namespace": "flying_hand",
        "task_config": str(task_config),
        "episode": int(episode),
        "seed": int(seed),
        "instruction": str(instruction),
        "success": bool(success),
        "fps": float(fps),
        "timestep": float(task_env.sim_timestep),
        "save_freq": int(task_env.save_freq),
        "enable_dynamics": bool(task_env.enable_dynamics),
        "render_shader": str(task_env.render_shader),
        "render_samples_per_pixel": int(task_env.render_samples_per_pixel),
        "render_path_depth": int(task_env.render_path_depth),
        "render_denoiser": str(task_env.render_denoiser),
        "video_sizes": dict(video_sizes),
        "action_count": int(task_env.take_action_cnt),
    }
    history_stem.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return history_stem.with_suffix(".npz")


def rerender_flying_hand_eval_histories(replay_dir, video_dir):
    """Render all saved Flying-Hand episodes after policy evaluation finishes."""
    replay_dir = Path(replay_dir)
    histories = sorted(replay_dir.glob("episode*_history.npz"))
    if not histories:
        raise RuntimeError(f"No Flying-Hand replay histories found in {replay_dir}")
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("rerender_flying_hand_eval.py")),
            str(replay_dir),
            "--output-dir",
            str(video_dir),
        ],
        check=True,
    )
    return len(histories)


def apply_task_overrides(task_args, usr_args):
    """Apply safe task overrides while preserving the Flying-Hand sim contract."""
    protected_flying_hand_keys = {
        "timestep",
        "save_freq",
        "render_shader",
        "texture_type",
        "enable_dynamics",
        "embodiment",
        "camera",
        "domain_randomization",
        "flying_hand_dynamics",
    }
    is_flying_hand = "flying-hand" in task_args.get("embodiment", [])
    for key in task_args.keys() & usr_args.keys():
        if is_flying_hand and key in protected_flying_hand_keys:
            if usr_args[key] != task_args[key]:
                raise ValueError(
                    f"Flying-Hand simulation field {key!r} is owned by the selected "
                    f"task YAML ({task_args[key]!r}); deployment requested {usr_args[key]!r}. "
                    "Select the matching direct/dynamics task_config instead."
                )
            continue
        task_args[key] = usr_args[key]


def flying_hand_simulation_contract(task_args, instruction_type):
    """Return a stable, policy-independent record of the eval environment."""
    keys = (
        "task_config",
        "timestep",
        "save_freq",
        "render_shader",
        "texture_type",
        "enable_dynamics",
        "embodiment",
        "camera",
        "domain_randomization",
        "flying_hand_dynamics",
    )
    contract = {key: task_args[key] for key in keys}
    contract["instruction_type"] = str(instruction_type)
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return {
        "format_version": 1,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "contract": contract,
    }


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    video_sizes = None
    eval_video_fps = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    apply_task_overrides(args, usr_args)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    for key, camera_type in args["camera"].items():
        if not key.endswith("_camera_type"):
            continue
        camera_name = key.removesuffix("_type")
        args[f"{camera_name}_h"] = _camera_config[camera_type]["h"]
        args[f"{camera_name}_w"] = _camera_config[camera_type]["w"]

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
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    eval_output_dir = usr_args.get("eval_output_dir")
    if eval_output_dir is None or str(eval_output_dir).strip().lower() in {"", "none", "null"}:
        save_dir = Path(f"eval_results/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    else:
        save_dir = Path(eval_output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    is_flying_hand = "flying-hand" in embodiment_type
    if is_flying_hand:
        contract = flying_hand_simulation_contract(args, instruction_type)
        (save_dir / "_simulation_contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Flying-Hand simulation contract: {contract['sha256']}")

    video_sizes = {}
    eval_video_fps = None
    replay_dir = None
    video_dir = None
    if args["eval_video_log"]:
        video_sizes = get_eval_video_sizes(args)
        eval_video_fps = get_eval_video_fps(args)
        if is_flying_hand:
            replay_dir = save_dir / "replay"
            video_dir = save_dir / "video"
            replay_dir.mkdir(parents=True, exist_ok=True)
            args["eval_video_save_dir"] = replay_dir
        else:
            args["eval_video_save_dir"] = save_dir

    # output camera config
    print("============= Config =============\n")
    if "cluttered_board" in args["domain_randomization"]:
        print("\033[95mCluttered Board:\033[0m " + str(args["domain_randomization"]["cluttered_board"]))
    if "cluttered_table" in args["domain_randomization"]:
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

    task_namespace = "flying_hand" if "flying-hand" in embodiment_type else None
    TASK_ENV = class_decorator(args["task_name"], task_namespace=task_namespace)
    args["task_namespace"] = task_namespace
    args["policy_name"] = policy_name
    # Policy adapters must not infer the robot from a task name such as
    # ``move_bottle``.  The selected task YAML is the authoritative source for
    # the embodiment, so expose one common, explicit deployment field.
    usr_args["robot_type"] = "flying_hand" if is_flying_hand else "robotwin"
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"].get("arm_joints_name", [[]])[0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"].get("arm_joints_name", [[], []])[1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = int(usr_args.get("eval_num_episodes", 100))
    topk = 1

    model = get_model(usr_args)
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_sizes=video_sizes,
                                   eval_video_fps=eval_video_fps,
                                   instruction_type=instruction_type,
                                   diagnostics_path=save_dir / "policy_diagnostics.jsonl")
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = save_dir / "_result.txt"
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    if is_flying_hand and args["eval_video_log"]:
        # Rendering starts only after policy inference has completely finished.
        # Releasing the model first leaves as much GPU memory as possible for RT/OIDN.
        del model
        gc.collect()
        torch_module = sys.modules.get("torch")
        if torch_module is not None and torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
        print(f"Evaluation complete; rendering all histories from {replay_dir}")
        rendered = rerender_flying_hand_eval_histories(replay_dir, video_dir)
        print(f"Rendered {rendered} complete episode(s) to {video_dir}")

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_sizes=None,
                eval_video_fps=None,
                instruction_type=None,
                diagnostics_path=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print(
                    f"Skipping invalid expert seed for task={task_name!r}, "
                    f"seed={now_seed - 1}: {type(e).__name__}: {e}"
                )
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        random.seed(now_seed)
        results = generate_episode_descriptions(
            args["task_name"],
            episode_info_list,
            test_num,
            instruction_namespace=args.get("task_namespace"),
        )
        # Flying-Hand training converters use the first generated instruction
        # for each episode, so evaluation must use the same deterministic text.
        instruction = results[0][instruction_type][0]
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        is_flying_hand = "flying-hand" in args.get("embodiment", [])
        record_episode_history = is_flying_hand and TASK_ENV.eval_video_path is not None
        live_episode_video = not is_flying_hand and TASK_ENV.eval_video_path is not None
        if record_episode_history:
            TASK_ENV.record_flying_hand_eval_history = True
            TASK_ENV.flying_hand_eval_action_chunks = []
            TASK_ENV.flying_hand_eval_actual_states = [
                np.asarray(TASK_ENV._get_flying_hand_actual_state(), dtype=np.float32)
            ]
            TASK_ENV.flying_hand_eval_target_states = [
                np.asarray(TASK_ENV._get_flying_hand_target_state(), dtype=np.float32)
            ]
        if live_episode_video:
            TASK_ENV.eval_video_frame_limit = None
            TASK_ENV.eval_video_frames_written = 0
            ffmpegs = {}
            for camera_name, video_size in video_sizes.items():
                safe_camera_name = camera_name.replace("/", "_")
                ffmpegs[camera_name] = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pixel_format",
                        "rgb24",
                        "-video_size",
                        video_size,
                        "-framerate",
                        str(eval_video_fps),
                        "-i",
                        "-",
                        "-pix_fmt",
                        "yuv420p",
                        "-vcodec",
                        "libx264",
                        "-crf",
                        "23",
                        f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}_{safe_camera_name}.mp4",
                    ],
                    stdin=subprocess.PIPE,
                )
            TASK_ENV._set_eval_video_ffmpeg(ffmpegs)

        succ = False
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if live_episode_video:
            TASK_ENV._del_eval_video_ffmpeg()

        if record_episode_history:
            history_path = save_flying_hand_eval_history(
                TASK_ENV,
                task_name=task_name,
                task_config=args["task_config"],
                episode=TASK_ENV.test_num,
                seed=now_seed,
                instruction=instruction,
                success=succ,
                video_sizes=video_sizes,
                fps=eval_video_fps,
            )
            TASK_ENV.record_flying_hand_eval_history = False
            print(f"Saved 20 Hz replay history: {history_path}")

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        if diagnostics_path is not None and hasattr(model, "get_episode_diagnostics"):
            episode_diagnostics = model.get_episode_diagnostics()
            episode_diagnostics.update({
                "task": task_name,
                "episode": int(now_id),
                "seed": int(now_seed),
                "success": bool(succ),
                "action_count": int(TASK_ENV.take_action_cnt),
            })
            with open(diagnostics_path, "a", encoding="utf-8") as diagnostics_file:
                diagnostics_file.write(json.dumps(episode_diagnostics, allow_nan=False) + "\n")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval_output_dir", type=str)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Parse overrides
    def parse_override_pairs(pairs):
        if len(pairs) % 2 != 0:
            parser.error("--overrides requires KEY VALUE pairs")
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = yaml.safe_load(pairs[i + 1])
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)
    if args.eval_output_dir is not None:
        config["eval_output_dir"] = args.eval_output_dir

    return config


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.test_render import SapienRenderSmokeTest
    if os.environ.get("SKIP_RENDER_SMOKE", "0").strip().lower() not in {"1", "true", "yes"}:
        SapienRenderSmokeTest()

    usr_args = parse_args_and_config()

    main(usr_args)
