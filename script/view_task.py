import os
import sys
import time
import yaml
from argparse import ArgumentParser

sys.path.append("./")

from envs import *  # noqa: F403


def class_decorator(task_name):
    try:
        env_class = load_task_class(task_name)  # noqa: F405
        return env_class()
    except AttributeError:
        raise SystemExit(f"No such task: {task_name}")


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def load_task_args(task_name, task_config, render_freq):
    config_path = f"./task_config/{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["render_freq"] = render_freq
    args["collect_data"] = False
    args["save_data"] = False
    args["need_plan"] = True
    args["eval_mode"] = False
    args["save_path"] = os.path.join(args["save_path"], task_name, task_config)

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")  # noqa: F405
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(name):
        robot_file = embodiment_types[name]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"missing embodiment files for {name}")
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
        args["embodiment_name"] = f"{embodiment_type[0]}+{embodiment_type[1]}"
    else:
        raise RuntimeError("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def keep_viewer_open(task):
    if not getattr(task, "render_freq", 0) or not hasattr(task, "viewer"):
        return

    print("Viewer is open. Close the SAPIEN window or press Ctrl+C to exit.")
    while not task.viewer.closed:
        task._update_render()
        task.viewer.render()
        time.sleep(1 / 60)


def main():
    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--render-freq", type=int, default=1)
    parser.add_argument("--no-play", action="store_true")
    args_cli = parser.parse_args()

    task = class_decorator(args_cli.task_name)
    args = load_task_args(args_cli.task_name, args_cli.task_config, args_cli.render_freq)
    args["seed"] = args_cli.seed
    args["now_ep_num"] = args_cli.episode

    try:
        task.setup_demo(**args)
        if args_cli.no_play:
            keep_viewer_open(task)
        else:
            task.play_once()
            keep_viewer_open(task)
    finally:
        task.close_env()
        if getattr(task, "render_freq", 0) and hasattr(task, "viewer") and not task.viewer.closed:
            task.viewer.close()


if __name__ == "__main__":
    main()
