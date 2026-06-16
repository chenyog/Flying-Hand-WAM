from .utils import *
from ._GLOBAL_CONFIGS import *
import importlib


def load_task_class(task_name):
    module_name = task_name.replace("-", "_").replace("/", ".")
    envs_module = importlib.import_module(f"envs.{module_name}")
    return getattr(envs_module, module_name.split(".")[-1])
