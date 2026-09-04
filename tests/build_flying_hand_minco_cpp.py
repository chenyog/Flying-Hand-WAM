#!/usr/bin/env python3
"""Build the self-contained in-process C++ MINCO optimizer for tests/runtime.

The original header-only MINCO and L-BFGS implementation is vendored under
``envs/flying_hand/cpp``. Eigen remains the only system header dependency
(Ubuntu package: ``libeigen3-dev``).
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CPP_SOURCE_DIR = REPOSITORY_ROOT / "envs/flying_hand/cpp"
SOURCE = CPP_SOURCE_DIR / "minco_optimizer.cpp"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    compiler = shutil.which(args.compiler)
    if compiler is None:
        raise FileNotFoundError(f"C++ compiler not found: {args.compiler}")

    eigen_include = Path("/usr/include/eigen3")
    if not (eigen_include / "Eigen/Core").is_file():
        raise FileNotFoundError(
            "Eigen headers were not found. Install them with: "
            "apt-get install libeigen3-dev"
        )
    try:
        import pybind11
    except ImportError as error:
        raise ImportError(
            "pybind11 is required. Install script/requirements.txt in the "
            "project virtual environment."
        ) from error

    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not extension_suffix:
        raise RuntimeError("Python did not report an extension-module suffix")
    output = REPOSITORY_ROOT / f"envs/flying_hand/_minco_cpp{extension_suffix}"

    optimization_flags = ["-O0", "-g"] if args.debug else ["-O3", "-DNDEBUG"]
    command = [
        compiler,
        "-std=c++17",
        "-shared",
        "-fPIC",
        *optimization_flags,
        f"-I{pybind11.get_include()}",
        f"-I{sysconfig.get_path('include')}",
        f"-I{eigen_include}",
        f"-I{CPP_SOURCE_DIR}",
        str(SOURCE),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)
    print(output)


if __name__ == "__main__":
    main()
