# SPDX-License-Identifier: Apache-2.0
"""Compile the Driver API bridge during image build, without importing it."""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import torch

if torch.__version__ != "2.11.0+cu130":
    raise RuntimeError("Engram bridge requires the validated torch 2.11.0+cu130 ABI")
root = Path(__file__).resolve().parents[1]
cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
setup(
    name="engram_native",
    ext_modules=[
        CppExtension(
            "engram_native_dev_492ecf45",
            [str(root / "native/engram_native.cc")],
            include_dirs=[str(cuda / "include")],
            library_dirs=[str(cuda / "lib64/stubs")],
            libraries=["cuda"],
            extra_compile_args=["-O0", "-g0", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
