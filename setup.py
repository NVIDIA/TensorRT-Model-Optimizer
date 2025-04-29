# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The package setup script for modelopt customizing certain aspects of the installation process."""

import os
import platform

import setuptools

# Package configuration ############################################################################
name = "nvidia-modelopt"
version = os.environ.get(
    "SETUPTOOLS_SCM_PRETEND_VERSION", "0.27.0" if platform.system() == "Windows" else "0.27.1"
)
packages = setuptools.find_namespace_packages(include=["modelopt*"])
package_dir = {"": "."}
package_data = {"modelopt": ["**/*.h", "**/*.cpp", "**/*.cu"]}
setup_kwargs = {}

# Required and optional dependencies ###############################################################
required_deps = [
    f"nvidia-modelopt-core=={version}",
    "ninja",  # for faster building of C++ / CUDA extensions
    "numpy",
    "packaging",
    "pydantic>=2.0",
    "rich",
    "scipy",
    "tqdm",
    # TODO: (temporary fix) Python 3.12+ venv does not install setuptools by default but it is
    #   required to load cuda extensions using torch.utils.cpp_extension.load
    # Torch will have a dependency on setuptools after github.com/pytorch/pytorch/pull/127921 (2.4+)
    "setuptools ; python_version >= '3.12'",
]

optional_deps = {
    "deploy": [],
    "onnx": [
        "cppimport",
        "cupy-cuda12x; platform_machine != 'aarch64' and platform_system != 'Darwin'",
        "onnx",
        "onnxconverter-common",
        "onnx-graphsurgeon",
        # Onnxruntime 1.20+ is not supported on Python 3.9
        "onnxruntime~=1.18.1 ; python_version < '3.10'",
        "onnxruntime~=1.20.1 ; python_version >= '3.10' and (platform_machine == 'aarch64' or platform_system == 'Darwin')",  # noqa: E501
        "onnxruntime-gpu~=1.20.1 ; python_version >= '3.10' and platform_machine != 'aarch64' and platform_system != 'Darwin' and platform_system != 'Windows'",  # noqa: E501
        "onnxruntime-directml==1.20.0; python_version >= '3.10' and platform_system == 'Windows'",
        "onnxsim ; python_version < '3.12' and platform_machine != 'aarch64'",
    ],
    "torch": [
        "pulp",
        "pynvml>=11.5.0",
        "regex",
        "safetensors",
        "torch>=2.2",
        "torchprofile>=0.0.4",
        "torchvision",
    ],
    "hf": [
        "accelerate>=1.0.0",
        "datasets>=3.0.0",
        "diffusers>=0.32.2",
        "huggingface_hub>=0.24.0",
        "peft>=0.12.0",
        # TODO: Upgrade to 4.51.0 after fixing unittests
        # User can post install 4.51.0 to use the new transformers features
        "transformers>=4.48.0,<4.51.0",
    ],
    # linter tools
    "dev-lint": [
        "bandit[toml]==1.7.9",  # security/compliance checks
        "mypy==1.14.1",
        "pre-commit==4.1.0",
        "ruff==0.9.4",
    ],
    # testing
    "dev-test": [
        "coverage",
        "onnxscript",  # For test_onnx_dynamo_export unit test
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "pytest-timeout",
        "timm",
        "toml",
        "tox",
        "tox-current-env>=0.0.12",  # Incompatible with tox==4.18.0
    ],
    # docs
    "dev-docs": [
        "autodoc_pydantic>=2.1.0",
        "ipython",
        "ipywidgets",
        "nbsphinx>=0.9.3",
        "pypandoc",  # Required by nbsphinx
        "sphinx~=7.2.0",  # AttributeError for sphinx~=7.3.0
        "sphinx-autobuild>=2024.10.3",
        "sphinx-copybutton>=0.5.2",
        "sphinx-inline-tabs>=2023.4.21",
        "sphinx-rtd-theme~=3.0.0",  # 3.0 does not show version, which we want as Linux & Windows have separate releases
        "sphinx-togglebutton>=0.3.2",
    ],
    # build/packaging tools
    "dev-build": [
        "cython",
        "setuptools>=67.8.0",
        "setuptools_scm>=7.1.0",
        "twine",
    ],
}

# create "compound" optional dependencies
optional_deps["all"] = [
    deps for k in optional_deps if not k.startswith("dev") for deps in optional_deps[k]
]
optional_deps["dev"] = [deps for k in optional_deps for deps in optional_deps[k]]


if __name__ == "__main__":
    setuptools.setup(
        name=name,
        version=version,
        description="Nvidia TensorRT Model Optimizer: a unified model optimization and deployment toolkit.",
        long_description="Checkout https://github.com/nvidia/TensorRT-Model-Optimizer for more information.",
        long_description_content_type="text/markdown",
        author="NVIDIA Corporation",
        url="https://github.com/NVIDIA/TensorRT-Model-Optimizer",
        license="Apache 2.0",
        license_files=("LICENSE",),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        python_requires=">=3.9,<3.13",
        install_requires=required_deps,
        extras_require=optional_deps,
        packages=packages,
        package_dir=package_dir,
        package_data=package_data,
        **setup_kwargs,
    )
