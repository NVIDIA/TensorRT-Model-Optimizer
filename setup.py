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

import glob
import os

import setuptools
from Cython.Build import cythonize
from setuptools.command.build_py import build_py
from setuptools.extension import Extension
from setuptools_scm import get_version

# Package configuration ############################################################################
name = "nvidia-modelopt"
# TODO: Set version to static stable release version when creating the release branch
# version = os.environ.get("SETUPTOOLS_SCM_PRETEND_VERSION", "X.Y.Z")
version = get_version(root=".", fallback_version="0.0.0")
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
]

optional_deps = {
    "deploy": [],
    "onnx": [
        "cppimport",
        "cupy-cuda12x; platform_machine != 'aarch64' and platform_system != 'Darwin'",
        "onnx>=1.18.0",
        "onnxconverter-common",
        "onnx-graphsurgeon",
        "onnxruntime~=1.22.0 ; platform_machine == 'aarch64' or platform_system == 'Darwin'",
        "onnxruntime-gpu~=1.22.0 ; platform_machine != 'aarch64' and platform_system != 'Darwin' and platform_system != 'Windows'",  # noqa: E501
        "onnxruntime-gpu==1.20.0; platform_system == 'Windows'",
        "onnxsim ; python_version < '3.12' and platform_machine != 'aarch64'",
        "polygraphy>=0.49.22",
    ],
    "torch": [
        "pulp",
        "nvidia-ml-py>=12",
        "regex",
        "safetensors",
        "torch>=2.4",
        "torchprofile>=0.0.4",
        "torchvision",
    ],
    "hf": [
        "accelerate>=1.0.0",
        "datasets>=3.0.0",
        "diffusers>=0.32.2",
        "huggingface_hub>=0.24.0",
        "peft>=0.12.0",
        "transformers>=4.48.0,<4.52",
    ],
    # linter tools
    "dev-lint": [
        "bandit[toml]==1.7.9",  # security/compliance checks
        "mypy==1.15.0",
        "pre-commit==4.2.0",
        "ruff==0.11.9",
    ],
    # testing
    "dev-test": [
        "coverage",
        "onnxscript",  # For test_onnx_dynamo_export unit test
        "pytest",
        "pytest-cov",
        "pytest-timeout",
        "timm",
        "tox",
        "tox-current-env>=0.0.12",  # Incompatible with tox==4.18.0
    ],
    # docs
    "dev-docs": [
        "autodoc_pydantic>=2.1.0",
        "sphinx~=8.1.0",
        "sphinx-argparse>=0.5.2",
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


# External release overwrites ######################################################################
# TODO: Remove this section before copying the setup.py to the modelopt github repository

# You can modify the installation process with the following env variables:
# - ``MODELOPT_EXTERNAL``: if set to ``true`` (Default is ``False``), external packages (excluding
#     `modelopt.core`) will be packaged into a wheel `nvidia-modelopt`. Also, internal dependencies
#     will not be installed or packaged. This is useful for external releases.
# - ``MODELOPT_CORE_EXTERNAL``: if set to ``true`` (Default is ``False``), only `modelopt.core` will
#     be compiled into a separate wheel `nvidia-modelopt-core`. This wheel will not have any pip
#     dependencies. This is useful for external releases.

MODELOPT_EXTERNAL = os.environ.get("MODELOPT_EXTERNAL", "false").lower() == "true"
MODELOPT_CORE_EXTERNAL = os.environ.get("MODELOPT_CORE_EXTERNAL", "false").lower() == "true"

assert not (MODELOPT_EXTERNAL and MODELOPT_CORE_EXTERNAL), (
    "Cannot set both `MODELOPT_EXTERNAL` and `MODELOPT_CORE_EXTERNAL` to true."
)


if MODELOPT_EXTERNAL:
    packages = setuptools.find_namespace_packages(
        include=[  # Modules for external release (everything except modelopt.core)
            "modelopt",  # __init__.py
            "modelopt.deploy*",
            "modelopt.onnx*",
            "modelopt.torch*",
        ]
    )
elif MODELOPT_CORE_EXTERNAL:
    name = "nvidia-modelopt-core"
    packages = ["modelopt_core"] + [
        f"modelopt_core.{p}" for p in setuptools.find_namespace_packages(where="modelopt/core")
    ]
    package_dir = {"modelopt_core": "modelopt/core"}
    package_data = {}
    required_deps = []
    optional_deps = {}

    # Cythonize all non-init files in modelopt_core
    compiled_files = [
        f.replace(os.sep, "/")  # Windows compatible
        for f in glob.iglob("modelopt/core/**/*.py", recursive=True)
        if not f.endswith("__init__.py")
    ]
    ext_modules = cythonize(
        [
            Extension(
                f.replace("modelopt/core", "modelopt_core").replace(".py", "").replace("/", "."),
                sources=[f],
            )
            for f in compiled_files
        ],
        compiler_directives={"language_level": "3"},
        build_dir="build/modelopt_core_build",
    )

    class ModeloptBuildPy(build_py):
        """A custom builder class to modify the python build process for regular installs.

        The build process is executed during ``pip install .``. This is also triggered in certain cases
        during editable installs, i.e., ``pip install -e .``, starting from Python 3.9+. One trigger is
        when new packages are discovered!
        """

        def find_package_modules(self, *args, **kwargs):
            """If a package exists as compiled version skip python version."""
            return [
                pm
                for pm in super().find_package_modules(*args, **kwargs)
                if pm[-1].replace(os.sep, "/") not in compiled_files
            ]

    setup_kwargs["ext_modules"] = ext_modules
    setup_kwargs["cmdclass"] = {"build_py": ModeloptBuildPy}
else:
    # remove nvidia-modelopt-core dependency for internal installations / wheels
    required_deps = [dep for dep in required_deps if not dep.startswith("nvidia-modelopt-core")]


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
        python_requires=">=3.10,<3.13",
        install_requires=required_deps,
        extras_require=optional_deps,
        packages=packages,
        package_dir=package_dir,
        package_data=package_data,
        **setup_kwargs,
    )
