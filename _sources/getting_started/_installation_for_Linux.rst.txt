======================
Installation for Linux
======================

System requirements
===================

Latest Model Optimizer (``nvidia-modelopt``) currently has the following system requirements:

+-------------------------+-----------------------------+
| OS                      |  Linux                      |
+-------------------------+-----------------------------+
| Architecture            |  x86_64, aarch64 (SBSA)     |
+-------------------------+-----------------------------+
| Python                  |  >=3.9,<3.13                |
+-------------------------+-----------------------------+
| CUDA                    |  >=12.0                     |
+-------------------------+-----------------------------+
| PyTorch (Optional)      |  >=2.2                      |
+-------------------------+-----------------------------+
| TensorRT-LLM (Optional) |  0.17                       |
+-------------------------+-----------------------------+
| ONNX Runtime (Optional) |  1.20 (Python>=3.10)        |
+-------------------------+-----------------------------+
| TensorRT (Optional)     |  >=10.0                     |
+-------------------------+-----------------------------+

Environment setup
=================

.. tab:: Docker image (Recommended)

    **Using ModelOpt's docker image**

    Easiest way to get started with using Model Optimizer and additional dependencies (e.g. TensorRT-LLM deployment) is to start from our docker image.

    After installing the `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_,
    please run the following commands to build the Model Optimizer docker container which has all the necessary
    dependencies pre-installed for running the examples.

    .. code-block:: shell

        # Clone the ModelOpt repository
        git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer.git
        cd TensorRT-Model-Optimizer

        # Build the docker (will be tagged `docker.io/library/modelopt_examples:latest`)
        # You may customize `docker/Dockerfile` to include or exclude certain dependencies you may or may not need.
        bash docker/build.sh

        # Run the docker image
        docker run --gpus all -it --shm-size 20g --rm docker.io/library/modelopt_examples:latest bash

        # Check installation (inside the docker container)
        python -c "import modelopt; print(modelopt.__version__)"

    **Using alternative NVIDIA docker images**

    For PyTorch, you can also use `NVIDIA NGC PyTorch container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags>`_
    and for NVIDIA NeMo framework, you can use the `NeMo container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags>`_.
    Both of these containers come with Model Optimizer pre-installed. NeMo container also comes with the HuggingFace and TensorRT-LLM
    dependencies. Make sure to update the Model Optimizer to the latest version if not already.

.. tab:: Local environment (PIP / Conda)

    **Setting up a virtual environment**

    We recommend setting up a virtual environment if you don't have one already. Run the following
    command to set up and activate a ``conda`` virtual environment named ``modelopt`` with Python 3.12:

    .. code-block:: shell

        conda create -n modelopt python=3.12 pip
        conda activate modelopt

    (Optional) **Install desired PyTorch version**

    By default, the latest PyTorch version available on ``pip`` will
    be installed. If you want to install a specific PyTorch version for a specific CUDA version, please first
    follow the instructions to `install your desired PyTorch version <https://pytorch.org/get-started/locally/>`_.

    (Optional) **Install other NVIDIA dependencies**

    If you wish to use ModelOpt in conjunction with other NVIDIA libraries (e.g. TensorRT, TensorRT-LLM, NeMo, Triton, etc.),
    please make sure to check the ease of installation of these libraries in a local environment. If you face any
    issues, we recommend using a docker image for a seamless experience. For example, `TensorRT-LLM documentation <https://nvidia.github.io/TensorRT-LLM/>`_.
    requires installing in a docker image. You may still choose to use other ModelOpt's features locally for example,
    quantizing a HuggingFace model and then use a docker image for deployment.

Install Model Optimizer
=======================

ModelOpt including its dependencies can be installed via ``pip``. Please review the license terms of ModelOpt and any
dependencies before use.

If you build and use ModelOpt's docker image, you can skip this step as the image already contains ModelOpt and all
optional dependencies pre-installed.
If you use other suggested docker images, ModelOpt is pre-installed with some of the below optional dependencies.
Make sure to upgrade to the latest version of ModelOpt (with appropriate optional dependencies you need) using pip as shown below.

.. code-block:: bash

    pip install "nvidia-modelopt[all]" -U --extra-index-url https://pypi.nvidia.com

If you want to install only partial dependencies, please replace ``[all]`` with the desired
optional dependencies as described below.

**Identify correct partial dependencies**

Note that when installing ``nvidia-modelopt`` without any optional dependencies, only the barebone
requirements are installed and none of the modules will work without the appropriate optional
dependencies or ``[all]`` optional dependencies. Below is a list of optional dependencies that
need to be installed to correctly use the corresponding modules:

.. list-table::
    :widths: 30 30
    :header-rows: 1

    *   - Module
        - Optional dependencies
    *   - ``modelopt.deploy``
        - ``[deploy]``
    *   - ``modelopt.onnx``
        - ``[onnx]``
    *   - ``modelopt.torch``
        - ``[torch]``
    *   - ``modelopt.torch._deploy``
        - ``[torch, deploy]``

Additionally, we support installing dependencies for following 3rd-party packages:

.. list-table::
    :widths: 30 30
    :header-rows: 1

    *   - Third-party package
        - Optional dependencies
    *   - Huggingface (``transformers``, ``diffusers``, etc.)
        - ``[hf]``

**Accelerated Quantization with Triton Kernels**

ModelOpt includes optimized quantization kernels implemented with Triton language that accelerate quantization
operations by approximately 40% compared to the default implementation. These kernels are particularly
beneficial for :doc:`AWQ <../guides/_choosing_quant_methods>` and Quantization-aware Training (QAT) workflows.

The Triton-based kernels currently support the NVFP4 quantization format, with support for additional
formats coming in future releases. To use these accelerated kernels, you need:

* CUDA device with compute capability >= 8.9 (e.g. RTX 40 series, RTX 6000, NVIDIA L40 or later)
* Triton package installed: ``pip install triton``

No additional configuration is required - the optimized kernels are used automatically when available
for your hardware and quantization format.

Check installation
==================

.. tip::

    When you use ModelOpt's PyTorch quantization APIs for the first time, it will compile the fast quantization kernels
    using your installed torch and CUDA if available.
    This may take a few minutes but subsequent quantization calls will be much faster.
    To invoke the compilation and check if it is successful or pre-compile for docker builds, run the following command:

    .. code-block:: bash

        python -c "import modelopt.torch.quantization.extensions as ext; ext.precompile()"
