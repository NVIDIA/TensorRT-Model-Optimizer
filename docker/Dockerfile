FROM nvidia/cuda:12.5.1-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get -y install python3.10 python3-pip python-is-python3 openmpi-bin libopenmpi-dev wget git git-lfs unzip jq

ARG PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"
ENV PIP_EXTRA_INDEX_URL=$PIP_EXTRA_INDEX_URL
ENV PIP_NO_CACHE_DIR=off

# Install the latest setuptools using pip
RUN rm -rf /usr/lib/python3/dist-packages/setuptools*
RUN pip install setuptools -U

# TensorRT LLM
ARG TRT_LLM_VERSION=0.12.0
RUN pip install "tensorrt-llm~=$TRT_LLM_VERSION" -U
RUN git clone --depth 1 --branch "v$TRT_LLM_VERSION" https://github.com/NVIDIA/TensorRT-LLM.git && \
    mkdir tensorrt-llm && \
    mv TensorRT-LLM/benchmarks/ tensorrt-llm && \
    rm -rf TensorRT-LLM

# ModelOpt installation with all optional dependencies.
ARG MODELOPT_VERSION=0.17.0
RUN pip install "nvidia-modelopt[all]~=$MODELOPT_VERSION" -U
RUN python -c "import modelopt"

# Export the path to 'libcudnn.so.X' needed by 'libonnxruntime_providers_tensorrt.so'
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# TensorRT dev environment installation.
ARG TENSORRT_URL=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.3.0/tars/TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.5.tar.gz
RUN wget -q -O tensorrt.tar.gz $TENSORRT_URL && \
    tar -xf tensorrt.tar.gz && \
    cp TensorRT-*/bin/trtexec /usr/local/bin && \
    cp TensorRT-*/include/* /usr/include/x86_64-linux-gnu && \
    python -m pip install TensorRT-*/python/tensorrt-*-cp310-none-linux_x86_64.whl && \
    rm -rf TensorRT-*.Linux.x86_64-gnu.cuda-*.tar.gz TensorRT-* tensorrt.tar.gz

# Set the TensorRT library path. Used while building CUDA plugins.
ENV TRT_LIBPATH=/usr/local/lib/python3.10/dist-packages/tensorrt_libs
RUN ln -s /usr/local/lib/python3.10/dist-packages/tensorrt_libs/libnvinfer.so.* /usr/local/lib/python3.10/dist-packages/tensorrt_libs/libnvinfer.so

ENV LD_LIBRARY_PATH=$TRT_LIBPATH:$LD_LIBRARY_PATH

# Find and install requirements.txt files for all examples
COPY . /workspace/TensorRT-Model-Optimizer
RUN find /workspace/TensorRT-Model-Optimizer -name "requirements.txt" | while read req_file; do \
    echo "Installing from $req_file"; \
    pip install -r "$req_file"; \
    done

# Precompile quantization extensions since this may take several minutes on every docker image restart so it's best to do it once ahead of time
RUN python -c "import modelopt.torch.quantization.extensions as ext; print(ext.cuda_ext); print(ext.cuda_ext_fp8)"

# Allow users to run without root
RUN chmod -R 777 /workspace
