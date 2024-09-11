FROM nvidia/cuda:12.5.1-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get -y install python3.10 python3-pip python-is-python3 openmpi-bin libopenmpi-dev wget git git-lfs unzip jq

ARG PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"
ENV PIP_EXTRA_INDEX_URL=$PIP_EXTRA_INDEX_URL

# Install the latest setuptools using pip
RUN rm -rf /usr/lib/python3/dist-packages/setuptools*
RUN pip install setuptools -U

# TensorRT LLM
RUN pip install tensorrt-llm~=0.12 -U
RUN git clone --depth 1 --branch v0.11.0 https://github.com/NVIDIA/TensorRT-LLM.git && \
    mkdir -p tensorrt-llm && \
    cd TensorRT-LLM && \
    mv benchmarks/cpp tensorrt-llm && \
    rm -rf TensorRT-LLM

# Modelopt
ARG MODELOPT_VERSION=0.15.0
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
    rm -rf TensorRT-*.Linux.x86_64-gnu.cuda-*.tar.gz TensorRT-*

# Set the TensorRT library path. Used while building CUDA plugins.
ENV TRT_LIBPATH=/usr/local/lib/python3.10/dist-packages/tensorrt_libs
RUN ln -s /usr/local/lib/python3.10/dist-packages/tensorrt_libs/libnvinfer.so.* /usr/local/lib/python3.10/dist-packages/tensorrt_libs/libnvinfer.so

ENV LD_LIBRARY_PATH=$TRT_LIBPATH:$LD_LIBRARY_PATH

# Install the example requirements
COPY llm_ptq examples/llm_ptq
RUN pip install -r examples/llm_ptq/requirements.txt

COPY vlm_ptq examples/vlm_ptq

COPY llm_eval examples/llm_eval
RUN pip install -r examples/llm_eval/requirements.txt

COPY llm_qat examples/llm_qat
RUN pip install -r examples/llm_qat/requirements.txt

COPY llm_sparsity examples/llm_sparsity
RUN pip install -r examples/llm_sparsity/requirements.txt

COPY diffusers examples/diffusers
RUN pip install -r examples/diffusers/quantization/requirements.txt

COPY onnx_ptq examples/onnx_ptq
RUN pip install -r examples/onnx_ptq/requirements.txt

COPY speculative_decoding examples/speculative_decoding
RUN pip install -r examples/speculative_decoding/requirements.txt

# Allow users to run without root
RUN chmod -R 777 /workspace
