FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && \
    apt-get -y install python3.10 python3-pip python-is-python3 openmpi-bin libopenmpi-dev libgl1 libglib2.0-0 wget git git-lfs unzip jq cmake vim && \
    rm -rf /var/lib/apt/lists/*

ARG PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com"
ENV PIP_EXTRA_INDEX_URL=$PIP_EXTRA_INDEX_URL
ENV PIP_NO_CACHE_DIR=off

# Install the latest setuptools using pip
RUN rm -rf /usr/lib/python3/dist-packages/setuptools* && \
    pip install --upgrade pip setuptools

# Install TensorRT-LLM
ARG TRT_LLM_VERSION=0.17.0
RUN pip install "tensorrt-llm~=$TRT_LLM_VERSION" -U
RUN git clone --depth 1 --branch "v$TRT_LLM_VERSION" https://github.com/NVIDIA/TensorRT-LLM.git && \
    mkdir tensorrt-llm && \
    mv TensorRT-LLM/benchmarks/ tensorrt-llm && \
    rm -rf TensorRT-LLM
RUN cd /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs && ln -s libnvinfer_plugin_tensorrt_llm.so libnvinfer_plugin_tensorrt_llm.so.10
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs:$LD_LIBRARY_PATH

# Export the path to 'libcudnn.so.X' needed by 'libonnxruntime_providers_tensorrt.so'
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Install TensorRT dev environment
ARG TENSORRT_URL=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.8.0/tars/TensorRT-10.8.0.43.Linux.x86_64-gnu.cuda-12.8.tar.gz
RUN wget -q -O tensorrt.tar.gz $TENSORRT_URL && \
    tar -xf tensorrt.tar.gz && \
    cp TensorRT-*/bin/trtexec /usr/local/bin && \
    cp TensorRT-*/include/* /usr/include/x86_64-linux-gnu && \
    python -m pip install TensorRT-*/python/tensorrt-*-cp310-none-linux_x86_64.whl && \
    cp -a TensorRT-*/targets/x86_64-linux-gnu/lib/* /usr/local/lib/python3.10/dist-packages/tensorrt_libs && \
    rm -rf TensorRT-*.Linux.x86_64-gnu.cuda-*.tar.gz TensorRT-* tensorrt.tar.gz
ENV TRT_LIB_PATH=/usr/local/lib/python3.10/dist-packages/tensorrt_libs
ENV LD_LIBRARY_PATH=$TRT_LIB_PATH:$LD_LIBRARY_PATH

# Install modelopt with all optional dependencies and pre-compile CUDA extensions otherwise they take several minutes on every docker run
RUN pip install "nvidia-modelopt[all]" -U
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.7 8.9 9.0+PTX"
RUN python -c "import modelopt.torch.quantization.extensions as ext; ext.precompile()"

# Find and install requirements.txt files for all examples excluding windows
COPY . TensorRT-Model-Optimizer
RUN rm -rf TensorRT-Model-Optimizer/.git
RUN find TensorRT-Model-Optimizer/examples -name "requirements.txt" | grep -v "windows" | while read req_file; do \
        echo "Installing from $req_file"; \
        pip install -r "$req_file" || exit 1; \
    done

# Allow users to run without root
RUN chmod -R 777 /workspace
