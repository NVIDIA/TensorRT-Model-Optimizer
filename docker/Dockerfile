FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get -y install python3.10 python3-pip python-is-python3 openmpi-bin libopenmpi-dev wget git git-lfs unzip

ARG PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com"
ENV PIP_EXTRA_INDEX_URL=$PIP_EXTRA_INDEX_URL

# Install the latest setuptools using pip
RUN rm -rf /usr/lib/python3/dist-packages/setuptools*
RUN pip install setuptools -U

# TensorRT LLM
RUN pip install tensorrt-llm~=0.10 -U

# Modelopt
ARG MODELOPT_VERSION=0.13.0
RUN pip install "nvidia-modelopt[all]~=$MODELOPT_VERSION" -U
RUN python -c "import modelopt"

# Medusa
# Replace is_flash_attn_available with is_flash_attn_2_available as it no longer exists in transformers
RUN git clone https://github.com/FasterDecoding/Medusa.git && \
    cd Medusa && \
    git checkout e2a5d20 && \
    sed -i 's/is_flash_attn_available/is_flash_attn_2_available/g' medusa/model/*.py && \
    pip install -e .

# TensorRT dev environment installation.
ARG TENSORRT_URL=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz
RUN wget -q -O tensorrt.tar.gz $TENSORRT_URL && \
    tar -xf tensorrt.tar.gz && \
    cp TensorRT-*/bin/trtexec /usr/local/bin && \
    cp TensorRT-*/include/* /usr/include/x86_64-linux-gnu && \
    rm -rf TensorRT-*.Linux.x86_64-gnu.cuda-*.tar.gz TensorRT-*

# TensorRT plugins.
ENV TRT_LIBPATH=/usr/local/lib/python3.10/dist-packages/tensorrt_libs
RUN ln -s /usr/local/lib/python3.10/dist-packages/tensorrt_libs/libnvinfer.so.* /usr/local/lib/python3.10/dist-packages/tensorrt_libs/libnvinfer.so

COPY plugins examples/plugins
RUN cp examples/plugins/prebuilt/* $TRT_LIBPATH
RUN cd examples/plugins && make -j $(nproc)

ENV LD_LIBRARY_PATH=$TRT_LIBPATH:$LD_LIBRARY_PATH

# Clone the TesnorRT repo
RUN git clone --depth 1 --branch v10.0.1 https://github.com/NVIDIA/TensorRT.git

# Install the example requirements
COPY llm_ptq examples/llm_ptq
RUN pip install -r examples/llm_ptq/requirements.txt

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

# Allow users to run without root
RUN chmod -R 777 /workspace
