FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get -y install python3.10 python3-pip python-is-python3 openmpi-bin libopenmpi-dev wget git git-lfs unzip

ENV PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com"

# Install the latest setuptools using pip
RUN rm -rf /usr/lib/python3/dist-packages/setuptools*
RUN pip install setuptools -U

# TensorRT LLM
RUN pip install tensorrt-llm~=0.9 -U

# Modelopt
ARG MODELOPT_VERSION=0.11.0
RUN pip install "nvidia-modelopt[all]~=$MODELOPT_VERSION" -U --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple --extra-index-url https://gitlab-master.nvidia.com/api/v4/projects/95421/packages/pypi/simple
RUN python -c "import modelopt"

# TensorRT dev environment installation.
RUN wget -q https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.3.0/tensorrt-9.3.0.1.linux.x86_64-gnu.cuda-12.2.tar.gz && \
    tar -xf tensorrt-9.3.0.1.linux.x86_64-gnu.cuda-12.2.tar.gz && \
    cp TensorRT-9.3.0.1/bin/trtexec /usr/local/bin && \
    cp TensorRT-9.3.0.1/include/* /usr/include/x86_64-linux-gnu && \
    rm -rf tensorrt-9.3.0.1.linux.x86_64-gnu.cuda-12.2.tar.gz TensorRT-9.3.0.1

# TensorRT plugins.
ENV TRT_LIBPATH=/usr/local/lib/python3.10/dist-packages/tensorrt_libs
RUN ln -s /usr/local/lib/python3.10/dist-packages/tensorrt_libs/libnvinfer.so.9 /usr/local/lib/python3.10/dist-packages/tensorrt_libs/libnvinfer.so

COPY plugins examples/plugins
RUN cp examples/plugins/prebuilt/* $TRT_LIBPATH
RUN cd examples/plugins && make -j $(nproc)

ENV LD_LIBRARY_PATH=$TRT_LIBPATH:$LD_LIBRARY_PATH

# Clone the TesnorRT repo
RUN git clone --depth 1 --branch v9.3.0 https://github.com/NVIDIA/TensorRT.git

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
RUN pip install -r examples/diffusers/requirements.txt

# Allow users to run without root
RUN chmod -R 777 /workspace
