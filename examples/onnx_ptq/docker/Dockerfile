FROM nvcr.io/nvidia/tensorrt:25.08-py3

ARG CMAKE_VERSION=3.28.0

ENV PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com" \
    PIP_NO_CACHE_DIR=off

RUN python -m pip install --upgrade pip \
    && pip install cmake==${CMAKE_VERSION} \
    && mkdir -p -m 0600 ~/.ssh \
    && ssh-keyscan github.com >> ~/.ssh/known_hosts

WORKDIR /workspace

RUN pip install tensorrt==10.13.2.6
ENV TRT_PATH=/usr/local/lib/python3.12/dist-packages/tensorrt
ENV CUDNN_LIB_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib
ENV LD_LIBRARY_PATH="${CUDNN_LIB_DIR}:${TRT_PATH}/lib:/usr/include:${LD_LIBRARY_PATH}"
ENV PATH="${TRT_PATH}/bin:${PATH}"

# Copy application code and install requirements
COPY modelopt modelopt/modelopt
COPY examples/onnx_ptq modelopt/examples/onnx_ptq
COPY setup.py modelopt/setup.py
COPY pyproject.toml modelopt/pyproject.toml

# Install onnx_ptq requirements
RUN pip install -r modelopt/examples/onnx_ptq/requirements.txt

# Install modelopt
RUN pip install -e "./modelopt[hf,onnx]"

# Allow users to run without root
RUN chmod -R 777 /workspace
