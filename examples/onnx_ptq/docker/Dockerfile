FROM nvcr.io/nvidia/tensorrt:25.06-py3

ARG CMAKE_VERSION=3.28.0

ENV PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com" \
    PIP_NO_CACHE_DIR=off

RUN python -m pip install --upgrade pip \
    && pip install cmake==${CMAKE_VERSION} \
    && mkdir -p -m 0600 ~/.ssh \
    && ssh-keyscan github.com >> ~/.ssh/known_hosts

WORKDIR /workspace

RUN pip install tensorrt==10.12.0.36 && \
    export TRT_PATH=$(python -c "import tensorrt; import os; print(os.path.dirname(tensorrt.__file__))") && \
    export LD_LIBRARY_PATH="$TRT_PATH/lib:${LD_LIBRARY_PATH}" && \
    export PATH="$TRT_PATH/bin:${PATH}"

# Update PATH variables for local TensorRT installation
ENV LD_LIBRARY_PATH="/workspace/TensorRT/lib:${LD_LIBRARY_PATH}" \
    PATH="/workspace/TensorRT/bin:${PATH}"

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
