FROM nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc2.post2

ARG PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com"
ENV PIP_EXTRA_INDEX_URL=$PIP_EXTRA_INDEX_URL \
    PIP_NO_CACHE_DIR=off \
    PIP_CONSTRAINT= \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.7 8.9 9.0 10.0+PTX"

RUN apt-get update && \
    apt-get install -y libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN ln -s /app/tensorrt_llm /workspace/tensorrt_llm

# Update PATH and LD_LIBRARY_PATH variables for the TensorRT binaries
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/tensorrt/targets/x86_64-linux-gnu/lib:${LD_LIBRARY_PATH}" \
    PATH="/usr/local/tensorrt/targets/x86_64-linux-gnu/bin:${PATH}"

# Install modelopt from source with all optional dependencies and pre-compile CUDA extensions otherwise they take several minutes on every docker run
# Pre-install llm_ptq requirements.txt
COPY . TensorRT-Model-Optimizer
RUN pip install -e "./TensorRT-Model-Optimizer[all,dev-test]"
RUN rm -rf TensorRT-Model-Optimizer/.git
RUN python -c "import modelopt.torch.quantization.extensions as ext; ext.precompile()"
RUN pip install -r TensorRT-Model-Optimizer/examples/llm_ptq/requirements.txt

# Allow users to run without root
RUN chmod -R 777 /workspace
