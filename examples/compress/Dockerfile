# Docker file for compress example

FROM nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc5

# TODO: The MIP solver would not work with this torch version. 
# Fix it, otherwise, mamba models will not be supported by the Compress algorithm.
# # Required for mamba_ssm to work (the default torch version in the 1.1.0rc5 does not work)
# RUN pip uninstall -y torch
# RUN pip uninstall -y torchvision
# RUN pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# # Mamba SSM
# RUN pip install causal-conv1d --no-build-isolation
# RUN pip install mamba_ssm --no-build-isolation

# Required for puzzletron calc_subblock_stats
RUN pip install hydra-core==1.3.2
RUN pip install wandb~=0.17.5
RUN pip install "frozendict>=2.4.4"
RUN pip install fire
RUN pip install mip
RUN pip install lru-dict

WORKDIR /workspace/

COPY . .
