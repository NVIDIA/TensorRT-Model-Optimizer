FROM nvcr.io/nvidia/pytorch:25.04-py3

ARG PIP_CONSTRAINT=

RUN pip install jsonlines omegaconf pulp
RUN pip install datasets transformers
RUN pip install tiktoken blobfile
RUN pip install flask flask_restful fire nltk

WORKDIR /workspace

RUN git clone https://github.com/NVIDIA/Megatron-LM.git
RUN pip install -e Megatron-LM/
RUN chmod -R 777 Megatron-LM/

RUN git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer.git
RUN pip install -e TensorRT-Model-Optimizer
RUN chmod -R 777 TensorRT-Model-Optimizer

WORKDIR /workspace/nmm-sandbox
