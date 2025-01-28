# Deploy quantized models from Nvidia's Hugging Face model hub with TensorRT-LLM and vLLM

The provided bash scripts show an example to deploy and run the [quantized LLama 3.1 8B Instruct FP8 model](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8) from Nvidia's Hugging Face model hub on TensorRT-LLM and vLLM respectively.

Before running the bash scripts, please make sure you have setup the environment properly:

- Download the modelopt quantized checkpoints. You can either download all checkpoints with [this script](download_hf_ckpt.py), or use `huggingface-cli download <HF repo> --local-dir  <local_dir>` to download a specific one.
- Git clone the [TensorRT-LLM repo](https://github.com/NVIDIA/TensorRT-LLM) and install it properly by following instructions [here](https://nvidia.github.io/TensorRT-LLM/installation/linux.html).
- Install vLLM properly by following instructions [here](https://docs.vllm.ai/en/latest/getting_started/installation.html#install-released-versions).

Then, to deploy and run on TensorRT-LLM:

```sh
bash llama_fp8_deploy_trtllm.sh <YOUR_HF_CKPT_DIR> <YOUR_TensorRT_LLM_DIR>
```

To deploy and run on vLLM:

```sh
bash llama_fp8_deploy_vllm.sh <YOUR_HF_CKPT_DIR>
```

If you want to run post-training quantization with TensorRT Model Optimizer for your selected models, check [here](../llm_ptq/README.md).
