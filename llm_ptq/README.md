# Post-training quantization (PTQ)

## What's This Example Folder About?

This folder demonstrates how the Model Optimizer does PTQ quantization on an LLM and deploys the quantized LLM with TensorRT-LLM.

This document introduces:

- The scripts to quantize, convert and evaluate LLMs,
- The Python code and APIs to quantize and deploy the models.

If you are interseted in quantization-aware training (QAT) for LLMs, please refer to the [QAT README](../llm_qat/README.md).

## Model Quantization and TRT LLM Conversion

### All-in-one Scripts for Quantization and Building

There are many quantization schemes supported in the example scripts:

1. The [FP8 format](https://developer.nvidia.com/blog/nvidia-arm-and-intel-publish-fp8-specification-for-standardization-as-an-interchange-format-for-ai/) is available on the Hopper and Ada GPUs with [CUDA compute capability](https://developer.nvidia.com/cuda-gpus) greater than or equal to 8.9.

1. The [INT8 SmoothQuant](https://arxiv.org/abs/2211.10438), developed by MIT HAN Lab and NVIDIA, is designed to reduce both the GPU memory footprint and inference latency of LLM inference.

1. The [INT4 AWQ](https://arxiv.org/abs/2306.00978) is an INT4 weight only quantization and calibration method. INT4 AWQ is particularly effective for low batch inference where inference latency is dominated by weight loading time rather than the computation time itself. For low batch inference, INT4 AWQ could give lower latency than FP8/INT8 and lower accuracy degradation than INT8.

1. The W4A8 AWQ is an extension of the INT4 AWQ quantization that it also uses FP8 for activation for more speed up and acceleration.

The following scripts provide an all-in-one and step-by-step model quantization example for Llama-3, LLAVA, NeMo Nemotron, Megatron-LM, and Medusa models. The quantization format and the number of GPUs will be supplied as inputs to these scripts. By default, we build the engine for the fp8 format and 1 GPU.

```bash
cd <this example folder>
```

**For the Hugging Face models:**

For LLM models like [Llama-3](https://huggingface.co/meta-llama):

```bash
# Before start, install model specific pip dependencies if needed

export HF_PATH=<the downloaded LLaMA checkpoint from the Hugging Face hub, or simply the model card>
scripts/huggingface_example.sh --type llama --model $HF_PATH --quant [fp8|int8_sq|int4_awq|w4a8_awq] --tp [1|2|4|8]
```

This example also supports the vision language model (VLM) like [LLAVA](https://huggingface.co/llava-hf/llava-1.5-7b-hf):

```bash
# Make sure git-lfs is installed before running the example.
git clone https://huggingface.co/llava-hf/llava-1.5-7b-hf
scripts/huggingface_example.sh --type llava --model llava-1.5-7b-hf --quant [fp8|int8_sq|int4_awq] --tp [1|2|4|8]
```

The above command generates the quantized checkpoint for TensorRT-LLM. After this, please follow up with the TensorRT-LLM [multimodal example, step 3](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal#llava-and-vila) for engine building and evaluation.

> *If the Huggingface model calibration fails on a multi-GPU system due to mismatched tensor placement, please try setting CUDA_VISIBLE_DEVICES to a smaller number.*

**For NeMo models like [nemotron](https://huggingface.co/nvidia/nemotron-3-8b-base-4k):**

NeMo PTQ requires the NeMo package installed. It's recommended to start from the NeMo container(`nvcr.io/nvidia/nemo:24.05.framework`) directly.

```bash
# Inside the NeMo container:
# Download the nemotron model from the Hugging Face.
export GPT_MODEL_FILE=Nemotron-3-8B-Base-4k.nemo

# Install modelopt
pip install "nvidia-modelopt[torch]" -U

scripts/nemo_example.sh --type gptnext --model $GPT_MODEL_FILE --quant [fp8|int8_sq|int4_awq] --tp [1|2|4|8]
```

**For Megatron-LM models:**

Megatron-LM framework PTQ and TensorRT-LLM deployment examples are maintained in the Megatron-LM GitHub repo. Please refer to the examples [here](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/inference).

> *If GPU out-of-memory error is reported running the scripts, please try editing the scripts and reducing the max batch size of the TensorRT-LLM engine to save GPU memory.*

The example scripts above also have an additional flag `--tasks`, where the actual tasks run in the script can be customized. The allowed tasks are `build,summarize,mmlu,humaneval,benchmark` specified in the script [parser](./scripts/parser.sh). The tasks combo can be specified with a comma-separated task list. Some tasks like mmlu, humaneval can take a long time to run.

Please refer to the `Technical Details` section below about the stage executed inside the script and the outputs per stage.

### Model Support List

Model | type | fp8 | int8_sq | int4_awq | w4a8_awq<sup>1</sup>
--- | --- | --- | --- | --- | ---
GPT2 | gpt2 | Yes | Yes | No | No
GPTJ | llama | Yes | Yes | Yes | Yes
LLAMA 2 | llama | Yes | Yes | Yes | Yes
LLAMA 3 | llama | Yes | No | Yes | No
LLAMA 2 (Nemo) | llama | Yes | Yes | Yes | Yes
CodeLlama | llama | Yes | Yes | Yes | No
Mistral | llama | Yes | Yes | Yes | No
Mixtral 8x7B | llama | Yes<sup>3</sup> | No | Yes<sup>2</sup> | No
Snowflake Arctic<sup>2</sup> | llama | Yes | No | Yes | No
Falcon 40B, 180B | falcon | Yes | Yes | Yes | Yes
Falcon 7B | falcon | Yes | Yes | No | No
MPT 7B, 30B | mpt | Yes | Yes | Yes | No
Baichuan 1, 2 | baichuan | Yes | Yes | Yes | Yes
ChatGLM2, 3 6B | chatglm | No | No | Yes | No
Bloom | bloom | Yes | Yes | Yes | Yes
Phi-1,2,3<sup>2</sup> | phi | Yes | Yes | Yes | Yes
Nemotron 8B | gptnext | Yes | No | Yes | No
Gemma 2B, 7B | gemma | Yes | No | Yes | No
RecurrentGemma 2B | recurrentgemma | Yes | Yes | Yes | No
StarCoder 2 | gptnext | Yes | Yes | Yes | No
QWen-1,1.5 | qwen | Yes | Yes | Yes | Yes

> *<sup>1.</sup>The w4a8_awq is an experimental quantization scheme that may result in a higher accuracy penalty. Only available on sm90 GPUs*

> *<sup>2.</sup>For some models (e.g. phi3 small, medium), there is only support for exporting quantized checkpoints.*

> *<sup>3.</sup>Mixtral FP8 only available on sm90 GPUs*

> *The accuracy loss after PTQ may vary depending on the actual model and the quantization method. Different models may have different accuracy loss and usually the accuracy loss is more significant when the base model is small. If the accuracy after PTQ is not meeting the requirement, please try either modifying [hf_ptq.py](./hf_ptq.py) and disabling the KV cache quantization or using the [QAT](./../llm_qat/README.md) instead.*

### Deploy FP8 quantized model using vLLM

> *This feature is experimental.*

Besides TensorRT-LLM, the Model Optimizer also supports deploying the FP8 quantized Hugging Face LLM using [vLLM](https://github.com/vllm-project/vllm).

```bash
# First install vllm dependency
pip install vllm>=0.4.3

# Quantize and export
scripts/huggingface_example.sh --type <type> --model <huggingface model card> --quant fp8 --deployment vllm
```

Then start the inference instance using vLLM in python, for example:

```python
from vllm import LLM

llm_fp8 = LLM(model="<the exported model path>", quantization="fp8")
print(llm.generate(["What's the age of the earth? "]))
```

### Medusa model Quantization and Building

[Medusa](https://github.com/FasterDecoding/Medusa) is a simple framework that democratizes the acceleration techniques for LLM generation with multiple decoding heads. It adds extra "heads" to LLMs to predict multiple future tokens simultaneously. During generation, these heads each produce multiple likely words for the corresponding position. These options are then combined and processed using a tree-based attention mechanism. Finally, a typical acceptance scheme is employed to pick the longest plausible prefix from the candidates for further decoding. The base model of medusa can be quantized using ModelOpt to further speed up the inference.

```bash
# Install Medusa
git clone https://github.com/FasterDecoding/Medusa.git
pushd Medusa
# Commit e2a5d20 has been tested so we recommend use this commit
git checkout e2a5d20
# Replace is_flash_attn_available with is_flash_attn_2_available as it no longer exists in transformers
sed -i 's/is_flash_attn_available/is_flash_attn_2_available/g' medusa/model/*.py
# Install from source
pip install -e .
popd

# Make sure git-lfs is installed before running the example.
git clone https://huggingface.co/lmsys/vicuna-7b-v1.3
git clone https://huggingface.co/FasterDecoding/medusa-vicuna-7b-v1.3
# The medusa_num_heads in medusa-vicuna-7b-v1.3/config.json is incorrect
# Make sure to manualy change it from 2 to 5 as there are 5 medusa heads in the
# Medusa state dict
jq -c '.medusa_num_heads=5' medusa-vicuna-7b-v1.3/config.json >> medusa-vicuna-7b-v1.3/tmp.json && mv medusa-vicuna-7b-v1.3/tmp.json medusa-vicuna-7b-v1.3/config.json

# First combine the base model and medusa head state dict into one model state dict
# You can skip this step if the medusa model is saved in the same state dict
python prepare_medusa.py --base_model vicuna-7b-v1.3 --medusa_model medusa-vicuna-7b-v1.3

# Perform PTQ on the medusa model
scripts/medusa_example.sh --type llama --model vicuna-7b-v1.3 --quant fp8 --tp [1|2|4|8]
```

## Technical Details

### Quantization

[`hf_ptq.py`](./hf_ptq.py) and [`nemo_ptq.py`](nemo_ptq.py) will use the Model Optimizer to calibrate the PyTorch models, and generate a [TensorRT-LLM checkpoint](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/architecture/checkpoint.md), saved as a json (for the model structure) and safetensors files (for the model weights) that TensorRT-LLM could parse.

Quantization requires running the model in original precision (fp16/bf16). Below is our recommended number of GPUs based on our testing for as an example:

| Minimum number of GPUs | 24GB (4090, A5000, L40) | 48GB (A6000, L40s) | 80GB (A100, H100) |
|------------------------|-------------------------|--------------------|-------------------|
| Llama2 7B              |                       1 |                  1 |                 1 |
| Llama2 13B             |                       2 |                  1 |                 1 |
| Llama2 70B             |                       8 |                  4 |                 2 |
| Falcon 180B            | Not supported           | Not supported      |                 8 |

### Post-training Sparsification

Before quantizing the model, [`hf_ptq.py`](./hf_ptq.py) offers users the option to sparsify the weights of their models using a 2:4 pattern. This reduces the memory footprint during inference and can lead to performance improvements. The following is an example command to enable weight sparsity:

```bash
scripts/huggingface_example.sh --type llama --model $HF_PATH --quant [fp16|fp8|int8_sq] --tp [1|2|4|8] --sparsity sparsegpt
```

> *The accuracy loss due to post-training sparsification depends on the model and the downstream tasks. To best preserve accuracy, [sparsegpt](https://arxiv.org/abs/2301.00774) is recommended. However, users should be aware that there could still be a noticeable drop in accuracy. Read more about Post-training Sparsification and Sparsity Aware Training (SAT) in the [Sparsity README](../llm_sparsity/README.md).*

### TensorRT-LLM Engine Build

The script [`modelopt_to_tensorrt_llm.py`](modelopt_to_tensorrt_llm.py) constructs the TensorRT-LLM network and builds the TensorRT-LLM engine for deployment using the quantization outputs model config files from the previous step. The generated engine(s) will be saved as .engine file(s), ready for deployment.

### TensorRT-LLM Engine Validation

The [`summarize.py`](summarize/summarize.py) script can be used to test the accuracy and latency on [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset. For each summary, the script can compute the [ROUGE](<https://en.wikipedia.org/wiki/ROUGE_(metric)>) scores. The `ROUGE-1` score is used for implementation validation.

When the script finishes, it will report the latency and the ROUGE score.

The TensorRT-LLM engine and Hugging Face model evaluations are reported in separate stages.

> *By default, the evaluation only runs on 20 data samples. For accurate accuracy evaluation, it is recommended to use higher data sample counts, e.g. 2000 and above. Please modify the script and increase max_ite accordingly to customize the evaluation sample count.*

The [`benchmark.py`](benchmarks/benchmark.py) script is used as a fast performance benchmark with faked inputs that match the max batch size and input length of the built engines. This benchmark runs with the TensorRT-LLM python runtime and reports the metrics like tokens per second and the peak memory.

> *This benchmark runs static batching and only supports multi-TP inference. For more advanced performance benchmarking, please check with the [TensorRT-LLM repo](https://github.com/NVIDIA/TensorRT-LLM/tree/main/benchmarks) directly.*

This example also covers the MMLU and the human eval accuracy benchmarks, whose details can be found [here](../llm_eval/README.md).

## APIs

### PTQ (Post Training Quantization)

PTQ can be achieved with simple calibration on a small set of training or evaluation data (typically 128-512 samples) after converting a regular PyTorch model to a quantized model. The accuracy of PTQ is typically robust across different choices of calibration data, so we use [`cnn_dailymail`](https://huggingface.co/datasets/cnn_dailymail) by default. Users can try other datasets by easily modifying the `get_calib_dataloader` in [example_utils.py](./example_utils.py).

```python
import modelopt.torch.quantization as mtq

model = AutoModelForCausalLM.from_pretrained("...")

# Select the quantization config, for example, INT8 Smooth Quant
config = mtq.INT8_SMOOTHQUANT_CFG


# Prepare the calibration set and define a forward loop
def forward_loop(model):
    for data in calib_set:
        model(data)


# PTQ with in-place replacement to quantized modules
model = mtq.quantize(model, config, forward_loop)
```

### Export Quantized Model

After the model is quantized, the TensorRT-LLM checkpoint can be stored. The user can specify the inference time TP and PP size and the export API will organize the weights to fit the target GPUs.

The export API is

```python
from modelopt.torch.export import export_tensorrt_llm_checkpoint

with torch.inference_mode():
    export_tensorrt_llm_checkpoint(
        model,  # The quantized model.
        decoder_type,  # The type of the model, e.g gptj, llama or gptnext.
        dtype,  # The exported weights data type.
        export_dir,  # The directory where the exported files will be stored.
        inference_tensor_parallel,  # The number of GPUs used in the inference time tensor parallel.
        inference_pipeline_parallel,  # The number of GPUs used in the inference time pipeline parallel.
        use_nfs_workspace,  # If exporting in a multi-node setup, please specify a shared directory like NFS for cross-node communication.
    )
```

### Build the TensorRT-LLM engines

After the TensorRT-LLM checkpoint export, you can use the `trtllm-build` build command to build the engines from the exported checkpoints. Please check the [ TensorRT-LLM Build API](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/architecture/workflow.md#build-apis) documentation for reference.

In this example, we use [modelopt_to_tensorrt_llm.py](./modelopt_to_tensorrt_llm.py) script as the easy-to-use wrapper over the TensorRT-LLM build API to generate the engines.
