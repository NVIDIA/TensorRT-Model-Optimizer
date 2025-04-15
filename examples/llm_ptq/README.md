# Post-training quantization (PTQ)

## What's This Example Folder About?

This folder demonstrates how the Model Optimizer does PTQ quantization on an LLM and deploys the quantized LLM with TensorRT-LLM.
To learn more about the quantization feature, please refer to the [documentation](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/1_quantization.html).
Users can also choose to export the quantized models in a unified format that is deployable on vLLM and SGLang, in addition to TensorRT-LLM. For details please refer to its [documentation](https://nvidia.github.io/TensorRT-Model-Optimizer/deployment/3_unified_hf.html).

This document introduces:

- The scripts to quantize, convert and evaluate LLMs,
- The Python code and APIs to quantize and deploy the models.

If you are interested in quantization-aware training (QAT) for LLMs, please refer to the [QAT README](../llm_qat/README.md).

## Model Quantization and TRT LLM Conversion

### All-in-one Scripts for Quantization and Building

There are many quantization schemes supported in the example scripts:

1. The [FP8 format](https://developer.nvidia.com/blog/nvidia-arm-and-intel-publish-fp8-specification-for-standardization-as-an-interchange-format-for-ai/) is available on the Hopper and Ada GPUs with [CUDA compute capability](https://developer.nvidia.com/cuda-gpus) greater than or equal to 8.9.

1. The [INT8 SmoothQuant](https://arxiv.org/abs/2211.10438), developed by MIT HAN Lab and NVIDIA, is designed to reduce both the GPU memory footprint and inference latency of LLM inference.

1. The [INT4 AWQ](https://arxiv.org/abs/2306.00978) is an INT4 weight only quantization and calibration method. INT4 AWQ is particularly effective for low batch inference where inference latency is dominated by weight loading time rather than the computation time itself. For low batch inference, INT4 AWQ could give lower latency than FP8/INT8 and lower accuracy degradation than INT8.

1. The W4A8 AWQ is an extension of the INT4 AWQ quantization that it also uses FP8 for activation for more speed up and acceleration.

1. The [NVFP4](https://blogs.nvidia.com/blog/generative-ai-studio-ces-geforce-rtx-50-series/) is one of the new FP4 formats supported by NVIDIA Blackwell GPU and demonstrates good accuracy compared with other 4-bit alternatives. NVFP4 can be applied to both model weights as well as activations, providing the potential for both a significant increase in math throughput and reductions in memory footprint and memory bandwidth usage compared to the FP8 data format on Blackwell.

The following scripts provide an all-in-one and step-by-step model quantization example for Llama-3, NeMo Nemotron, and Megatron-LM models. The quantization format and the number of GPUs will be supplied as inputs to these scripts. By default, we build the engine for the fp8 format and 1 GPU.

```bash
cd <this example folder>
```

#### For the Hugging Face models:

For LLM models like [Llama-3](https://huggingface.co/meta-llama):

```bash
# Install model specific pip dependencies if needed

export HF_PATH=<the downloaded LLaMA checkpoint from the Hugging Face hub, or simply the model card>
scripts/huggingface_example.sh --model $HF_PATH --quant [fp8|nvfp4|int8_sq|int4_awq|w4a8_awq] --tp [1|2|4|8]
```

> *By default `trust_remote_code` is set to false. Please turn it on if model calibration and eval requires it using `--trust_remote_code`.*

> *If the Huggingface model calibration fails on a multi-GPU system due to mismatched tensor placement, please try setting CUDA_VISIBLE_DEVICES to a smaller number.*

> *FP8 calibration over a large model with limited GPU memory is not recommended but possible with the [accelerate](https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling) package. Please tune the device_map setting in [`example_utils.py`](./example_utils.py) if needed for model loading and the calibration process can be slow.*

> *Huggingface models trained with `modelopt.torch.speculative` can be used as regular Huggingface models in PTQ. Note: there is a known issue with Huggingface models loaded across multiple GPUs for inference (i.e., "Expected all tensors to be on the same device, but found at least two devices..."). When encountered this error in PTQ of speculative decoding models, try reducing the number of GPUs used.*

> *Calibration by default uses left padding_side for the Huggingface tokenizer as it usually leads to lower accuracy loss. The exported tokenizer files restores the default padding_side.*

#### Llama 4

We support FP8 and NVFP4 quantized Llama 4 model Hugging Face checkpoint export using the following command:

```bash
python hf_ptq.py --pyt_ckpt_path=<llama4 model path> --export_path=<quantized hf checkpoint> --qformat=[fp8|nvfp4] --export_fmt=hf
```

The quantized checkpoint can be deployed following the TensorRT-LLM instructions.

#### For NeMo models like [nemotron](https://huggingface.co/nvidia/nemotron-3-8b-base-4k):

NeMo PTQ requires the NeMo package installed. It's recommended to start from the NeMo containers like `nvcr.io/nvidia/nemo:24.07` or latest `nvcr.io/nvidia/nemo:dev` directly.

```bash
# Inside the NeMo container:
# Download the nemotron model from the Hugging Face.
export GPT_MODEL_FILE=Nemotron-3-8B-Base-4k.nemo

# Reinstall latest modelopt and build the extensions if not already done.
pip install -U "nvidia-modelopt[torch]" --extra-index-url https://pypi.nvidia.com
python -c "import modelopt.torch.quantization.extensions as ext; ext.precompile()"

scripts/nemo_example.sh --type gpt --model $GPT_MODEL_FILE --quant [fp8|nvfp4|int8_sq|int4_awq] --tp [1|2|4|8]
```

> *If the TensorRT-LLM version in the NeMo container is older than the supported version, please continue building TRT-LLM engine with the `docker.io/library/modelopt_examples:latest` container built in the ModelOpt docker build step. Additionally you would also need to `pip install megatron-core "nemo-toolkit[all]" --extra-index-url https://pypi.nvidia.com` to install required NeMo dependencies*

#### For Megatron-LM models:

Megatron-LM framework PTQ and TensorRT-LLM deployment examples are maintained in the Megatron-LM GitHub repo. Please refer to the examples [here](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/export).

> *If GPU out-of-memory error is reported running the scripts, please try editing the scripts and reducing the max batch size of the TensorRT-LLM engine to save GPU memory.*

The example scripts above also have an additional flag `--tasks`, where the actual tasks run in the script can be customized. The allowed tasks are `build,mmlu,benchmark,lm_eval,livecodebench` specified in the script [parser](./scripts/parser.sh). The tasks combo can be specified with a comma-separated task list. Some tasks like mmlu can take a long time to run. To run lm_eval tasks, please also specify the `--lm_eval_tasks` flag with comma separated lm_eval tasks [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks).

Please refer to the `Technical Details` section below about the stage executed inside the script and the outputs per stage.

### Model Support List

Model | fp8 | int8_sq | int4_awq | w4a8_awq<sup>1</sup> | nvfp4<sup>5</sup> |
--- | --- | --- | --- | --- | ---
GPTJ | Yes | Yes | Yes | Yes | -
LLAMA 2 | Yes | Yes | Yes | Yes | -
LLAMA 3, 3.1, 3.3 | Yes | No | Yes | Yes<sup>3</sup> | Yes
LLAMA 4 | Yes | No | No | No | Yes
LLAMA 2 (Nemo) | Yes | Yes | Yes | Yes | -
CodeLlama | Yes | Yes | Yes | No | -
Mistral | Yes | Yes | Yes | No | Yes
Mixtral 8x7B, 8x22B | Yes | No | Yes<sup>2</sup> | No | Yes
Snowflake Arctic<sup>2</sup> | Yes | No | Yes | No | -
Falcon 40B, 180B | Yes | Yes | Yes | Yes | -
Falcon 7B | Yes | Yes | No | No | -
MPT 7B, 30B | Yes | Yes | Yes | Yes | -
Baichuan 1, 2 | Yes | Yes | Yes | Yes | -
ChatGLM2, 3 6B | No | No | Yes | No | -
Bloom | Yes | Yes | Yes | Yes | -
Phi-1,2,3,4 | Yes | Yes | Yes | Yes<sup>3</sup> |
Phi-3.5 MOE | Yes | No | No | No | -
Nemotron 8B | Yes | No | Yes | No | -
Gemma 2B, 7B | Yes | No | Yes | Yes | -
Gemma 2 9B, 27B | Yes<sup>2</sup> | No | Yes | No | -
RecurrentGemma 2B | Yes | Yes | Yes | No | -
StarCoder 2 | Yes | Yes | Yes | No | -
QWen 2, 2.5 <sup>4</sup> | Yes | Yes | Yes | Yes | Yes
QWen MOE | Yes | - | - | - | Yes
QwQ | Yes | - | - | - | Yes
DBRX | Yes | No | No | No | -
InternLM2 | Yes | No | Yes | Yes<sup>3</sup> | -
Exaone | Yes | Yes | Yes | Yes | -
Minitron | Yes | Yes | Yes | Yes<sup>2</sup> | Yes
T5 | Yes | Yes | Yes | Yes | -
Whisper | Yes | No | No | No | -

> *<sup>1.</sup>The w4a8_awq is an experimental quantization scheme that may result in a higher accuracy penalty.*

> *<sup>2.</sup>For some models, there is only support for exporting quantized checkpoints.*

> *<sup>3.</sup>W4A8_AWQ is only available on some models but not all*

> *<sup>4.</sup>For some models, KV cache quantization may result in a higher accuracy penalty.*

> *<sup>5.</sup>A selective set of the popular models are internally tested. The actual model support list may be longer. NVFP4 inference requires Blackwell GPUs and TensorRT-LLM v0.17 or later*

> *The accuracy loss after PTQ may vary depending on the actual model and the quantization method. Different models may have different accuracy loss and usually the accuracy loss is more significant when the base model is small. If the accuracy after PTQ is not meeting the requirement, please try either modifying [hf_ptq.py](./hf_ptq.py) and disabling the KV cache quantization or using the [QAT](./../llm_qat/README.md) instead.*

### Deploy FP8 quantized model using vLLM and SGLang

Besides TensorRT-LLM, the Model Optimizer also supports deploying the FP8 quantized Hugging Face LLM using [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang). Model Optimizer supports exporting a unified checkpoint<sup>1</sup> that is compatible for deployment with vLLM and SGLang. The unified checkpoint format design reflects two key characteristics: 1. The layer structures and tensor names remain aligned with the original Hugging Face checkpoint, and 2. The same checkpoint can be deployed across multiple inference frameworks without modification. A unified checkpoint can be exported using the following command:

```bash
# Quantize and export
python hf_ptq.py --pyt_ckpt_path <huggingface_model_card> --qformat fp8 --export_fmt hf --export_path <quantized_ckpt_path> --trust_remote_code
```

Alternatively, the wrapper script `huggingface_example.sh` also supports quantize and export:

```bash
scripts/huggingface_example.sh --model <huggingface_model_card> --quant fp8 --export_fmt hf
```

Then start the inference instance using vLLM in python, for example:

```python
from vllm import LLM

llm_fp8 = LLM(model="<the exported model path>", quantization="modelopt")
print(llm_fp8.generate(["What's the age of the earth? "]))
```

For SGLang:

```python
import sglang as sgl

llm_fp8 = sgl.Engine(model_path="<the exported model path>", quantization="modelopt")
print(llm_fp8.generate(["What's the age of the earth? "]))
```

> *<sup>1. Unified checkpoint export currently does not support sparsity. Speculative decoding is only supported in unified checkpoint export. The exported unified checkpoint then needs a TensorRT-LLM checkpoint converter (e.g., [this](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/eagle/convert_checkpoint.py)) to convert and build the TensorRT engine(s) for deployment. Alternatively, call TensorRT-LLM LLM-API to deploy the unified checkpoints e.g., check examples [here](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/pytorch#trt-llm-with-pytorch). </sup>*

### Model Support List

Model | FP8
--- | ---
LLAMA 2 | Yes
LLAMA 3, 3.1 | Yes
QWen2 | Yes
Mixtral 8x7B | Yes
CodeLlama | Yes

### Optimal Partial Quantization using AutoQuantize

[AutoQuantize](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html#modelopt.torch.quantization.model_quant.auto_quantize) is a PTQ algorithm from ModelOpt which quantizes a model by searching for the best quantization format per-layer while meeting the performance constraint specified by the user. This way, `AutoQuantize` enables to trade-off model accuracy for performance.

Currently `AutoQuantize` supports only `auto_quantize_bits` as the performance constraint (for both weight-only quantization and
weight & activation quantization). See
[AutoQuantize documentation](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html#modelopt.torch.quantization.model_quant.auto_quantize) for more details.

#### AutoQuantize for Hugging Face models

`AutoQuantize` can be performed for Huggingface LLM models like [Llama-3](https://huggingface.co/meta-llama) as shown below:

```bash
export HF_PATH=<the downloaded LLaMA checkpoint from the Hugging Face hub, or simply the model card>
# --auto_quantize_bits specifies the constraint for `AutoQuantize`
# --quant specifies the formats to be searched for `AutoQuantize`
# NOTE: auto_quantize_bits cannot be lower than the number of bits for the smallest quantization format in --quant
scripts/huggingface_example.sh --type llama --model $HF_PATH --quant w4a8_awq,fp8 --auto_quantize_bits 4.8 --tp [1|2|4|8]  --calib_batch_size 4
```

The above example perform `AutoQuantize` where the less quantization sensitive layers are quantized with `w4a8_awq` (specified by `--quant w4a8_awq`) and the more sensitive layers
are kept un-quantized such that the effective bits is 4.8 (specified by `--auto_quantize_bits 4.8`).

#### AutoQuantize for NeMo models

The usage is similar for NeMo models to perform `AutoQuantize`. Please refer to the earlier section on [NeMo models](#for-nemo-models-like-nemotron) for the full setup instructions.

```bash
# --auto_quantize_bits specifies the constraint for `AutoQuantize`
# --quant specifies the formats to be searched for `AutoQuantize`. Multiple formats can be searched over by passing them as comma separated values
scripts/nemo_example.sh --type gpt --model $GPT_MODEL_FILE --quant fp8,int4_awq --auto_quantize_bits 6.4 --tp [1|2|4|8]
```

## Technical Details

### Quantization

[`hf_ptq.py`](./hf_ptq.py) and [`nemo_ptq.py`](nemo_ptq.py) will use the Model Optimizer to calibrate the PyTorch models, and generate a [TensorRT-LLM checkpoint](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/architecture/checkpoint.md), saved as a json (for the model structure) and safetensors files (for the model weights) that TensorRT-LLM could parse.

Quantization requires running the model in original precision (fp16/bf16). Below is our recommended number of GPUs based on our testing for as an example:

| Minimum number of GPUs | 24GB (4090, A5000, L40) | 48GB (A6000, L40s) | 80GB (A100, H100) |
|------------------------|-------------------------|--------------------|-------------------|
| Llama2 7B | 1 | 1 | 1 |
| Llama2 13B | 2 | 1 | 1 |
| Llama2 70B | 8 | 4 | 2 |
| Falcon 180B | Not supported | Not supported | 8 |

### Post-training Sparsification

Before quantizing the model, [`hf_ptq.py`](./hf_ptq.py) offers users the option to sparsify the weights of their models using a 2:4 pattern. This reduces the memory footprint during inference and can lead to performance improvements. The following is an example command to enable weight sparsity:

```bash
scripts/huggingface_example.sh --model $HF_PATH --quant [fp16|fp8|int8_sq] --tp [1|2|4|8] --sparsity sparsegpt
```

> *The accuracy loss due to post-training sparsification depends on the model and the downstream tasks. To best preserve accuracy, [sparsegpt](https://arxiv.org/abs/2301.00774) is recommended. However, users should be aware that there could still be a noticeable drop in accuracy. Read more about Post-training Sparsification and Sparsity Aware Training (SAT) in the [Sparsity README](../llm_sparsity/README.md).*

### TensorRT-LLM Engine Build

The script [`modelopt_to_tensorrt_llm.py`](modelopt_to_tensorrt_llm.py) constructs the TensorRT-LLM network and builds the TensorRT-LLM engine for deployment using the quantization outputs model config files from the previous step. The generated engine(s) will be saved as .engine file(s), ready for deployment.

### TensorRT-LLM Engine Validation

A list of accuracy validation benchmarks are provided in the [llm_eval](../llm_eval/README.md) directory. Right now MMLU, and MTbench are supported in this example by specifying the `--tasks` flag running the scripts mentioned above. For MTBench, the task only runs the answer generation stage. Please follow [fastchat](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) to get the evaluation judge score.

The [`benchmark_suite.py`](benchmarks/benchmark_suite.py) script is used as a fast performance benchmark. For details, please refer to the [TensorRT-LLM documentation](https://github.com/NVIDIA/TensorRT-LLM/blob/main/benchmarks/)

This example also covers the [lm_evaluation_harness](https://github.com/EleutherAI/lm-evaluation-harness), MMLU and the human eval accuracy benchmarks, whose details can be found [here](../llm_eval/README.md). The supported lm_eval evaluation tasks are listed [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks)

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
        decoder_type,  # The type of the model, e.g gpt, gptj, or llama.
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
