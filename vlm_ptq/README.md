# Post-training quantization (PTQ)

## What's This Example Folder About?

This example demonstrates how the Model Optimizer does PTQ quantization on the LLM part of a VLM (visual language model) and deploys the VLM with TensorRT and TensorRT-LLM..

## Model Quantization and TRT LLM Conversion

### All-in-one Scripts for Quantization and Building

Please refer to the  [llm_ptq/README.md](../llm_ptq/README.md) about the details of model quantization.

The following scripts provide an all-in-one and step-by-step model quantization example for Llava, VILA and Phi-3-vision models. The quantization format and the number of GPUs will be supplied as inputs to these scripts. By default, we build the engine for the fp8 format and 1 GPU.

```bash
cd <this example folder>
```

**For the Hugging Face models:**

For [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf):

```bash
git clone https://huggingface.co/llava-hf/llava-1.5-7b-hf
scripts/huggingface_example.sh --type llava --model llava-1.5-7b-hf --quant [fp8|int8_sq|int4_awq|w4a8_awq] --tp [1|2|4|8]
```

For VILA models like [VILA1.5-3b](https://huggingface.co/Efficient-Large-Model/VILA1.5-3b):

```bash
git clone https://huggingface.co/Efficient-Large-Model/VILA1.5-3b vila1.5-3b
scripts/huggingface_example.sh --type vila --model vila1.5-3b --quant [fp8|int8_sq|int4_awq|w4a8_awq] --tp [1|2|4|8]
```

For [Phi-3-vision](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct):

```bash
git clone https://huggingface.co/microsoft/Phi-3-vision-128k-instruct
scripts/huggingface_example.sh --type phi --model Phi-3-vision-128k-instruct --quant [fp8|int8_sq|int4_awq|w4a8_awq]
```

### Model Support List

Model | type | fp8 | int8_sq | int4_awq | w4a8_awq<sup>1</sup>
--- | --- | --- | --- | --- | ---
Llava | llava | Yes | Yes | Yes | Yes
VILA | vila | Yes | Yes | Yes | Yes
Phi-3-vision | phi | Yes | Yes | Yes | Yes

> *<sup>1.</sup>The w4a8_awq is an experimental quantization scheme that may result in a higher accuracy penalty. Only available on sm90 GPUs*

> *The accuracy loss after PTQ may vary depending on the actual model and the quantization method. Different models may have different accuracy loss and usually the accuracy loss is more significant when the base model is small. If the accuracy after PTQ is not meeting the requirement, please try either modifying [hf_ptq.py](./hf_ptq.py) and disabling the KV cache quantization or using the [QAT](./../llm_qat/README.md) instead.*
