# Stable Diffusion XL (Base/Turbo) and Stable Diffusion 1.5 Quantization with Model Optimizer

This example shows how to use Model Optimizer to calibrate and quantize the UNet part of the SDXL and SD1.5. The UNet part typically consumes >95% of the e2e Stable Diffusion latency.

We also provide instructions on deploying and running E2E SDXL and SD1.5 pipelines with Model Optimizer quantized int8 and fp8 UNet to generate images and measure latency on target GPUs. Note, Jetson devices are not supported so far.

## Get Started

You may choose to run this example with docker or by installing the required software by yourself.

### Docker

Follow the instructions in [`../../README.md`](../../README.md#docker) to build the docker image and run the docker container.

### By yourself

We assume you already installed NVIDIA TensorRT Model Optimizer `modelopt`, now install required dependencies for Stable Diffusion:

```sh
pip install -r requirements.txt
```

## 8-bit TRT Engine Build Quick Start

You can run the following script to get the int8 or fp8 UNet engines built with default settings for SDXL, and then directly go to **Run End-to-end Stable Diffusion Pipeline with Model Optimizer Quantized Model and demoDiffusion** section to run E2E pipeline to generate images.

```sh
bash build_sdxl_8bit_engine.sh --format {FORMAT} # FORMAT can be int8 or fp8
```

If you prefer to customize parameters in calibration or run other models, please follow the instructions below.

## Calibration with Model Optimizer

We support calibration for both int8 and fp8 precision and for both weights and activations.

Note: Model calibration requires relatively more GPU computing powers and it does not need to be on the same GPUs as
the deployment target GPUs.

### SDXL|SD1.5|SDXL-Turbo INT8

```sh
python quantize.py \
  --model {stabilityai/stable-diffusion-xl-base-1.0|runwayml/stable-diffusion-v1-5|stabilityai/sdxl-turbo} \
  --format int8 --batch-size 2 \
  --calib-size {CALIB_SIZE} --collect-method min-mean \
  --percentile {PERCENTILE} --alpha {ALPHA} \
  --quant-level {QUANT_LEVEL} --n_steps {N_STEPS} \
  --exp_name {EXP_NAME}
```

### SDXL|SD1.5|SDXL-Turbo FP8

```sh
python quantize.py \
  --model {stabilityai/stable-diffusion-xl-base-1.0|runwayml/stable-diffusion-v1-5|stabilityai/sdxl-turbo} \
  --format fp8 --batch-size 2 --calib-size {CALIB_SIZE} --quant-level {QUANT_LEVEL} \
  --n_steps {N_STEPS} --exp_name {EXP_NAME} --collect-method default
```

#### Important Parameters

- `percentile`: Control quantization scaling factors (amax) collecting range, meaning that we will collect the chosen amax in the range of `(n_steps * percentile)` steps. Recommendation: 1.0

- `alpha`: A parameter in SmoothQuant, used for linear layers only. Recommendation: 0.8 for SDXL, 1.0 for SD 1.5

- `quant-level`: Which layers to be quantized, 1: `CNNs`, 2: `CNN + FFN`, 2.5: `CNN + FFN + QKV`, 3: `CNN + Linear (Including FFN, QKV, Proj and others)`. Recommendation: 2, 2.5 and 3, depending on the requirements for image quality & speedup.

- `calib-size`: For SDXL INT8, we recommend 32 or 64, for SDXL FP8, 128 is recommended. For SD 1.5, set it to 512 or 1024.

- `n_steps`: Recommendation: SD/SDXL 20 or 30, SDXL-Turbo 4.

**Then, we can load the generated checkpoint and export the INT8/FP8 quantized model in the next step.**

## ONNX Export

```sh
python run_export.py --model {stabilityai/stable-diffusion-xl-base-1.0|runwayml/stable-diffusion-v1-5|stabilityai/sdxl-turbo} --quantized-ckpt {YOUR_QUANTIZED_CKPT} --format {FORMAT} --quant-level {1.0|2.0|2.5|3.0} --onnx-dir {ONNX_PATH}
```

For ONNX export, we recommend using a device with CPU memory not less than 48GB.

#### Optional for the exported ONNX model

*If the quant-level is 2.5 or above, you can use the following script*

Run the QDQ fusion script

```sh
python onnx_utils/sdxl_graphsurgeon.py --onnx-path ./onnx/unet.onnx --output-onnx onnx_int8_fused/unet.onnx
```

**Next, we can build the TRT engine for INT8 on your desired platforms.**
**For FP8, we only support the TRT deployment on Ada and Hopper GPUs. Please refer to [`SDXL_FP8_README.md`](SDXL_FP8_README.md) for further steps.**

## Build the TRT engine for the INT8 Quantized ONNX UNet

Please make sure the TensorRT environment setup is present. This example requires TensorRT version >= 9.2.0. You can download the prebuilt TensorRT [here](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.1.0/tars/TensorRT-10.1.0.27.Linux.x86_64-gnu.cuda-12.4.tar.gz) if you setup the environment yourself.

```sh
# INT8 SDXL Base or SDXL-turbo
trtexec --onnx=./unet.onnx --shapes=sample:2x4x128x128,timestep:1,encoder_hidden_states:2x77x2048,text_embeds:2x1280,time_ids:2x6 --fp16 --int8 --builderOptimizationLevel=4 --saveEngine=unetxl.trt9.3.0.post12.dev1.engine
# INT8 SD 1.5
trtexec --onnx=./unet.onnx --shapes=sample:2x4x64x64,timestep:1,encoder_hidden_states:2x77x768 --fp16 --int8 --builderOptimizationLevel=4 --saveEngine=unet.trt9.3.0.post12.dev1.engine
```

We tested int8 TRT engine build and the following E2E pipeline on the following GPU platforms: A100, L40S, RTX6000, A10, L4, RTX4090

## Run End-to-end Stable Diffusion Pipeline with Model Optimizer Quantized Model and demoDiffusion

If you want to run end-to-end SD/SDXL pipeline with Model Optimizer quantized UNet to generate images and measure latency on target GPUs, here are the steps:

- Clone a copy of [demo/Diffusion repo](https://github.com/NVIDIA/TensorRT/tree/release/10.0/demo/Diffusion).

- Following the README from demoDiffusion to set up the pipeline (note: you can skip the installation instructions if you use the docker container built in this repo), and run a baseline txt2img example (fp16):

```sh
# SDXL, please refer to the examples provided in the demoDiffusion SD/SDXL pipeline.
python demo_txt2img_xl.py "enchanted winter forest, soft diffuse light on a snow-filled day, serene nature scene, the forest is illuminated by the snow" --negative-prompt "normal quality, low quality, worst quality, low res, blurry, nsfw, nude" --version xl-1.0 --scheduler Euler --denoising-steps 30 --seed 2946901
```

Note, it will take some time to build TRT engines for the first time.

- Replace the fp16 UNet TRT engine with int8 engine generated in [Build the TRT engine for the INT8 Quantized ONNX UNet](#build-the-trt-engine-for-the-int8-quantized-onnx-unet), e.g.,:

```sh
cp -r unetxl.{TRT_VERSION}.engine ./engine/
```

The engines must be built on the same GPU, with the INT8 engine name matching the names of the FP16 engines to enable compatibility with the demoDiffusion pipeline.

- Run the above txt2img example command again. You can compare the generated images and latency for fp16 vs int8.

  1. FP16: ![Image generated with fp16 engine](./assets/xl_base-fp16.png)
  1. INT8: ![Image generated with int8 engine](./assets/xl_base-int8.png)

Similarly, you could run end-to-end SD1.5 or SDXL-turbo pipeline with Model Optimizer quantized UNet and corresponding examples in demoDiffusion.

## LoRA

For optimal performance of INT8/FP8 quantized models, we highly recommend fusing the LoRA weights prior to quantization. Failing to do so can disrupt TensorRT kernel fusion when integrating the LoRA layer with INT8/FP8 Quantize-Dequantize (QDQ) nodes, potentially leading to performance losses.

### Recommended Workflow:

Start by fusing the LoRA weights in your model. This process can help ensure that the model is optimized for quantization. Detailed guidance on how to fuse LoRA weights can be found in the Hugging Face [PEFT documentation](https://github.com/huggingface/peft):

After fusing the weights, proceed with the calibration and you can follow our code to do the quantization.

```python
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")
pipe.load_lora_weights(
    "CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy"
)
pipe.fuse_lora(lora_scale=0.9)
...
# All the LoRA layers should be fused
check_lora(pipe.unet)

mtq.quantize(pipe.unet, quant_config, forward_loop)
mto.save(pipe.unet, ...)
```

When it's time to export the model to ONNX format, ensure that you load the PEFT-modified LoRA model first.

```python
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.load_lora_weights(
    "CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy"
)
pipe.fuse_lora(lora_scale=0.9)
mto.restore(pipe.unet, your_quantized_ckpt)
...
# Export the onnx model
```

By following these steps, your PEFT LoRA model should be efficiently quantized using ModelOPT, ready for deployment while maximizing performance.

### Notes About Randomness

Stable Diffusion pipelines rely heavily on random sampling operations, which include creating Gaussian noise tensors to denoise and adding noise in the scheduling step. In the quantization recipe, we don't fix the random seed. As a result, every time you run the calibration pipeline, you could get different quantizer amax values. This may lead to the generated images being different from the ones generated with the original model. We suggest to run a few more times and choose the best one.
