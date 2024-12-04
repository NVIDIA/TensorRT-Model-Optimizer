# Diffusion Models Quantization with Model Optimizer

This example shows how to use Model Optimizer to calibrate and quantize the backbone part of diffusion models. The backbone part typically consumes >95% of the e2e diffusion latency.

We also provide instructions on deploying and running E2E diffusion pipelines with Model Optimizer quantized INT8 and FP8 Backbone to generate images and measure latency on target GPUs. Note, Jetson devices are not supported so far.

## 8-bit ONNX Export Quick Start

You can run the following script to get the INT8 or FP8 Backbone onnx model built with default settings for SDXL, and then directly go to **Build the TRT engine for the Quantized ONNX Backbone** section to run E2E pipeline to generate images.

```sh
bash build_sdxl_8bit_engine.sh --format {FORMAT} # FORMAT can be int8 or fp8
```

If you prefer to customize parameters in calibration or run other models, please follow the instructions below.

## Calibration with Model Optimizer

We support calibration for both INT8 and FP8 precision and for both weights and activations.

Note: Model calibration requires relatively more GPU computing powers and it does not need to be on the same GPUs as
the deployment target GPUs. Using the command line below will execute both calibration and ONNX export.

### FLUX-Dev|SD3-Medium|SDXL|SD2.1|SDXL-Turbo INT8

```sh
python quantize.py \
  --model {flux-dev|sdxl-1.0|sdxl-turbo|sd2.1|sd2.1-base|sd3-medium} \
  --format int8 --batch-size 2 \
  --calib-size 32 --collect-method min-mean \
  --percentile 1.0 --alpha 0.8 \
  --quant-level 3.0 --n-steps 20 \
  --model-dtype {Half/BFloat16} \
  --quantized-torch-ckpt-save-path ./{MODEL}_int8.pt --onnx-dir {ONNX_DIR}
```

### FLUX-Dev|SDXL|SD2.1|SDXL-Turbo FP8

```sh
python quantize.py \
  --model {flux-dev|sdxl-1.0|sdxl-turbo|sd2.1|sd2.1-base} --model-dtype {Half|BFloat16} \
  --format fp8 --batch-size 2 --calib-size {128|256} --quant-level {3.0|4.0} \
  --n-steps 20 --quantized-torch-ckpt-save-path ./{MODEL}_fp8.pt --collect-method default \
  --onnx-dir {ONNX_DIR}
```

We recommend using a device with a minimum of 48GB of combined CPU and GPU memory for exporting ONNX models. Quant-level 4.0 requires additional memory. Exporting the **FLUX-Dev** ONNX model requires a minimum of 60GB of memory.

#### Important Parameters

- `percentile`: Control quantization scaling factors (amax) collecting range, meaning that we will collect the chosen amax in the range of `(n_steps * percentile)` steps. Recommendation: 1.0

- `alpha`: A parameter in SmoothQuant, used for linear layers only. Recommendation: 0.8 for SDXL

- `quant-level`: Which layers to be quantized, 1: `CNNs`, 2: `CNN + FFN`, 2.5: `CNN + FFN + QKV`, 3: `CNN + Almost all Linear (Including FFN, QKV, Proj and others)`, 4: `CNN + Almost all Linear + fMHA`. Recommendation: 2, 2.5 and 3, 4 is only for FP8, depending on the requirements for image quality & speedup. **You might notice a slight difference between FP8 quant level 3.0 and 4.0, as we are currently working to enhance the performance of FP8 fMHA.**

- `calib-size`: For SDXL INT8, we recommend 32 or 64, for SDXL FP8, 128 is recommended.

- `n_steps`: Recommendation: SD/SDXL 20 or 30, SDXL-Turbo 4.

**Then, we can load the generated checkpoint and export the INT8/FP8 quantized model in the next step. For FP8, we only support the TRT deployment on Ada/Hopper GPUs.**

## Build the TRT engine for the Quantized ONNX Backbone

We assume you already have TensorRT environment setup. INT8 requires **TensorRT version >= 9.2.0**. If you prefer to use the FP8 TensorRT, ensure you have **TensorRT version 10.2.0 or higher**. You can download the latest version of TensorRT at [here](https://developer.nvidia.com/tensorrt/download).

Then generate the INT8/FP8 Backbone Engine

```bash

# For SDXL
trtexec --builderOptimizationLevel=4 --stronglyTyped --onnx=./backbone.onnx \
  --minShapes=sample:2x4x128x128,timestep:1,encoder_hidden_states:2x77x2048,text_embeds:2x1280,time_ids:2x6 \
  --optShapes=sample:16x4x128x128,timestep:1,encoder_hidden_states:16x77x2048,text_embeds:16x1280,time_ids:16x6 \
  --maxShapes=sample:16x4x128x128,timestep:1,encoder_hidden_states:16x77x2048,text_embeds:16x1280,time_ids:16x6 \
  --saveEngine=backbone.plan

# For SD3-Medium
trtexec --builderOptimizationLevel=4 --stronglyTyped --onnx=./backbone.onnx \
  --minShapes=hidden_states:2x16x128x128,timestep:2,encoder_hidden_states:2x333x4096,pooled_projections:2x2048 \
  --optShapes=hidden_states:16x16x128x128,timestep:16,encoder_hidden_states:16x333x4096,pooled_projections:16x2048 \
  --maxShapes=hidden_states:16x16x128x128,timestep:16,encoder_hidden_states:16x333x4096,pooled_projections:16x2048 \
  --saveEngine=backbone.plan

# For FLUX-Dev FP8
trtexec --onnx=./backbone.onnx --fp8 --bf16 --stronglyTyped \
  --minShapes=hidden_states:1x4096x64,img_ids:1x4096x3,encoder_hidden_states:1x512x4096,txt_ids:1x512x3,timestep:1,pooled_projections:1x768,guidance:1 \
  --optShapes=hidden_states:1x4096x64,img_ids:1x4096x3,encoder_hidden_states:1x512x4096,txt_ids:1x512x3,timestep:1,pooled_projections:1x768,guidance:1 \
  --maxShapes=hidden_states:1x4096x64,img_ids:1x4096x3,encoder_hidden_states:1x512x4096,txt_ids:1x512x3,timestep:1,pooled_projections:1x768,guidance:1 \
  --saveEngine=backbone.plan
```

**Please note that `maxShapes` represents the maximum shape of the given tensor. If you want to use a larger batch size or any other dimensions, feel free to adjust the value accordingly.**

## Run End-to-end Stable Diffusion Pipeline with Model Optimizer Quantized ONNX Model and demoDiffusion

### demoDiffusion

If you want to run end-to-end SD/SDXL pipeline with Model Optimizer quantized UNet to generate images and measure latency on target GPUs, here are the steps:

- Clone a copy of [demo/Diffusion repo](https://github.com/NVIDIA/TensorRT/tree/release/10.2/demo/Diffusion).

- Following the README from demoDiffusion to set up the pipeline, and run a baseline txt2img example (fp16):

```sh
# SDXL
python demo_txt2img_xl.py "enchanted winter forest, soft diffuse light on a snow-filled day, serene nature scene, the forest is illuminated by the snow" --negative-prompt "normal quality, low quality, worst quality, low res, blurry, nsfw, nude" --version xl-1.0 --scheduler Euler --denoising-steps 30 --seed 2946901
# Please refer to the examples provided in the demoDiffusion SD/SDXL pipeline.
```

Note, it will take some time to build TRT engines for the first time

- Replace the fp16 backbone TRT engine with int8 engine generated in [Build the TRT engine for the Quantized ONNX Backbone](#build-the-trt-engine-for-the-quantized-onnx-backbone), e.g.,:

```sh
cp -r {YOUR_UNETXL}.plan ./engine/
```

Note, the engines must be built on the same GPU, and ensure that the INT8 engine name matches the names of the FP16 engines to enable compatibility with the demoDiffusion pipeline.

- Run the above txt2img example command again. You can compare the generated images and latency for fp16 vs int8.
  Similarly, you could run end-to-end pipeline with Model Optimizer quantized backbone and corresponding examples in demoDiffusion with other diffusion models.

### Running the inference pipeline with DeviceModel

DeviceModel is an interface designed to run TensorRT engines like torch models. It takes torch inputs and returns torch outputs. Under the hood, DeviceModel exports a torch checkpoint to ONNX and then generates a TensorRT engine from it. This allows you to swap the backbone of the diffusion pipeline with DeviceModel and execute the pipeline for your desired prompt.<br><br>

Generate a quantized torch checkpoint using the command shown below:

```bash
python quantize.py \
  --model {sdxl-1.0|sdxl-turbo|sd2.1|sd2.1-base|sd3-medium|flux-dev|flux-schnell} \
  --format fp8 \
  --batch-size {1|2} \
  --calib-size 128 \
  --quant-level 3.0 \
  --n-steps 20 \
  --quantized-torch-ckpt-save-path ./{MODEL}_fp8.pt \
  --collect-method default \
```

Generate images for the quantized checkpoint with the following command:

```bash
python diffusion_trt.py \
  --model {sdxl-1.0|sdxl-turbo|sd2.1|sd2.1-base|sd3-medium|flux-dev|flux-schnell} \
  --prompt "A cat holding a sign that says hello world" \
  [--restore-from ./{MODEL}_fp8.pt] \
  [--onnx-load-path {ONNX_DIR}] \
  [--trt_engine-path {ENGINE_DIR}]
```

This script will save the output image as `./{MODEL}.png` and report the latency of the TensorRT backbone.
To generate the image with FP16|BF16 precision, you can run the command shown above without the `--restore-from` argument.<br><br>

## Demo Images

<table align="center">
  <tr>
    <td align="center">
      <img src="./assets/xl_base-fp16.png" alt="FP16" width="700"/>
    </td>
    <td align="center">
      <img src="./assets/xl_base-int8.png" alt="INT8" width="700"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      SDXL FP16
    </td>
    <td align="center">
      SDXL INT8
    </td>
  </tr>
</table>

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
