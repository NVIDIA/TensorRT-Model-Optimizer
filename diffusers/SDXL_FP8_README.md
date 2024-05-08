## Run SDXL FP8 quantized model with TRT plugins

This page describes how to continue deploying Model Optimizer FP8 quantized SDXL model with TRT plugins after you get the FP8 ONNX model.

Note: We only support SDXL FP8 inference on Ada and Hopper GPUs (sm>=8.9).

### Run graphsurgeon

After exported the FP8 quantized SDXL ONNX model from Pytorch, a graph surgeon postprocess is required for FP8 ONNX model. The surgeon script only supports SDXL right now, SD1.5 and SD2.1 will be supported soon.

```sh
mkdir onnx_fp8_surgeoned && python ./onnx_utils/sdxl_fp8_graphsurgeon.py --onnx-path ./onnx_fp8/unet.onnx  --output-onnx ./onnx_fp8_surgeoned/sdxl_fp8_graphsurgeon.onnx
```

### Build TRT plugins

If you are already in the docker built with instructions in [`public/README.md`](../README.md), then you can go to the next step. Otherwise, build the TRT plugins by yourself. We require CUDA 12.x and TRT version >= 9.2.0.

Before building, update the addresses of "TRT" and "CUDA" according to your environment in file [`plugins/Makefile.config`](../plugins/Makefile.config).

Now build plugins.

```sh
cd plugins && make -j 4
```

If plugins are compiled successfully, you can find plugin files `plugins/bin/FP8Conv2DPlugin.so` and `plugins/bin/roupNormPlugin.so`.

Add the prebuilt kernel file [`plugins/prebuilt/libfp8convkernel.so`](../plugins/prebuilt/libfp8convkernel.so) to `LD_LIBRARY_PATH`:

```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libfp8convkernel_folder/
```

### Build the TRT engine for the FP8 Quantized ONNX UNet

We assume you already have TensorRT environment setup. This example requires TensorRT version >= 9.2.0. You can download the build [here](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.3.0/tensorrt-9.3.0.1.linux.x86_64-gnu.cuda-12.2.tar.gz) if you setup the environment by yourself.

```sh
# FP8 SDXL
trtexec --onnx=./onnx_fp8_surgeoned/sdxl_fp8_graphsurgeon.onnx --fp16 --saveEngine=fp8.engine --shapes=sample:2x4x128x128,timestep:1,encoder_hidden_states:2x77x2048,text_embeds:2x1280,time_ids:2x6 --builderOptimizationLevel=4 --fp8 --staticPlugins=./plugins/bin/FP8Conv2DPlugin.so --staticPlugins=./plugins/bin/groupNormPlugin.so
```

Make sure folder path to [`plugins/prebuilt/libfp8convkernel.so`](../plugins/prebuilt/libfp8convkernel.so) has been added in `LD_LIBRARY_PATH` before running `trtexec` command.

We tested FP8 TRT engine build and the following E2E pipeline on the following GPU platforms: L20, H20, RTX6000.

### Run End-to-end Stable Diffusion Pipeline with Model Optimizer Quantized ONNX Model and demoDiffusion

Follow instructions in the same section in [`README.md`](README.md), the only extra step when running E2E inference with fp8 TRT engine is:

- Load plugins files by adding the following code in `demo_txt2img_xl.py`

```sh
import ctypes
soFiles = [
    "/path to your/plugins/bin/FP8Conv2DPlugin.so",
    "/path to your/plugins/bin/groupNormPlugin.so"
]

for soFile in soFiles:
    ctypes.cdll.LoadLibrary(soFile)
```

Note, the engines must be built on the same GPU, and ensure that the FP8 engine name matches the names of the FP16 engines to enable compatibility with the demoDiffusion pipeline.

Here's an example prompt *"highly detailed concept art of a sakura plum tree made with water, overgrowth, Tristan Eaton, Artgerm, Studio Ghibli, Makoto Shinkai"* to run the txt2img pipeline, and you can compare the generated images and latency for fp16 vs fp8.

1. FP16: ![Image generated with fp16 engine](./assets/xl_base-fp16-sakura.png)
1. FP8: ![Image generated with fp8 engine](./assets/xl_base-fp8-sakura.png)
