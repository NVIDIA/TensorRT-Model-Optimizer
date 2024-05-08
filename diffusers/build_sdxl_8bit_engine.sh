#!/bin/bash
set -e
set -x

# Default format
format="int8"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --format)
            format="$2"
            shift
            shift
            ;;
        *)  # Unknown option
            echo "Unknown option: $key"
            exit 1
            ;;
    esac
done

# Check if the format is valid
if [[ "$format" != "int8" && "$format" != "fp8" ]]; then
    echo "Invalid format. Please choose either 'int8' or 'fp8'."
    exit 1
fi


# Assume this script is launched with the docker built from "docker" according to README.md
fp8_plugin="/workspace/examples/plugins/bin/FP8Conv2DPlugin.so"
groupNorm_plugin="/workspace/examples/plugins/bin/groupNormPlugin.so"

# Move to the diffusers directory
echo "current folder at $PWD"

# Configurations for quantization
model="stabilityai/stable-diffusion-xl-base-1.0"
quant_level=3

echo "=====>Assume the script is launched with the docker built from "docker" according to README.md. If not, please include the path to the plugins manually."
export LD_LIBRARY_PATH=/workspace/public/plugins/prebuilt:$LD_LIBRARY_PATH


cleaned_m="${model//\//-}"
curt_exp="${cleaned_m}_${quant_level}_${format}"
echo "=====>Processing $curt_exp"
python quantize.py --model "$model" --format "$format" --batch-size 2 --calib-size 64 --percentile 1.0 --alpha 0.8 --quant-level "$quant_level" --n_steps 20 --exp_name "$curt_exp"

onnx_folder="./onnx_${curt_exp}"
mkdir -p "$onnx_folder"
echo "=====>Exporting to ONNX model, it will take a few minutes."
python run_export.py --model "$model" --quantized-ckpt "unet.state_dict.${cleaned_m}_${quant_level}_${format}.pt" --format "$format" --quant-level "$quant_level" --onnx-dir "$onnx_folder"

# Only run graph surgeon on certain conditions

fp8=""
if [ "$format" == "fp8" ]; then
    fp8="_fp8"
fi
python onnx_utils/sdxl${fp8}_graphsurgeon.py --onnx-path "./$onnx_folder/unet.onnx" --output-onnx "./$onnx_folder/unet.onnx"

engine_folder="./engine_${curt_exp}"
mkdir -p "$engine_folder"

# Build engine conditionally based on model and format
plugin_flags=" --int8 "
if [ "$format" == "fp8" ]; then
    plugin_flags=" --fp8 --staticPlugins=$fp8_plugin --staticPlugins=$groupNorm_plugin "
fi

model_flags=" --shapes=sample:2x4x128x128,timestep:1,encoder_hidden_states:2x77x2048,text_embeds:2x1280,time_ids:2x6 "

trtflags="$model_flags $plugin_flags"
echo "=====>Building TRT engine for $curt_exp, it will take some time, please be patient."
trtexec --onnx="./$onnx_folder/unet.onnx" \
    --fp16 --builderOptimizationLevel=4 \
    --saveEngine="./$engine_folder/unet.engine" \
    $trtflags

# Verify the TRT engine creation
if [ ! -f "./$engine_folder/unet.engine" ]; then
    echo "./$engine_folder/unet.engine not found, exiting."
    exit -1
fi
