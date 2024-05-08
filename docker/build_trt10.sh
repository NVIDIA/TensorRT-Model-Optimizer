set -e

docker build . -f docker/Dockerfile.trt10 -t modelopt_onnx_examples:latest "$@"
