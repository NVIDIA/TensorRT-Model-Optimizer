# ModelOpt Docker

This folder contains the Dockerfile for the ModelOpt docker image.

## Building the Docker Image

To build the docker image, run the following command from the root of the repository:

```bash
bash docker/build.sh
```

The docker image will be built and tagged as `docker.io/library/modelopt_examples:latest`.

> [!NOTE]
> For ONNX PTQ, use the optimized docker image from [onnx_ptq Dockerfile](../examples/onnx_ptq/docker/) instead of this one.
