# Model Optimizer End-to-End Tests

This directory contains end-to-end tests for the Model Optimizer's [examples](../../examples/).

## Adding new tests

To add a test for a new example, create a new script in the `test_example.sh` script following the existing examples as guidance.
Make sure to use as small models and less data as possible to keep the tests fast. If possible, have tests finish under 15 minutes.

## Running the tests

To run a test, use the [ModelOpt docker image](../../README.md#installation--docker) so all required dependencies are available.
and mount your local modelopt directory to `/workspace/TensorRT-Model-Optimizer` and cd to this directory.

```bash
bash test_example.sh
```

NOTE: Some tests (e.g. `test_llm_ptq.sh`) have an option to disable using a smaller proxy model, and instead use the original model by setting the `MODELOPT_FAST_TESTS` environment variable to `false`. This is useful in nightly tests to ensure the original model is used.

```bash
MODELOPT_FAST_TESTS=false bash test_llm_ptq.sh
```
