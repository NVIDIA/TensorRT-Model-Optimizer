# Model Optimizer End-to-End Tests

This directory contains end-to-end tests for the Model Optimizer's [examples](../../examples/).

## Adding new tests

To add a test for a new example, create a new example directory in `tests/examples` following the existing examples as guidance.
Make sure to use as small models and less data as possible to keep the tests fast. Unless needed, have tests finish under 15 minutes.

## Running the tests

To run a test, use the [ModelOpt docker image](../../README.md#installation--docker) so all required dependencies are available.
and mount your local modelopt directory to `/workspace/TensorRT-Model-Optimizer` and run this from the root of the repository.

```bash
pytest tests/examples/$TEST
```

NOTE: Some tests (e.g. `llm_ptq`) have an option to disable using a smaller proxy model, and instead use the original model by setting the `MODELOPT_FAST_TESTS` environment variable to `false`. This is useful in nightly tests to ensure the original model is used.

```bash
MODELOPT_FAST_TESTS=false ROOT_SAVE_PATH=/tmp/test_llm_ptq/ pytest tests/examples/llm_ptq/
```
