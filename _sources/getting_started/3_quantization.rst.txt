=========================
Quick Start: Quantization
=========================

Quantization
------------

Quantization is an effective technique to reduce the memory footprint of deep learning models and to
accelerate the inference speed.

ModelOpt's :meth:`mtq.quantize() <modelopt.torch.quantization.model_quant.quantize>` API enables
users to quantize a model with advanced algorithms like SmoothQuant, AWQ, SVDQuant, and more. ModelOpt
supports both Post Training Quantization (PTQ) and Quantization Aware Training (QAT).

.. tip::

    Please refer to :any:`quantization-formats` for details on the ModelOpt supported quantization
    formats and their use-cases.

PTQ for PyTorch models
-----------------------------

:meth:`mtq.quantize <modelopt.torch.quantization.model_quant.quantize>` requires the model,
the appropriate quantization configuration, and a forward loop as inputs. Here is a quick example of
quantizing a model with int8 SmoothQuant using
:meth:`mtq.quantize <modelopt.torch.quantization.model_quant.quantize>`:

.. code-block:: python

    import modelopt.torch.quantization as mtq

    # Setup the model
    model = get_model()

    # The quantization algorithm requires calibration data. Below we show a rough example of how to
    # set up a calibration data loader with the desired calib_size
    data_loader = get_dataloader(num_samples=calib_size)


    # Define the forward_loop function with the model as input. The data loader should be wrapped
    # inside the function.
    def forward_loop(model):
        for batch in data_loader:
            model(batch)


    # Quantize the model and perform calibration (PTQ)
    model = mtq.quantize(model, mtq.INT8_SMOOTHQUANT_CFG, forward_loop)

Refer to :any:`quantization-configs` for the quantization configurations available from ModelOpt.

Deployment
----------------

The quantized model is just like a regular Pytorch model and is ready for evaluation or deployment.

Huggingface or Nemo LLM models can be exported to TensorRT-LLM using ModelOpt.
Please see the :doc:`TensorRT-LLM Deployment <../deployment/1_tensorrt_llm>` guide for
more details.

The model can be also exported to ONNX using
`torch.onnx.export <https://pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export>`_.

--------------------------------

**Next Steps**
    * Learn more about quantization and advanced usage of Model Optimizer quantization in
      :doc:`Quantization guide <../guides/1_quantization>`.
    * Checkout out the end-to-end examples on GitHub for PTQ and QAT
      `here <https://github.com/NVIDIA/TensorRT-Model-Optimizer?tab=readme-ov-file#examples>`_.
