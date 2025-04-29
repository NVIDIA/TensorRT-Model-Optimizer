.. _Quantization_Quick_Start_Windows:

===================================
Quick Start: Quantization (Windows)
===================================

Quantization is a crucial technique for reducing memory usage and speeding up inference in deep learning models.

The ONNX quantization API in ModelOpt-Windows offers advanced Post-Training Quantization (PTQ) options like Activation-Aware Quantization (AWQ).

ONNX Model Quantization (PTQ)
------------------------------

The ONNX quantization API requires a model, calibration data, along with quantization settings like algorithm, calibration-EPs etc. Here’s an example snippet to apply INT4 AWQ quantization:

.. code-block:: python

    from modelopt.onnx.quantization.int4 import quantize as quantize_int4
    # import other packages as needed
    calib_inputs = get_calib_inputs(dataset, model_name, cache_dir, calib_size, batch_size,...)
    quantized_onnx_model = quantize_int4(
        onnx_path,
        calibration_method="awq_lite",
        calibration_data_reader=None if use_random_calib else calib_inputs,
        calibration_eps=["dml", "cpu"]
    )
    onnx.save_model(
        quantized_onnx_model,
        output_path,
        save_as_external_data=True,
        location=os.path.basename(output_path) + "_data",
        size_threshold=0,
    )

Check :meth:`modelopt.onnx.quantization.quantize_int4 <modelopt.onnx.quantization.int4.quantize>` for details about INT4 quantization API.

Refer :ref:`Support_Matrix` for details about supported features and models.

To know more about ONNX PTQ, refer :ref:`ONNX_PTQ_Guide_Windows` and `example scripts <https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/windows/onnx_ptq/>`_.


Deployment
----------
The quantized onnx model can be deployed using frameworks like onnxruntime. Ensure that model's opset is 19+ for FP8 quantization, and it is 21+ for INT4 quantization. This is needed due to different opset requirements of  ONNX's `Q <https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html>`_/`DQ <https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html>`_ nodes for INT4, FP8 data-types support. Refer :ref:`Apply_ONNX_PTQ` for details.

.. code-block:: python

    # write steps (say, upgrade_opset() method) to upgrade or patch opset of the model, if needed
    # the opset-upgrade, if needed, can be done on either base ONNX model or on the quantized model
    # finally, save the quantized model

    quantized_onnx_model = upgrade_opset(quantized_onnx_model)
    onnx.save_model(
        quantized_onnx_model,
        output_path,
        save_as_external_data=True,
        location=os.path.basename(output_path) + "_data",
        size_threshold=0,
    )

For detailed instructions about deployment of quantized models with DirectML backend (ORT-DML), see the :ref:`DirectML_Deployment`. Also, refer `example scripts <https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/windows/onnx_ptq/>`_ for any possible model-specific inference guidance or script (if any).

.. note::

    The ready-to-deploy optimized ONNX models from ModelOpt-Windows are available at HuggingFace `NVIDIA collections <https://huggingface.co/collections/nvidia/optimized-onnx-models-for-nvidia-rtx-gpus-67373fe7c006ebc1df310613>`_.
