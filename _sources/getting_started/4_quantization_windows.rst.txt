.. _Quantization_Quick_Start_Windows:

===================================
Quick Start: Quantization (Windows)
===================================

Quantization is a crucial technique for reducing memory usage and speeding up inference in deep learning models.

The ONNX quantization API in ModelOpt-Windows offers advanced Post-Training Quantization (PTQ) options like Activation-Aware Quantization (AWQ).

ONNX Model Quantization (PTQ)
------------------------------

The ONNX quantization API requires a model, calibration data, along with quantization settings like algorithm, calibration-EPs etc. Here’s an example implementing int4 AWQ:

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

Check :meth:`modelopt.onnx.quantization.quantize_int4 <modelopt.onnx.quantization.int4.quantize>` for details about quantization API.

Refer :ref:`Support_Matrix` for details about supported features and models.

To know more about ONNX PTQ, refer :ref:`ONNX_PTQ_Guide_Windows` and `example script <https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/windows/onnx_ptq/>`_.


Deployment
----------
The quantized ONNX model is deployment-ready, equivalent to a standard ONNX model. ModelOpt-Windows uses ONNX’s `DequantizeLinear <https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html>`_ (DQ) nodes, which support INT4 data-type from opset version 21 onward. Ensure the model’s opset version is 21 or higher. Refer :ref:`Apply_ONNX_PTQ` for details.

.. code-block:: python

    # write steps (say, upgrade_opset_to_21() method) to upgrade opset to 21, if it is lower than 21.

    quantized_onnx_model = upgrade_opset_to_21(quantized_onnx_model)
    onnx.save_model(
        quantized_onnx_model,
        output_path,
        save_as_external_data=True,
        location=os.path.basename(output_path) + "_data",
        size_threshold=0,
    )

Deploy the quantized model using the DirectML backend. For detailed deployment instructions, see the :ref:`DirectML_Deployment`.

.. note::

    The ready-to-deploy optimized ONNX models from ModelOpt-Windows are available at HuggingFace `NVIDIA collections <https://huggingface.co/collections/nvidia/optimized-onnx-models-for-nvidia-rtx-gpus-67373fe7c006ebc1df310613>`_.
