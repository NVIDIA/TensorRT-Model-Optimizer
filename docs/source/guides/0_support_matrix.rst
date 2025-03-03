.. _Support_Matrix:

==============
Support Matrix
==============

Feature Support Matrix
======================

.. tab:: Linux

    .. list-table::
        :widths: 20 40 20 20
        :header-rows: 1
        :stub-columns: 1

        * - Quantization Format
          - Details
          - Supported Model Formats
          - Deployment
        * - FP8
          - * Per-Tensor FP8 Weight & Activations
            * GPUs: Ada and Later
          - PyTorch, ONNX*
          - TensorRT*, TensorRT-LLM
        * - INT8
          - * Per-channel INT8 Weights, Per-Tensor FP8 Activations
            * Uses Smooth Quant Algorithm
            * GPUs: Ampere and Later
          - PyTorch, ONNX*
          - TensorRT*, TensorRT-LLM
        * - W4A16 (INT4 Weights Only)
          - * Block-wise INT4 Weights, F16 Activations
            * Uses AWQ Algorithm
            * GPUs: Ampere and Later
          - PyTorch, ONNX
          - TensorRT, TensorRT-LLM
        * - W4A8 (INT4 Weights, FP8 Activations)
          - * Block-wise INT8 Weights, Per-Tensor FP8 Activations
            * Uses AWQ Algorithm
            * GPUs: Ada and Later
          - PyTorch*, ONNX*
          - TensorRT-LLM

.. tab:: Windows

    .. list-table::
        :widths: 20 40 20 20
        :header-rows: 1
        :stub-columns: 1

        * - Quantization Format
          - Details
          - Supported Model Formats
          - Deployment
        * - W4A16 (INT4 Weights Only)
          - * Block-wise INT4 Weights, F16 Activations
            * Uses AWQ Algorithm
            * GPUs: Ampere and Later
          - PyTorch*, ONNX
          - ORT-DirectML, TensorRT*, TensorRT-LLM*
        * - W4A8 (INT4 Weights, FP8 Activations)
          - * Block-wise INT8 Weights, Per-Tensor FP8 Activations
            * Uses AWQ Algorithm
            * GPUs: Ada and Later
          - PyTorch*
          - TensorRT-LLM*
        * - FP8
          - * Per-Tensor FP8 Weight & Activations
            * GPUs: Ada and Later
          - PyTorch*, ONNX*
          - TensorRT*, TensorRT-LLM*
        * - INT8
          - * Per-channel INT8 Weights, Per-Tensor FP8 Activations
            * Uses Smooth Quant Algorithm
            * GPUs: Ada and Later
          - PyTorch*, ONNX*
          - TensorRT*, TensorRT-LLM*

.. note:: Features marked with an asterisk (*) are considered experimental.


Model Support Matrix
====================

.. tab:: Linux

    Please checkout the model support matrix `here <https://github.com/NVIDIA/TensorRT-Model-Optimizer?tab=readme-ov-file#model-support-matrix>`_.

.. tab:: Windows

    .. list-table::
        :header-rows: 1

        * - Model
          - ONNX INT4 AWQ
        * - Llama3.1-8B-Instruct
          - Yes
        * - Phi3.5-mini-Instruct
          - Yes
        * - Mistral-7B-Instruct-v0.3
          - Yes
        * - Llama3.2-3B-Instruct
          - Yes
        * - Gemma-2b-it
          - Yes
        * - Nemotron Mini 4B Instruct
          - Yes
