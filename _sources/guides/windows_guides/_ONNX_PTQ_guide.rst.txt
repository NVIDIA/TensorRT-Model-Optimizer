.. _ONNX_PTQ_Guide_Windows:

===========================
ONNX Quantization - Windows
===========================

Quantization is a powerful method to reduce model size and boost computational efficiency. By converting model weights to low-precision formats, significant storage and memory savings are achieved. Quantization applies lower-precision data types to model parameters; for example:

- FP16 uses 2 bytes per value.
- INT4 requires only 4 bits (0.5 bytes) per value.

This transformation reduces model size and allows deployment on systems with limited memory.

**Quantization Techniques**:

1. **Post-Training Quantization (PTQ)**: Quantization is applied after model training.
2. **Quantization-Aware Training (QAT)**: Quantization is integrated during model training.


ModelOpt-Windows Quantization
-----------------------------

The TensorRT Model Optimizer - Windows is designed to create optimized ONNX models for DirectML and TensorRT* backends.

**Supported Techniques**:

- PTQ: Supported in ONNX
- QAT: Experimental for PyTorch

Refer :ref:`Support_Matrix` for details about supported features and models.

Preparing and Optimizing Base Models for ONNX PTQ
-------------------------------------------------

ModelOpt-Windows's ONNX PTQ API requires a base ONNX model, which can be obtained from Hugging Face or various ONNX exporter tools such as:

- `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html>`_
- `HuggingFace-Optimum <https://huggingface.co/docs/optimum/en/exporters/onnx/usage_guides/export_a_model/>`_
- `GenAI <https://github.com/microsoft/onnxruntime-genai/>`_
- `Olive <https://github.com/microsoft/Olive/>`_

Each tool offers unique features and options for conversion from PyTorch or other frameworks to the ONNX format.

**Base Model Precision**: ModelOpt-Windows supports base models in both FP16 and FP32 formats. Choosing FP16 over FP32 can help reduce memory usage and improve speed, especially on hardware optimized for lower precision, such as NVIDIA GPUs with Tensor Cores. However, FP16's smaller dynamic range may require careful tuning.


**ONNX FP16 Conversion Tools**: Some popular FP32 to FP16 ONNX conversion tools include:

- `ONNX Converter Tool  <https://onnxruntime.ai/docs/performance/model-optimizations/float16.html>`_
- Hugging Face *Optimum* tool with a *dtype* argument for FP16 generation - Refer to optimum's `CLI <https://huggingface.co/docs/optimum/en/exporters/onnx/usage_guides/export_a_model/>`_, and `API <https://github.com/huggingface/optimum/blob/main/optimum/exporters/onnx/convert.py/>`_ usage.
- Microsoft Olive, which supports FP16 via configuration files. Refer *float16* option in this example `config <https://github.com/microsoft/Olive/blob/main/examples/directml/llm/config_llm.json/>`_.


Once base model is obtained, ModelOpt-Windows's PTQ can be applied to get the quantized mode. The resulting quantized model can be deployed on DirectML and TensorRT* backend.

.. _Apply_ONNX_PTQ:

Apply Post Training Quantization (PTQ)
--------------------------------------

**Prepare calibration data**

The SmoothQuant (SQ) and Activation-Aware-Quantization (AWQ) algorithms require calibration data during quantization. If the *quantize* API's calibration-data argument is not provided (i.e., set to *None*), ModelOpt-Windows will internally use randomly generated model inputs for calibration. Refer to the sample code below for preparing calibration inputs.

Preparing calibration data for ModelOpt-Windows involves two steps:

1. **Generate Token Encodings**: Use a dataset like *cnn-dailymail* or *pile* with the model's tokenizer to generate token encodings and related data from the representative dataset
2. **Format for Model Input**: Convert encodings into model-compatible formats.

See the code example below for details.


.. code-block:: python

  # Refer get_calib_inputs() method below to prepare calibration inputs for your model.

  # Note that names and shapes of inputs and outputs can vary from model to model, and also between ONNX exporter tools.
  # So, use following code as reference for preparing calibration data for your model.

  def make_model_input(
      config,
      input_ids_arg,
      attention_mask_arg,
      add_past_kv_inputs,
      device,
      use_fp16,
      use_buffer_share,
      add_position_ids,
  ):
      input_ids = input_ids_arg
      attention_mask = attention_mask_arg

      if isinstance(input_ids_arg, list):
          input_ids = torch.tensor(input_ids_arg, device=device, dtype=torch.int64)
          attention_mask = torch.tensor(attention_mask_arg, device=device, dtype=torch.int64)

      inputs = {
          "input_ids": input_ids.contiguous(),
          "attention_mask": attention_mask.contiguous(),
      }

      if add_position_ids:
          position_ids = attention_mask.long().cumsum(-1) - 1
          position_ids.masked_fill_(attention_mask == 0, 1)
          inputs["position_ids"] = position_ids.contiguous()

      if add_past_kv_inputs:
          torch_dtype = torch.float16 if use_fp16 else torch.float32
          batch_size, sequence_length = input_ids.shape
          max_sequence_length = config.max_position_embeddings
          num_heads, head_size = (
              config.num_key_value_heads,
              config.hidden_size // config.num_attention_heads,
          )

          if hasattr(config, "head_dim"):
              head_size = config.head_dim

          for i in range(config.num_hidden_layers):
              past_key = torch.zeros(
                  batch_size,
                  num_heads,
                  max_sequence_length if use_buffer_share else 0,
                  head_size,
                  device=device,
                  dtype=torch_dtype,
              )
              past_value = torch.zeros(
                  batch_size,
                  num_heads,
                  max_sequence_length if use_buffer_share else 0,
                  head_size,
                  device=device,
                  dtype=torch_dtype,
              )
              inputs.update(
                  {
                      f"past_key_values.{i}.key": past_key.contiguous(),
                      f"past_key_values.{i}.value": past_value.contiguous(),
                  }
              )

      return inputs


  def get_calib_inputs(
      dataset_name,
      model_name,
      cache_dir,
      calib_size,
      batch_size,
      block_size,
      device,
      use_fp16,
      use_buffer_share,
      add_past_kv_inputs,
      max_calib_rows_to_load,
      add_position_ids,
      trust_remote_code,
  ):
      # from transformers import LlamaConfig
      # config = LlamaConfig.from_pretrained(
      #     model_name, use_auth_token=True, cache_dir=cache_dir, trust_remote_code=trust_remote_code
      # )
      config = AutoConfig.from_pretrained(
          model_name, use_auth_token=True, cache_dir=cache_dir, trust_remote_code=trust_remote_code
      )
      tokenizer = AutoTokenizer.from_pretrained(
          model_name, use_auth_token=True, cache_dir=cache_dir, trust_remote_code=trust_remote_code
      )
      tokenizer.add_special_tokens({"pad_token": "[PAD]"})
      tokenizer.pad_token = tokenizer.eos_token

      assert (
          calib_size <= max_calib_rows_to_load
      ), "calib size should be no more than max_calib_rows_to_load"

      dataset2 = load_dataset("cnn_dailymail", name="3.0.0", split="train").select(range(max_calib_rows_to_load))
      column = "article"

      # dataset2 = dataset2.shuffle(seed=42)
      dataset2 = dataset2[column][:calib_size]
      batch_encoded = tokenizer.batch_encode_plus(
          dataset2, return_tensors="pt", padding=True, truncation=True, max_length=block_size
      )  # return_tensors="pt",
      batch_encoded = batch_encoded.to(device)
      batch_encoded_input_ids = batch_encoded["input_ids"]
      batch_encoded_attention_mask = batch_encoded["attention_mask"]
      calib_dataloader_input_ids = DataLoader(batch_encoded_input_ids, batch_size=batch_size, shuffle=False)
      calib_dataloader_attenton_mask = DataLoader(batch_encoded_attention_mask, batch_size=batch_size, shuffle=False)

      number_of_batched_samples = calib_size // batch_size

      batched_input_ids = []
      for idx, data in enumerate(calib_dataloader_input_ids):
          batched_input_ids.append(data)
          if idx == (number_of_batched_samples - 1):
              break

      batched_attention_mask = []
      for idx, data in enumerate(calib_dataloader_attenton_mask):
          batched_attention_mask.append(data)
          if idx == (number_of_batched_samples - 1):
              break

      batched_inputs_list = []
      for i in range(number_of_batched_samples):
          input_ids = batched_input_ids[i]
          attention_mask = batched_attention_mask[i]

          inputs = make_model_input(config, input_ids, attention_mask, add_past_kv_inputs, device,
                                    use_fp16,
                                    use_buffer_share,
                                    add_position_ids,
          )
          inputs = {
              input_name: torch_tensor.cpu().numpy() for input_name, torch_tensor in inputs.items()
          }
          batched_inputs_list.append(inputs)

      return batched_inputs_list

**Call Quantization API**

The example below demonstrates how to quantize an ONNX model using ModelOpt-Windows with INT4 precision.

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

Check :meth:`modelopt.onnx.quantization.quantize_int4 <modelopt.onnx.quantization.int4.quantize>` for details about quantization API.

**Upgrade Opset to 21+**

ModelOpt-Windows uses ONNX’s `DequantizeLinear <https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html>`_ (DQ) nodes, which support INT4 data-type from opset version 21 onward. Ensure the model’s opset version is 21 or higher, for deployment on DirectML backend.

.. code-block:: python

  # Example steps to check opset

  def get_onnx_opset(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    # Get the opset import information
    opset_imports = model.opset_import

    # Print opset information
    for opset in opset_imports:
      print(f"Domain: {opset.domain}")
      print(f"  Version: {opset.version}\n")

Use the above steps to inspect the ONNX model's opset version.

    *ONNX Opset Upgrade Tools*:
      - `ONNX opset conversion utility <https://github.com/onnx/onnx/blob/main/onnx/version_converter.py>`_
      - `Optimum-CLI <https://huggingface.co/docs/optimum/en/exporters/onnx/usage_guides/export_a_model>`_

    .. code-block:: python

      # Example steps for opset-21 upgrade of default (onnx) domain
      # Update opsets for other domains as needed for your requirement (or exclude them as suitable).

      model = onnx.load(onnx_path)

      op = onnx.OperatorSetIdProto()
      op.version = 21
      new_opset_imports = [
          helper.make_opsetid("", 21),  # Default domain with opset version 15
          #helper.make_opsetid("ai.onnx.ml", 2)  # ai.onnx.ml domain with opset version 2
          helper.make_opsetid("com.microsoft", 1)
      ]

      updated_quantized_onnx_model = onnx.helper.make_model(model.graph, opset_imports=new_opset_imports)


**Save Quantized Model**

To save a quantized ONNX model with external data, use the following code:

.. code-block:: python

    onnx.save_model(
        updated_quantized_onnx_model,
        output_path,
        save_as_external_data=True,
        location=os.path.basename(output_path) + "_data",
        size_threshold=0,
    )

Deploy Quantized ONNX Model
---------------------------

Inference of the quantized models can be done using tools like `GenAI <https://github.com/microsoft/onnxruntime-genai/>`_, `OnnxRunTime (ORT) <https://onnxruntime.ai//>`_. These APIs can do inference on backends like DML. For details about DirectML deployment, see :ref:`DirectML_Deployment`.
