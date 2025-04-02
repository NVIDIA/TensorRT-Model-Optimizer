====================
PyTorch Quantization
====================

Key advantages offered by ModelOpt's PyTorch quantization:

#. Support advanced quantization formats, e.g., Block-wise Int4 and FP8.
#. Native support for LLM models in Hugging Face and NeMo.
#. Advanced Quantization algorithms, e.g., SmoothQuant, AWQ.
#. Deployment support to ONNX and NVIDIA TensorRT.

.. note::

    ModelOpt quantization is fake quantization, which means it only simulates the low-precision computation in PyTorch.
    Real speedup and memory saving should be achieved by exporting the model to deployment frameworks.

.. tip::

    This guide covers the usage of ModelOpt quantization. For details on the quantization formats and recommended use cases,
    please refer to :any:`quantization-formats`.

Apply Post Training Quantization (PTQ)
======================================

PTQ can be achieved with simple calibration on a small set of training or evaluation data (typically 128-512 samples) after converting a regular PyTorch model to a quantized model.
The simplest way to quantize a model using ModelOpt is to use :meth:`mtq.quantize() <modelopt.torch.quantization.model_quant.quantize>`.

:meth:`mtq.quantize` takes a model, a quantization config and a forward loop callable as input.  The quantization config specifies the layers to quantize, their quantization formats as well as the algorithm to use for calibration. Please
refer to :any:`quantization-configs` for the list of quantization configs supported by default. You may also define your own quantization config as
described in :ref:`customizing quantizer config <customize_quantizer_config>`.

ModelOpt supports algorithms such as AWQ, SmoothQuant, SVDQuant or max for calibration. Please refer to :meth:`mtq.calibrate <modelopt.torch.quantization.model_quant.calibrate>`
for more details.

The forward loop is used to pass data through the model in-order to collect statistics for calibration.
It should wrap around the calibration dataloader and the model.

Here is an example of performing PTQ using ModelOpt:

.. code-block:: python

    import modelopt.torch.quantization as mtq

    # Setup the model
    model = get_model()

    # Select quantization config
    config = mtq.INT8_SMOOTHQUANT_CFG

    # Quantization need calibration data. Setup calibration data loader
    # An example of creating a calibration data loader looks like the following:
    data_loader = get_dataloader(num_samples=calib_size)


    # Define forward_loop. Please wrap the data loader in the forward_loop
    def forward_loop(model):
        for batch in data_loader:
            model(batch)


    # Quantize the model and perform calibration (PTQ)
    model = mtq.quantize(model, config, forward_loop)

To verify that the quantizer nodes are placed correctly in the model, let's print the quantized model summary as show below:

.. code-block:: python

    # Print quantization summary after successfully quantizing the model with mtq.quantize
    # This will show the quantizers inserted in the model and their configurations
    mtq.print_quant_summary(model)


After PTQ, the model can be exported to ONNX with the normal PyTorch ONNX export flow.

.. code-block:: python

    torch.onnx.export(model, sample_input, onnx_file)

ModelOpt also supports direct export of Huggingface or Nemo LLM models to TensorRT-LLM for deployment.
Please see :doc:`TensorRT-LLM Deployment <../deployment/1_tensorrt_llm>` for more details.

Compressing model weights after quantization
============================================

ModelOpt provides a API :meth:`mtq.compress() <modelopt.torch.quantization.compress>` to compress the model weights after quantization.
This API can be used to reduce the memory footprint of the quantized model for future evaluation or fine-tuning such as QLoRA. Note that
this API only supports selected quantization formats.

After PTQ, the model can be compressed with the following code:

.. code-block:: python

    # Compress the model
    mtq.compress(model)


Quantization-aware Training (QAT)
=================================

QAT is the technique of fine-tuning a quantized model to recover model quality degradation due to quantization.
While QAT requires much more compute resources than PTQ, it is highly effective in recovering model quality.

A model quantized using  :meth:`mtq.quantize() <modelopt.torch.quantization.model_quant.quantize>` could be directly fine-tuned with QAT.
Typically during QAT, the quantizer states are frozen and the model weights are fine-tuned.

Here is an example of performing QAT:

.. code-block:: python

    import modelopt.torch.quantization as mtq

    # Select quantization config
    config = mtq.INT8_DEFAULT_CFG


    # Define forward loop for calibration
    def forward_loop(model):
        for data in calib_set:
            model(data)


    # QAT after replacement of regular modules to quantized modules
    model = mtq.quantize(model, config, forward_loop)

    # Fine-tune with original training pipeline
    # Adjust learning rate and training duration
    train(model, train_loader, optimizer, scheduler, ...)

.. tip::

    We recommend QAT for 10% of the original training epochs. For LLMs, we find that QAT fine-tuning for even
    less than 1% of the original pre-training duration is often sufficient to recover the model quality.

Storing and restoring quantized model
======================================

The model weights and quantizer states need to saved for future use or to resume training.
Please see :ref:`saving and restoring of ModelOpt-modified models <save-restore>` to learn
how to save and restore the quantized model.


Optimal Partial Quantization using AutoQuantize(``auto_quantize``)
===================================================================

:meth:`auto_quantize <modelopt.torch.quantization.model_quant.auto_quantize>` or ``AutoQuantize`` is a PTQ algorithm from ModelOpt which
quantizes a model by searching for the best quantization format per-layer
while meeting the performance constraint specified by the user. ``AutoQuantize`` enables to trade-off model accuracy
for performance. Please see :meth:`auto_quantize <modelopt.torch.quantization.model_quant.auto_quantize>` for more details
on the API usage.

Currently ``AutoQuantize`` supports only ``effective_bits`` as the performance constraint (for both weight-only
quantization and weight & activation quantization). ``effective_bits`` constraint specifies the effective number of bits for the quantized model.

You may specify a ``effective_bits`` constraint such as 8.8 for partial quantization with :attr:`FP8_DEFAULT_CFG`.
``AutoQuantize`` will skip quantizing the most quantization sensitive layers so that the final partially quantized model's
effective bits is 8.8. This model will have a better accuracy than the model quantized with default configuration since quantization was
skipped for some layers which are highly sensitive to quantization.

Here is how to perform ``AutoQuantize``:

.. code::

    import modelopt.torch.quantization as mtq
    import modelopt.torch.opt as mto

    # Define the model & calibration dataloader
    model = ...
    calib_dataloader = ...

    # Define forward_step function.
    # forward_step should take the model and data as input and return the output
    def forward_step(model, data):
        output =  model(data)
        return output

    # Define loss function which takes the model output and data as input and returns the loss
    def loss_func(output, data):
        loss = ...
        return loss


    # Perform AutoQuantize
    model, search_state_dict = mtq.auto_quantize(
        model,
        constraints = {"effective_bits": 4.8},
        # supported quantization formats are listed in `modelopt.torch.quantization.config.choices`
        quantization_formats = ["NVFP4_DEFAULT_CFG", "FP8_DEFAULT_CFG", None]
        data_loader = calib_dataloader,
        forward_step=forward_step,
        loss_func=loss_func,
        ...
        )

    # Save the searched model for future use
    mto.save(model, "auto_quantize_model.pt")


Advanced Topics
===============

TensorQuantizer
---------------

Under the hood, ModelOpt :meth:`mtq.quantize() <modelopt.torch.quantization.model_quant.quantize>` inserts
:class:`TensorQuantizer <modelopt.torch.quantization.nn.modules.tensor_quantizer.TensorQuantizer>`
(quantizer modules) into the model layers like linear layer, conv layer etc. and patches their forward method to perform quantization.

The quantization parameters are as described in :class:`QuantizerAttributeConfig <modelopt.torch.quantization.config.QuantizerAttributeConfig>`.
They can be set at initialization by passing :class:`QuantizerAttributeConfig <modelopt.torch.quantization.config.QuantizerAttributeConfig>`
or later by calling  :meth:`TensorQuantizer.set_from_attribute_config() <modelopt.torch.quantization.nn.modules.tensor_quantizer.TensorQuantizer.set_from_attribute_config>`.
If the quantization parameters are not set explicitly, the quantizer will use the default values.

Here is an example of creating a quantizer module:

.. code-block:: python

    from modelopt.torch.quantization.config import QuantizerAttributeConfig
    from modelopt.torch.quantization.nn import TensorQuantizer

    # Create quantizer module with default quantization parameters
    quantizer = TensorQuantizer()

    quant_x = quantizer(x)  # Quantize input x

    # Create quantizer module with custom quantization parameters
    # Example setting for INT4 block-wise quantization
    quantizer_custom = TensorQuantizer(QuantizerAttributeConfig(num_bits=4, block_sizes={-1: 128}))

    # Quantize input with custom quantization parameters
    quant_x = quantizer_custom(x)  # Quantize input x


.. _customize_quantizer_config:

Customize quantizer config
--------------------------

ModelOpt inserts input quantizer, weight quantizer and output quantizer into common layers, but by default disables the output quantizer.
Expert users who want to customize the default quantizer configuration can update the ``config`` dictionary provided to ``mtq.quantize`` using wildcard or filter function match.

Here is an example of specifying a custom quantizer configuration to ``mtq.quantize``:

.. code-block:: python

    # Select quantization config
    config = mtq.INT8_DEFAULT_CFG.copy()
    config["quant_cfg"]["*.bmm.output_quantizer"] = {
        "enable": True
    }  # Enable output quantizer for bmm layer

    # Perform PTQ/QAT;
    model = mtq.quantize(model, config, forward_loop)


.. _custom_quantied_module:

Custom quantized module and quantizer placement
-----------------------------------------------

``modelopt.torch.quantization`` has a default set of quantized modules (see :mod:`modelopt.torch.quantization.nn.modules <modelopt.torch.quantization.nn.modules>` for a detailed list) and quantizer placement rules (input, output and weight quantizers). However, there might be cases where you want to define a custom quantized module and/or customize the quantizer placement.

ModelOpt provides a way to define custom quantized modules and register them with the quantization framework. This allows you to:

#. Handle unsupported modules, e.g., a subclassed Linear layer that require quantization.
#. Customize the quantizer placement, e.g., placing the quantizer in special places like the KV Cache of an Attention layer.

Here is an example of defining a custom quantized LayerNorm module:

.. code-block:: python

    from modelopt.torch.quantization.nn import TensorQuantizer


    class QuantLayerNorm(nn.LayerNorm):
        def __init__(self, normalized_shape):
            super().__init__(normalized_shape)
            self._setup()

        def _setup(self):
            # Method to setup the quantizers
            self.input_quantizer = TensorQuantizer()
            self.weight_quantizer = TensorQuantizer()

        def forward(self, input):
            # You can customize the quantizer placement anywhere in the forward method
            input = self.input_quantizer(input)
            weight = self.weight_quantizer(self.weight)
            return F.layer_norm(input, self.normalized_shape, weight, self.bias, self.eps)

After defining the custom quantized module, you need to register this module so ``mtq.quantize`` API will automatically replace the original module with the quantized version.
Note that the custom ``QuantLayerNorm`` must have a ``_setup`` method which instantiates the quantizer attributes that are called in the forward method.
Here is the code to register the custom quantized module:

.. code-block:: python

    import modelopt.torch.quantization as mtq

    # Register the custom quantized module
    mtq.register(original_cls=nn.LayerNorm, quantized_cls=QuantLayerNorm)

    # Perform PTQ
    # nn.LayerNorm modules in the model will be replaced with the QuantLayerNorm module
    model = mtq.quantize(model, config, forward_loop)

The quantization config might need to be customized if you define a custom quantized module. Please see
:ref:`customizing quantizer config <customize_quantizer_config>` for more details.

Fast evaluation
---------------

Weight folding avoids repeated quantization of weights during each inference forward pass and speedup evaluation. This can be done with the following code:

.. code-block:: python

    # Fold quantizer together with weight tensor
    mtq.fold_weight(quantized_model)

    # Run model evaluation
    user_evaluate_func(quantized_model)

.. note::

    After weight folding, the model can no longer be exported to ONNX or fine-tuned with QAT.

Migrate from pytorch_quantization
=================================

ModelOpt PyTorch quantization is refactored from and extends upon
`pytorch_quantization <https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/index.html>`_.

Previous users of ``pytorch_quantization`` can simply migrate to ``modelopt.torch.quantization`` by
replacing the import statements.
