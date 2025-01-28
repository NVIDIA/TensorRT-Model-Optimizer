=====================
Quick Start: Sparsity
=====================

Sparsity
--------

ModelOpt's :doc:`sparsity<../guides/5_sparsity>` feature is an effective technique to reduce the
memory footprint of deep learning models and accelerate the inference speed. ModelOpt provides the
easy-to-use API :meth:`mts.sparsify() <modelopt.torch.sparsity.sparsification.sparsify>` to apply
weight sparsity to a given model.
:meth:`mts.sparsify() <modelopt.torch.sparsity.sparsification.sparsify>` supports
`NVIDIA 2:4 Sparsity <https://arxiv.org/abs/2104.08378>`_ sparsity pattern and various sparsification
methods, such as `NVIDIA ASP <https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity>`_
and `SparseGPT <https://arxiv.org/abs/2301.00774>`_.

This guide provides a quick start to apply weight sparsity to a PyTorch model using ModelOpt.

Post-Training Sparsification (PTS) for PyTorch models
-----------------------------------------------------

:meth:`mts.sparsify() <modelopt.torch.sparsity.sparsification.sparsify>` requires the model,
the appropriate sparsity configuration, and a forward loop as inputs.
Here is a quick example of sparsifying a model to 2:4 sparsity pattern with SparseGPT method using
:meth:`mts.sparsify() <modelopt.torch.sparsity.sparsification.sparsify>`.

.. code-block:: python

    import modelopt.torch.sparsity as mts

    # Setup the model
    model = get_model()

    # Setup the data loaders. An example usage:
    data_loader = get_train_dataloader(num_samples=calib_size)

    # Define the sparsity configuration
    sparsity_config = {"data_loader": data_loader, "collect_func": lambda x: x}

    # Sparsify the model and perform calibration (PTS)
    model = mts.sparsify(model, mode="sparsegpt", config=sparsity_config)

.. note::
    `data_loader` is only required in case of data-driven sparsity, e.g., SparseGPT for calibration.
    `sparse_magnitude` does not require `data_loader` as it is purely based on the weights of the model.

.. note::
    `data_loader` and `collect_func` can be substituted with a `forward_loop` that iterates the model through the
    calibration dataset.

Sparsity-aware Training (SAT) for PyTorch models
------------------------------------------------

After sparsifying the model, you can save the checkpoint for the sparsified model and use it for
fine-tuning the sparsified model. Check out the
`GitHub end-to-end example <https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/llm_sparsity>`_
to learn more about SAT.


--------------------------------

**Next Steps**
    * Learn more about sparsity and advanced usage of ModelOpt sparsity in
      :doc:`Sparsity guide <../guides/5_sparsity>`.
    * Checkout out the `end-to-end example on GitHub <https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/llm_sparsity>`_
      for PTS and SAT.
