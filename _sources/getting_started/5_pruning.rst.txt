====================
Quick Start: Pruning
====================

.. tip::

    Checkout `ResNet20 on CIFAR-10 Notebook <https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/pruning/cifar_resnet.ipynb>`_ and
    `Llama 3.1 NeMo Minitron Pruning <https://github.com/NVIDIA/NeMo/tree/main/tutorials/llm/llama/pruning-distillation>`_
    for an end-to-end example of pruning.

ModelOpt's :doc:`Pruning<../guides/2_pruning>` library provides many light-weight pruning methods
like Minitron, FastNAS, and GradNAS that can be run on any user-provided model.
Check out this doc for more details on these pruning methods and recommendations on when what pruning method to use.

Pruning a pretrained model involves three steps which are setting up your model, setting up the
search, and finally running the search (pruning).

Set up your model
-----------------

To set up your model for pruning, simply initialize the model and load a pre-trained checkpoint.
Alternatively, you can also train the model from scratch.


Set up the search
-----------------

Setting up search for pruning involves setting up the training and validation data loaders, and optionally defining a
score function (FastNAS) or loss function (GradNAS) and specifying the desired pruning constraints.
The most common score function is the validation accuracy of the model and is used to rank the sub-nets sampled from
the search space. Loss function is used to run some forward and backward passes on the model to get the gradients
of the model.

Please see an example of FastNAS pruning set up below.

.. code-block:: python

    import modelopt.torch.prune as mtp
    from torchvision.models import resnet50

    # User-defined model
    model = resnet50()

    # Load pretrained weights here
    model.load_state_dict(pretrained_weights)


    # Wrap your original validation function to only take the model as input.
    # This function acts as the score function to rank models.
    def score_func(model):
        return validate(model, val_loader, ...)


    # Define a dummy input with similar shape as that of your input data
    dummy_input = torch.randn(1, 3, 224, 244)

    # Prune the model to at most 60% of the original FLOPs
    prune_constraints = {"flops": "60%"}

Prune the model
---------------

To prune your model, you can simply call the :mod:`mtp.prune<.modelopt.torch.prune.pruning.prune>`
API and save the pruned model using :meth:`mto.save<.modelopt.torch.opt.save>`.

An example of FastNAS pruning is given below:

.. code-block:: python

    import modelopt.torch.opt as mto
    import modelopt.torch.prune as mtp

    # prune_res (dict) contains state_dict / stats of the pruner/searcher.
    pruned_model, prune_res = mtp.prune(
        model=model,
        mode="fastnas",
        constraints=prune_constraints,
        dummy_input=dummy_input,
        config={
            "data_loader": train_loader,  # training data is used for calibrating BN layers
            "score_func": score_func,  # validation score is used to rank the subnets
            # checkpoint to store the search state and resume or re-run the search with different constraint
            "checkpoint": "modelopt_fastnas_search_checkpoint.pth",
        },
    )

    # Save the pruned model.
    mto.save(pruned_model, "modelopt_pruned_model.pth")

.. note::
    Fine-tuning is required after pruning to recover the accuracy.
    Please refer to :ref:`pruning fine-tuning<pruning_fine_tuning>` for mode details.


--------------------------------

**Next steps**
    * Learn more about :doc:`Pruning<../guides/2_pruning>` API and supported algorithms / models.
    * Learn more about :doc:`NAS<../guides/3_nas>`, which is a generalization of pruning.
    * See ModelOpt :doc:`API documentation<../reference/1_modelopt_api>` for detailed functionality and
      usage information.
