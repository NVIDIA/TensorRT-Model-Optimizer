
=========================
Quick Start: Distillation
=========================

ModelOpt's :doc:`Distillation <../guides/4_distillation>` is a set of wrappers and utilities
to easily perform Knowledge Distillation among teacher and student models.
Given a pretrained teacher model, Distillation has the potential to train a smaller student model
faster and/or with higher accuracy than the student model could achieve on its own.

This quick-start guide shows the necessary steps to integrate Distillation into your
training pipeline.

Set up your base models
-----------------------

First obtain both a pretrained model to act as the teacher and a (usually smaller) model to serve
as the student.

.. code-block:: python

    from torchvision.models import resnet50, resnet18

    # Define student
    student_model = resnet18()


    # Define callable which returns teacher
    def teacher_factory():
        teacher_model = resnet50()
        teacher_model.load_state_dict(pretrained_weights)
        return teacher_model


Set up the meta model
---------------------

As Knowledge Distillation involves (at least) two models, ModelOpt simplifies the integration
process by wrapping both student and teacher into one meta model.

Please see an example Distillation setup below. This example assumes the outputs
of ``teacher_model`` and ``student_model`` are logits.

.. code-block:: python

    import modelopt.torch.distill as mtd

    distillation_config = {
        "teacher_model": teacher_factory,  # model initializer
        "criterion": mtd.LogitsDistillationLoss(),  # callable receiving student and teacher outputs, in order
        "loss_balancer": mtd.StaticLossBalancer(),  # combines multiple losses; omit if only one distillation loss used
    }

    distillation_model = mtd.convert(student_model, mode=[("kd_loss", distillation_config)])

The ``teacher_model`` can be either a callable which returns an ``nn.Module`` or a tuple of ``(model_cls, args, kwargs)``.
The ``criterion`` is the distillation loss used between student and teacher tensors.
The ``loss_balancer`` determines how the original and distillation losses are combined (if needed).

See :doc:`Distillation <../guides/4_distillation>` for more info.


Distill during training
-----------------------

To Distill from teacher to student, simply use the meta model in the usual training loop, while
also using the meta model's ``.compute_kd_loss()`` method to compute the distillation loss, in addition to
the original user loss.

An example of Distillation training is given below:

.. code-block:: python
    :emphasize-lines: 14

    # Setup the data loaders. As example:
    train_loader = get_train_loader()

    # Define user loss function. As example:
    loss_fn = get_user_loss_fn()

    for input, labels in train_dataloader:
        distillation_model.zero_grad()
        # Forward through the wrapped models
        out = distillation_model(input)
        # Same loss as originally present
        loss = loss_fn(out, labels)
        # Combine distillation and user losses
        loss_total = distillation_model.compute_kd_loss(student_loss=loss)
        loss_total.backward()


.. note::
    `DataParallel <https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html>`_ may
    break ModelOpt's Distillation feature.
    Note that `HuggingFace Trainer <https://huggingface.co/docs/transformers/en/main_classes/trainer>`_
    uses DataParallel by default.


Export trained model
--------------------

The model can easily be reverted to its original class for further use (i.e deployment)
without any ModelOpt modifications attached.

.. code-block:: python

    model = mtd.export(distillation_model)


--------------------------------

**Next steps**
    * Learn more about :doc:`Distillation <../guides/4_distillation>`.
    * See ModelOpt's :doc:`API documentation <../reference/1_modelopt_api>` for detailed
      functionality and usage information.
