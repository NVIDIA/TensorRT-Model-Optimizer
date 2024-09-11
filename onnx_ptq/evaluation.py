# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Module to evaluate a device model for the specified task."""

import random
from typing import Any, Final, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from tqdm import tqdm

from modelopt.onnx.utils import validate_onnx
from modelopt.torch._deploy.compilation import compile

deployment = {
    "runtime": "TRT",
    "accelerator": "GPU",
    "version": "8.6",
    "precision": "fp32",
    "onnx_opset": "13",
}

ACCURACY: Final[str] = "accuracy"
RANDOM_SEED: Final[int] = 1996
DEVICE_MODEL: Final[str] = "device_model"
TORCH_MODEL: Final[str] = "torch_model"


def evaluate(
    model: Union[torch.nn.Module, bytes],
    dummy_input: Union[Any, Tuple],
    dataset_path: str,
    evaluation_type: str = ACCURACY,
    batch_size=1,
    evaluate_torch=False,
    num_examples=None,
):
    """Evaluate a model for the given dataset.

    Args:
        model: PyTorch model or compiled onnx model bytes to evaluate.
        dummy_input: Arguments of ``model.forward()``. This is used for exporting and calculating
            inference-based metrics, such as latency/FLOPs. The format of ``dummy_inputs`` follows
            the convention of the ``args`` argument in
            `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_.
            Specifically, ``dummy_input`` can be:

            #. a single argument (``type(dummy_input) != tuple``) corresponding to

               .. code-block:: python

                    model.forward(dummy_input)

            #. a tuple of arguments corresponding to

               .. code-block:: python

                    model.forward(*dummy_input)

            #. a tuple of arguments such that ``type(dummy_input[-1]) == dict`` corresponding to

               .. code-block:: python

                    model.forward(*dummy_input[:-1], **dummy_input[-1])

               .. warning::

                   In this case the model's ``forward()`` method **cannot** contain keyword-only
                   arguments (e.g. ``forward(..., *, kw_only_args)``) or variable keyword arguments
                   (e.g. ``forward(..., **kwargs)``) since these cannot be sorted into positional
                   arguments.

            .. note::

                In order to pass a dict as last non-keyword argument, you need to use a tuple as
                ``dummy_input`` and add an *empty* dict as the last element, e.g.,

                .. code-block:: python

                    dummy_input = (x, {"y": y, "z": z}, {})

                The empty dict at the end will then be interpreted as the keyword args.

            See `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_
            for more info.

            Note that if you provide a ``{arg_name}`` with batch size ``b``, the results will be
            computed based on batch size ``b``.
        dataset_path: Path to the dataset to evaluate on.
            The dataset should be saved in the following format:

            .. code-block:: none

                dataset_path/
                    val/
                        n01440764/
                            ILSVRC2012_val_00000293.JPEG
                            ILSVRC2012_val_00002138.JPEG
                            ...
                        n01728920/
                            ILSVRC2012_val_00001042.JPEG
                            ILSVRC2012_val_00001703.JPEG
                            ...
                        ...

            n01440764, n01728920, etc. are the wordnet ids of the respective class names.
            An example script to preproces this dataset is provided under examples/onnx_ptq/preprocess_imagenet.py.
        evaluation_type: Type of evaluation to perform. Currently only accuracy is supported.
        TODO: Add support for segmentation tasks.
        batch_size: Batch size to use for evaluation. Currenlty only batch_size=1 is supported.
        evaluate_torch: Whether to evaluate the original PyTorch model.
        num_examples: Number of examples to evaluate on. If None, evaluate on the entire dataset.

    Returns:
        The evaluation result.
    """
    # set seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    device_model = None
    if isinstance(model, torch.nn.Module):
        device_model = compile(model, dummy_input, deployment)
    else:
        # TODO: Create device model from ONNX model bytes
        assert validate_onnx(model), "Invalid ONNX model."
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_dataset = ImageNet(root=dataset_path, split="val", transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    if evaluation_type == ACCURACY:
        results = {
            DEVICE_MODEL: {},
            TORCH_MODEL: {},
        }
        results[DEVICE_MODEL][ACCURACY] = evaluate_accuracy(device_model, val_loader, num_examples)
        if evaluate_torch:
            results[TORCH_MODEL][ACCURACY] = evaluate_accuracy(model, val_loader, num_examples)
        return results
    return {}


def evaluate_accuracy(model, val_loader, num_examples, batch_size, topk=(1,)):
    """Evaluate the accuracy of the model on the validation dataset.

    Args:
        model: Model to evaluate.
        val_loader: DataLoader for the validation dataset.
        num_examples: Number of examples to evaluate on. If None, evaluate on the entire dataset.
        batch_size: Batch size to use for evaluation.
        topk: fuction support topk accuracy. Return list of accuracy equal to topk length.
            example of usage `top1, top5 = evaluate_accuracy(..., topk=(1,5))`
            `top1, top5, top10 = evaluate_accuracy(..., topk=(1,5,10))`

    Returns:
        The accuracy of the model on the validation dataset.
    """
    if isinstance(model, torch.nn.Module):
        model.eval()
    total = 0
    corrects = [0] * len(topk)
    for _, (inputs, labels) in tqdm(
        enumerate(val_loader),
        total=num_examples // batch_size if num_examples is not None else len(val_loader),
        desc="Evaluation progress: ",
    ):
        if num_examples is not None and total >= num_examples:
            break
        # Forward pass
        if not isinstance(model, torch.nn.Module):
            inputs = [inputs]
        outputs = model(inputs)

        # Calculate accuracy
        if isinstance(outputs, list):
            outputs = outputs[0]
        else:
            outputs = outputs.data

        labels_size = labels.size(0)
        total += labels_size

        for ind, k in enumerate(topk):
            _, predicted = torch.topk(outputs, k, dim=1)
            corrects[ind] += (predicted == labels.unsqueeze(1)).any(dim=1).sum().item()

    res = [100 * corr / total for corr in corrects]
    return res
