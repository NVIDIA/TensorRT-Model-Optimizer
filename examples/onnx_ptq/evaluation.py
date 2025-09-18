# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module to evaluate a device model for the specified task."""

import random
from typing import Any, Final

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from tqdm import tqdm

from modelopt.onnx.utils import validate_onnx
from modelopt.torch._deploy.compilation import compile

deployment = {
    "runtime": "TRT",
    "accelerator": "GPU",
    "precision": "stronglyTyped",
    "onnx_opset": "21",
}

ACCURACY: Final[str] = "accuracy"
DEVICE_MODEL: Final[str] = "device_model"
TORCH_MODEL: Final[str] = "torch_model"


def evaluate(
    model: torch.nn.Module | bytes,
    dummy_input: Any | tuple,
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
            An example script to preprocess this dataset is provided under examples/onnx_ptq/preprocess_imagenet.py.
        evaluation_type: Type of evaluation to perform. Currently only accuracy is supported.
        TODO: Add support for segmentation tasks.
        batch_size: Batch size to use for evaluation. Currently only batch_size=1 is supported.
        evaluate_torch: Whether to evaluate the original PyTorch model.
        num_examples: Number of examples to evaluate on. If None, evaluate on the entire dataset.

    Returns:
        The evaluation result.
    """

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
        results[DEVICE_MODEL][ACCURACY] = evaluate_accuracy(
            device_model, val_loader, num_examples, batch_size
        )
        if evaluate_torch:
            results[TORCH_MODEL][ACCURACY] = evaluate_accuracy(
                model, val_loader, num_examples, batch_size
            )
        return results
    return {}


def evaluate_accuracy(model, val_loader, num_examples, batch_size, topk=(1,), random_seed=None):
    """Evaluate the accuracy of the model on the validation dataset.

    Args:
        model: Model to evaluate.
        val_loader: DataLoader for the validation dataset.
        num_examples: Number of examples to evaluate on. If None, evaluate on the entire dataset.
        batch_size: Batch size to use for evaluation.
        topk: function support topk accuracy. Return list of accuracy equal to topk length.
            example of usage `top1, top5 = evaluate_accuracy(..., topk=(1,5))`
            `top1, top5, top10 = evaluate_accuracy(..., topk=(1,5,10))`
        random_seed: Random seed to use for evaluation.

    Returns:
        The accuracy of the model on the validation dataset.
    """

    if random_seed:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        random.seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
        outputs = outputs[0] if isinstance(outputs, list) else outputs.data

        labels_size = labels.size(0)
        total += labels_size

        labels = labels.to(outputs.device)

        for ind, k in enumerate(topk):
            _, predicted = torch.topk(outputs, k, dim=1)
            corrects[ind] += (predicted == labels.unsqueeze(1)).any(dim=1).sum().item()

    res = [100 * corr / total for corr in corrects]
    return res
