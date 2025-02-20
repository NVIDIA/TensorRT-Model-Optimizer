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

"""Utility functions for getting samples and forward loop function for different datasets."""

import math
from typing import TYPE_CHECKING, Callable, Optional, Union
from warnings import warn

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# Use dict to store the config for each dataset.
# If we want to export more options to user like target languages, we need more standardized approach like dataclass.
SUPPORTED_DATASET_CONFIG = {
    "cnn_dailymail": {
        "config": {"path": "cnn_dailymail", "name": "3.0.0"},
        "target": "article",
    },
    "pile": {
        "config": {"path": "monology/pile-uncopyrighted"},
        "target": "text",
    },
    "pg19": {
        "config": {"path": "pg19"},
        "target": "text",
    },
    "wikipedia": {
        "config": {"path": "wikipedia", "name": "20220301.en"},
        "target": "text",
    },
    "c4": {
        "config": {"path": "c4", "name": "en"},
        "target": "text",
    },
}

__all__ = [
    "create_forward_loop",
    "get_dataset_dataloader",
    "get_max_batch_size",
    "get_supported_datasets",
]


def _get_dataset_samples(dataset_name: str, num_samples: int) -> list[str]:
    """Load a portion of train dataset with the dataset name and a given size.

    Args:
        dataset_name: Name of the dataset to load.
        num_samples: Number of samples to load from the dataset.

    Returns:
        Samples: The list of samples.
    """
    # Load the dataset
    if dataset_name in SUPPORTED_DATASET_CONFIG:
        from datasets import load_dataset

        # Use streaming can reduce the downloading time for large datasets
        dataset = load_dataset(
            split="train",
            streaming=True,
            **SUPPORTED_DATASET_CONFIG[dataset_name]["config"],
        )
    else:
        raise NotImplementedError(
            f"dataset {dataset_name} is not supported. Please use one of the following:"
            f" {get_supported_datasets()}."
        )

    # Access only the required samples
    samples = []
    target_key = SUPPORTED_DATASET_CONFIG[dataset_name]["target"]
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        samples.append(sample[target_key])

    return samples


class _CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(next(iter(self.encodings.values())))


def get_dataset_dataloader(
    dataset_name: str = "cnn_dailymail",
    tokenizer: Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"] = None,
    batch_size: int = 1,
    num_samples: int = 512,
    max_sample_length: int = 512,
    device: Optional[str] = None,
    include_labels: bool = False,
) -> DataLoader:
    """Get a dataloader with the dataset name and toknizer of the target model.

    Args:
        dataset_name: Name of the dataset to load.
        tokenizer: Instancne of Hugginface tokenizer.
        batch_size: Batch size of the returned dataloader.
        num_samples: Number of samples from the dataset.
        max_sample_length: Maximum length of a sample.
        device: Target device for the returned dataloader.
        include_labels: Whether to include labels in the dataloader.

    Returns:
        A instance of dataloader.
    """
    assert tokenizer is not None, "Please provide a tokenizer."

    if tokenizer.padding_side != "left":
        warn(
            "Tokenizer with the right padding_side may impact calibration accuracy. Recommend set to left"
        )

    num_samples = math.ceil(num_samples / batch_size) * batch_size

    dataset = _get_dataset_samples(dataset_name, num_samples=num_samples)

    batch_encoded = tokenizer.batch_encode_plus(
        dataset,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_sample_length,
    )
    if device:
        batch_encoded = batch_encoded.to(device)

    if include_labels:
        # Labels are needed when backward is called in the model.
        # The labels should be a shifted version of the input_ids.
        # However, we should not shift the input_ids here since the labels are shifted by
        # Huggingface models during loss calculation as shown here -
        # https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/llama/modeling_llama.py#L1093-L1095
        batch_encoded["labels"] = torch.where(
            batch_encoded["attention_mask"] > 0.5, batch_encoded["input_ids"], -100
        )
        tokenized_dataset = _CustomDataset(batch_encoded)
    else:
        # For backward compatibility, if labels are not needed, we only return the input_ids.
        tokenized_dataset = _CustomDataset({"input_ids": batch_encoded["input_ids"]})

    calib_dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)

    return calib_dataloader


def get_supported_datasets() -> list[str]:
    """Retrieves a list of datasets supported.

    Returns:
        A list of strings, where each string is the name of a supported dataset.

    Example usage:

        .. code-block:: python

            from modelopt.torch.utils import get_supported_datasets

            print("Supported datasets:", get_supported_datasets())
    """
    return list(SUPPORTED_DATASET_CONFIG.keys())


def get_max_batch_size(
    model: torch.nn.Module,
    max_sample_length: int = 512,
    sample_memory_usage_ratio: float = 1.0,
):
    """Get the maximum batch size that can be used for the model."""

    def _get_free_gpu_mem():
        min_gpu_free_mem = torch.cuda.get_device_properties(0).total_memory
        max_allocated_mem = 0
        for device in range(torch.cuda.device_count()):
            free_mem = torch.cuda.mem_get_info(device)[0]
            if free_mem < min_gpu_free_mem:
                min_gpu_free_mem = free_mem
                max_allocated_mem = torch.cuda.max_memory_allocated(device)
        return min_gpu_free_mem, max_allocated_mem

    torch.cuda.empty_cache()

    free_mem_before, max_allocated_before = _get_free_gpu_mem()
    is_enc_dec = model_type_is_enc_dec(model)
    infer_method = model.generate if is_enc_dec else model.forward
    # Calculate single batch inference with dummy input.
    with torch.no_grad():
        infer_method(torch.ones([1, max_sample_length]).int().to(model.device) * 100)
    free_mem_after, max_allocated_after = _get_free_gpu_mem()

    mem_diff_per_data_batch = (
        max(
            (free_mem_before - free_mem_after),
            (max_allocated_after - max_allocated_before),
        )
        * sample_memory_usage_ratio
    )
    target_data_batch = max(int(free_mem_before / mem_diff_per_data_batch), 1)

    # For some models on multi GPU, we observe the memory per batch is not a constant.
    # So we just test the target batch size and make sure we do not go OOM.
    while target_data_batch > 1:
        with torch.no_grad():
            try:
                infer_method(
                    torch.ones([target_data_batch, max_sample_length]).int().to(model.device) * 100
                )
                break
            except torch.cuda.OutOfMemoryError:
                target_data_batch = target_data_batch // 2

    # Regulate the data batch target to be 1, 2, 4, 8, 12, ..., capped at 64
    if target_data_batch < 2:
        return 1
    elif target_data_batch < 4:
        return 2
    elif target_data_batch < 64:
        return target_data_batch // 4 * 4
    else:
        return 64


def create_forward_loop(
    model: torch.nn.Module = None,
    dataset_name: str = "cnn_dailymail",
    tokenizer: Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"] = None,
    batch_size: int = 1,
    num_samples: int = 512,
    max_sample_length: int = 512,
    device: Optional[str] = None,
    include_labels: bool = False,
    dataloader: DataLoader = None,
) -> Callable:
    """Creates and returns a forward loop function configured for a specific model, dataset, and tokenizer.

    This function initializes a forward loop function tailored to process batches of data from the specified dataset
    using the given model and tokenizer. The forward loop function, when called, iterates over the dataset, applies the
    tokenizer to prepare the input data, feeds it into the model, and returns the model's predictions.

    Args:
        model: The PyTorch model for inference.
        dataset_name: The name of the dataset to be used. Must be one of the datasets in get_supported_datasets().
        tokenizer: The tokenizer used to preprocess text data into a format suitable
            for the model.
        batch_size: Batch size of the returned dataloader. If 0 is provided, we auto determine the batch_size.
        num_samples: Number of samples from the dataset.
        max_sample_length: Maximum length of a sample.
        device: Target device for the returned dataloader.
        include_labels: Whether to include labels in the dataloader.
        dataloader: If provided, use the provided dataloader instead.

    Example usage for quantization:

    .. code-block:: python

        import modelopt.torch.quantization as mtq
        from modelopt.torch.utils import create_forward_loop

        # Initialize model and tokenizer
        # ...

        # Create forward loop for calibration
        forward_loop = create_forward_loop(
            model=model, dataset_name="cnn_dailymail", tokenizer=tokenizer
        )

        # Quantize the model with the calibration dataset
        mtq.quantize(model, quant_cfg, forward_loop=forward_loop)

    Returns:
        A forward loop function that can be called with no arguments. When called, this function iterates over
            the dataset specified by `dataset_name`.
    """
    if dataloader is None:
        if batch_size == 0:
            batch_size = get_max_batch_size(model, max_sample_length)
            print(f"Update calib batch {batch_size}")

        dataloader = get_dataset_dataloader(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_samples=num_samples,
            max_sample_length=max_sample_length,
            device=device,
            include_labels=include_labels,
        )

    def forward_loop(model):
        with torch.no_grad():
            safe_factor = 1  # Worst-case split factor needed so far
            is_enc_dec = model_type_is_enc_dec(model)
            infer_method = model.generate if is_enc_dec else model.forward

            def run_splits(data, factor: int):
                batch_size = data[list(data.keys())[0]].shape[0]
                l_of_indices = torch.chunk(torch.arange(batch_size), chunks=factor)
                for idxs in l_of_indices:
                    infer_method(**{k: v[idxs] for k, v in data.items()})

            for idx, data in enumerate(tqdm(dataloader)):
                batch_size = data[list(data.keys())[0]].shape[0]
                factor = min(safe_factor, batch_size)
                while True:  # until inference is successful
                    try:
                        if factor == 1:
                            infer_method(**data)
                        else:
                            run_splits(data, factor)
                        break  # Inference successful, break out of the loop
                    except torch.cuda.OutOfMemoryError:
                        warn(
                            f"torch.OutOfMemoryError with batch size {batch_size} split into {safe_factor}. "
                            "Trying higher splits..."
                        )
                        factor *= 2  # Increase the split factor
                        if factor >= batch_size:
                            raise RuntimeError(
                                f"Forward loop failed on batch {idx}: "
                                " even processing one example at a time causes torch.OutOfMemoryError."
                            )
                safe_factor = max(safe_factor, factor)

    return forward_loop


def model_type_is_enc_dec(model):
    enc_dec_model_list = ["t5", "bart"]
    return any(model_name in model.__class__.__name__.lower() for model_name in enc_dec_model_list)
