"""Multi-node PTQ (Post-Training Quantization) with FSDP2 support."""

import argparse
import copy
import json
import os
import random
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from example_utils import apply_kv_cache_quant, get_tokenizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedTokenizerFast

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.export import get_model_type
from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
from modelopt.torch.export.unified_export_hf import _export_hf_checkpoint
from modelopt.torch.quantization.config import need_calibration
from modelopt.torch.quantization.utils import patch_fsdp_mp_dtypes
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader, get_supported_datasets

# Constants
RAND_SEED = 1234

QUANT_CFG_CHOICES: dict[str, dict[str, Any]] = {
    "int8_wo": mtq.INT8_WEIGHT_ONLY_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    "nvfp4_awq": mtq.NVFP4_AWQ_LITE_CFG,
    "nvfp4_mlp_only": mtq.NVFP4_MLP_ONLY_CFG,
}

KV_QUANT_CFG_CHOICES = {
    "none": "none",
    "fp8": "FP8_KV_CFG",
    "nvfp4": "NVFP4_KV_CFG",
    "nvfp4_affine": "NVFP4_AFFINE_KV_CFG",
}


# Enable HuggingFace checkpointing
mto.enable_huggingface_checkpointing()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-node post-training quantization with FSDP2")

    parser.add_argument(
        "--pyt_ckpt_path",
        required=True,
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--qformat",
        default="fp8",
        choices=QUANT_CFG_CHOICES.keys(),
        help="Quantization format",
    )
    parser.add_argument(
        "--kv_cache_qformat",
        default="fp8",
        choices=list(KV_QUANT_CFG_CHOICES.keys()),
        help="KV cache quantization format",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for calibration",
    )
    parser.add_argument(
        "--calib_size",
        type=str,
        default="512",
        help="Comma-separated list of calibration sizes per dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help=f"Comma-separated list of datasets. Choices: {get_supported_datasets()}",
    )
    parser.add_argument(
        "--export_path",
        default="exported_model",
        help="Directory to export the quantized model",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code for HuggingFace models",
    )
    parser.add_argument("--awq_block_size", default=0, type=int)

    args = parser.parse_args()

    # Parse comma-separated lists
    args.dataset = args.dataset.split(",") if args.dataset else None
    args.calib_size = [int(x) for x in args.calib_size.split(",")]

    return args


def load_and_prepare_model(
    model_path: str,
    accelerator: Accelerator,
    trust_remote_code: bool = False,
) -> tuple[nn.Module, str, list[str]]:
    """Load model and prepare it for FSDP2 distributed execution.

    Args:
        model_path: Path to the HuggingFace model
        accelerator: Accelerate Accelerator instance
        trust_remote_code: Whether to trust remote code

    Returns:
        Tuple of (prepared_model, model_type)
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    model_type = get_model_type(model)
    # Need the original architectures for export
    # FSDP prefix is added to the architectures for FSDP2 wrapped models
    original_architectures = model.config.architectures

    # FSDP2 requires an optimizer to be prepared together with the model
    dummy_optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    model, _ = accelerator.prepare(model, dummy_optimizer)

    return model, model_type, original_architectures


def create_calibration_dataloader(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    dataset_names: list[str] | None,
    calib_sizes: list[int],
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """Create calibration dataloader from dataset.

    Args:
        tokenizer: HuggingFace tokenizer
        dataset_names: List of dataset names (defaults to cnn_dailymail)
        calib_sizes: Number of samples for each dataset
        batch_size: Batch size for calibration

    Returns:
        DataLoader for calibration
    """
    if dataset_names is None:
        dataset_names = ["cnn_dailymail"]
        warnings.warn("No dataset specified. Defaulting to cnn_dailymail.")

    return get_dataset_dataloader(
        dataset_name=dataset_names,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_samples=calib_sizes,
        device=None,  # Keep data on CPU, calibration loop handles device transfer
        include_labels=False,
    )


def get_quantization_config(
    qformat: str,
    kv_cache_qformat: str,
    model_type: str,
    awq_block_size: int | None = None,
) -> dict[str, Any]:
    """Build quantization configuration.

    Args:
        qformat: Quantization format
        kv_cache_qformat: KV cache quantization format
        model_type: Model type (e.g., 'llama', 'gemma')
        awq_block_size: Optional AWQ block size

    Returns:
        Quantization configuration dictionary
    """
    quant_cfg = copy.deepcopy(QUANT_CFG_CHOICES[qformat])

    # Configure AWQ if needed
    if "awq" in qformat:
        weight_quantizer = quant_cfg["quant_cfg"]["*weight_quantizer"]
        if isinstance(weight_quantizer, list):
            weight_quantizer = weight_quantizer[0]

        if awq_block_size:
            weight_quantizer["block_sizes"][-1] = awq_block_size

        # Coarser search for certain models to avoid overflow
        if qformat == "w4a8_awq" and model_type in ["gemma", "mpt"]:
            quant_cfg["algorithm"] = {"method": "awq_lite", "alpha_step": 1}

    # Configure KV cache quantization
    enable_kv_quant = kv_cache_qformat != "none"
    print(f"{'Enable' if enable_kv_quant else 'Disable'} KV cache quantization")

    if enable_kv_quant:
        kv_cfg = getattr(mtq, KV_QUANT_CFG_CHOICES[kv_cache_qformat])["quant_cfg"]
        quant_cfg = apply_kv_cache_quant(quant_cfg, kv_cfg)

    # Model-specific adjustments
    if model_type == "gemma" and "int8_sq" in qformat:
        quant_cfg["algorithm"] = {"method": "smoothquant", "alpha": 0.5}

    return quant_cfg


def create_fsdp2_calibration_loop(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
):
    """Create calibration loop compatible with FSDP2.

    For FSDP2, we need to use the outer FSDP-wrapped model instead of
    the parameter passed by mtq.quantize to properly handle DTensor.

    Args:
        model: FSDP2-wrapped model
        dataloader: Calibration dataloader
        accelerator: Accelerator instance for device management

    Returns:
        Calibration function compatible with mtq.quantize
    """

    def calibrate(unwrapped_model):
        """Calibration loop that uses the FSDP-wrapped model."""
        for batch in tqdm(dataloader, desc="Calibrating"):
            if isinstance(batch, dict):
                batch = {
                    k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            # Use outer model (FSDP-wrapped), not the parameter
            # Important: We should forward pass using the unwrapped model
            # mtq.quantize will unwrap the model & pass to the forward_loop
            model(**batch)

    return calibrate


def export_model(
    model: nn.Module,
    accelerator: Accelerator,
    export_path: str | Path,
    architectures: list[str],
):
    """Export quantized model to HuggingFace format.

    Args:
        model: Quantized model
        accelerator: Accelerator instance for state dict gathering
        export_path: Directory to export model to
    """
    export_dir = Path(export_path)
    export_dir.mkdir(parents=True, exist_ok=True)

    post_state_dict, hf_quant_config = _export_hf_checkpoint(
        model, torch.bfloat16, accelerator=accelerator
    )

    if accelerator.is_main_process:
        # Save hf_quant_config.json for backward compatibility
        with open(f"{export_dir}/hf_quant_config.json", "w") as file:
            json.dump(hf_quant_config, file, indent=4)

        hf_quant_config = convert_hf_quant_config_format(hf_quant_config)

        # Save model
        model.save_pretrained(export_dir, state_dict=post_state_dict, save_modelopt_state=False)

        original_config = f"{export_dir}/config.json"
        config_data = {}

        with open(original_config) as file:
            config_data = json.load(file)

        config_data["quantization_config"] = hf_quant_config
        # Update config architectures to use original architectures that does not have FSDP prefix
        config_data["architectures"] = architectures

        with open(original_config, "w") as file:
            json.dump(config_data, file, indent=4)


def main(args):
    """Main quantization workflow."""
    # Validate GPU availability
    if not torch.cuda.is_available():
        raise OSError("GPU is required for quantization.")

    # Validate quantization format
    if args.qformat not in QUANT_CFG_CHOICES:
        raise ValueError(
            f"Quantization format {args.qformat} not supported. Choose from: {QUANT_CFG_CHOICES.keys()}"
        )

    # Set random seeds
    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)

    # Initialize accelerator
    accelerator = Accelerator()

    print(f"Rank: {os.environ.get('RANK', 'Not set')}")
    print(f"World Size: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"Local Rank: {os.environ.get('LOCAL_RANK', 'Not set')}")

    # Load tokenizer
    tokenizer = get_tokenizer(args.pyt_ckpt_path, trust_remote_code=args.trust_remote_code)
    tokenizer.padding_side = "left"  # Left padding for better calibration

    # Create calibration dataloader
    calib_dataloader = create_calibration_dataloader(
        tokenizer=tokenizer,
        dataset_names=args.dataset,
        calib_sizes=args.calib_size,
        batch_size=args.batch_size,
    )

    # Load and prepare model
    model, model_type, original_architectures = load_and_prepare_model(
        model_path=args.pyt_ckpt_path,
        accelerator=accelerator,
        trust_remote_code=args.trust_remote_code,
    )

    # Build quantization config
    quant_cfg = get_quantization_config(
        qformat=args.qformat,
        kv_cache_qformat=args.kv_cache_qformat,
        model_type=model_type,
        awq_block_size=args.awq_block_size,
    )

    # Quantize the model
    if accelerator.is_main_process:
        print("Starting quantization...")

    start_time = time.time()

    if need_calibration(quant_cfg):
        calibrate_fn = create_fsdp2_calibration_loop(model, calib_dataloader, accelerator)
    else:
        calibrate_fn = None
        warnings.warn("Dynamic quantization. Calibration skipped.")

    with torch.no_grad():
        model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_fn)

    elapsed = time.time() - start_time

    if accelerator.is_main_process:
        print(f"Quantization completed in {elapsed:.2f}s")
        mtq.print_quant_summary(model)

    start_time = time.time()
    export_model(model, accelerator, args.export_path, original_architectures)
    elapsed = time.time() - start_time

    if accelerator.is_main_process:
        # Export the model
        print(f"Export completed in {elapsed:.2f}s")
        print(f"Model exported to {args.export_path}")

    print("Unpatching FSDP2 MP dtypes")


if __name__ == "__main__":
    args = parse_args()
    # This context manager can be removed once the update to FSDP2 function is reflected in torch
    with patch_fsdp_mp_dtypes():
        main(args)
