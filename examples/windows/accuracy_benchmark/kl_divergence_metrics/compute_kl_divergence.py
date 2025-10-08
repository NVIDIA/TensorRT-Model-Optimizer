#!/usr/bin/env python3
r"""
Generic model comparison script that compares a Hugging Face baseline model
against multiple ONNX Runtime GenAI models with different execution providers.

Usage:
python compare_models_generic.py --hf_model "F:\shared\Llama-3.1-8B-Instruct"
    --ep cuda --path "G:\models\cuda_model"
    --ep directml --path "G:\models\directml_model"
    --output "comparison_results.json"
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch

# We'll use subprocess calls to run extraction scripts with fresh Python processes
# This ensures no import cache issues when switching packages

# Mapping of execution providers to their corresponding ONNX Runtime packages
EP_PACKAGE_MAP = {
    "cuda": "onnxruntime-genai-cuda",
    "directml": "onnxruntime-genai-directml",
    "cpu": "onnxruntime-genai",
}

DEBUG = False  # Global debug flag


def debug_print(message):
    """Print debug message only if DEBUG is True"""
    if DEBUG:
        print(f"[DEBUG] {message}")


def run_command(cmd, description="", capture_output=True):
    """Run a command and handle errors"""
    debug_print(f"[INFO] {description}")
    debug_print(f"Running: {' '.join(cmd)}")

    try:
        if capture_output:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)
            if result.stdout and DEBUG:
                print(f"[OUT] {result.stdout}")
        else:
            # Real-time output - shows prints as they happen
            result = subprocess.run(cmd, check=True, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {e}")
        if capture_output and hasattr(e, "stdout") and e.stdout:
            print(f"[STDOUT] {e.stdout}")
        if capture_output and hasattr(e, "stderr") and e.stderr:
            print(f"[STDERR] {e.stderr}")
        return False


def get_python_executable():
    """Get the current Python executable being used"""
    return sys.executable


def uninstall_onnxruntime_packages():
    """Uninstall all ONNX Runtime packages"""
    packages_to_remove = [
        "onnxruntime",
        "onnxruntime-genai",
        "onnxruntime-genai-cuda",
        "onnxruntime-gpu",
        "onnxruntime-directml",
        "onnxruntime-genai-directml",
    ]

    debug_print(f"Packages to remove: {packages_to_remove}")
    python_exe = get_python_executable()
    debug_print(f"Python executable: {python_exe}")
    cmd = [python_exe, "-m", "pip", "uninstall", "-y", *packages_to_remove]
    run_command(cmd, "Uninstalling existing ONNX Runtime packages")


def install_package(package_name):
    """Install a specific package"""
    debug_print(f"Installing package: {package_name}")
    python_exe = get_python_executable()
    debug_print(f"Python executable: {python_exe}")
    cmd = [python_exe, "-m", "pip", "install", package_name, "--force-reinstall"]
    debug_print(f"Install command: {' '.join(cmd)}")
    return run_command(cmd, f"Installing {package_name}")


# Module cache clearing is no longer needed since we use subprocess calls


def extract_hf_logits_subprocess(model_path, device="cuda"):
    """Extract logits from Hugging Face model using subprocess"""
    print("[INFO] Extracting logits from Hugging Face baseline model...")
    debug_print(f"Model path: {model_path}, Device: {device}")

    # Create temporary output file
    output_file = f"temp_logits_hf_{int(time.time())}.pkl"
    debug_print(f"Temporary output file: {output_file}")

    try:
        python_exe = get_python_executable()
        cmd = [
            python_exe,
            "extract_logits_hf.py",
            "--model_path",
            model_path,
            "--output_file",
            output_file,
            "--device",
            device,
        ]
        if DEBUG:
            cmd.append("--debug")

        if not run_command(cmd, "Running HF logits extraction", capture_output=False):
            raise RuntimeError("HF logits extraction failed")

        # Load the extracted logits
        debug_print(f"Loading logits from: {output_file}")
        with open(output_file, "rb") as f:
            logits_data = pickle.load(f)

        debug_print(f"Loaded logits data keys: {logits_data.keys()}")
        debug_print(
            f"Total chunks: {logits_data['total_chunks']}, Seq len: {logits_data['seq_len']}"
        )

        # Clean up temporary file
        try:
            os.remove(output_file)
            debug_print(f"Cleaned up temporary file: {output_file}")
        except Exception:
            pass

        print(f"[INFO] HF logits extraction completed ({logits_data['total_chunks']} chunks)")
        return logits_data

    except Exception as e:
        # Clean up temporary file on error
        import contextlib

        with contextlib.suppress(BaseException):
            os.remove(output_file)
        print(f"[ERROR] Failed to extract HF logits: {e}")
        raise


def extract_onnx_logits_subprocess(model_path, provider):
    """Extract logits from ONNX Runtime GenAI model using subprocess"""
    print(f"[INFO] Extracting logits from {provider.upper()} model...")
    debug_print(f"Model path: {model_path}, Provider: {provider}")

    # Create temporary output file
    output_file = f"temp_logits_{provider}_{int(time.time())}.pkl"
    debug_print(f"Temporary output file: {output_file}")

    try:
        python_exe = get_python_executable()
        cmd = [
            python_exe,
            "extract_logits.py",
            "--model_path",
            model_path,
            "--output_file",
            output_file,
            "--provider",
            provider,
        ]
        if DEBUG:
            cmd.append("--debug")

        if not run_command(
            cmd, f"Running {provider.upper()} logits extraction", capture_output=False
        ):
            raise RuntimeError(f"{provider.upper()} logits extraction failed")

        # Load the extracted logits
        debug_print(f"Loading logits from: {output_file}")
        with open(output_file, "rb") as f:
            logits_data = pickle.load(f)

        debug_print(f"Loaded logits data keys: {logits_data.keys()}")
        debug_print(
            f"Total chunks: {logits_data['total_chunks']}, Seq len: {logits_data['seq_len']}"
        )

        # Clean up temporary file
        import contextlib

        with contextlib.suppress(BaseException):
            os.remove(output_file)
            debug_print(f"Cleaned up temporary file: {output_file}")

        print(
            f"[INFO] {provider.upper()} logits extraction completed ({logits_data['total_chunks']} chunks)"
        )
        return logits_data

    except Exception as e:
        # Clean up temporary file on error
        import contextlib

        with contextlib.suppress(BaseException):
            os.remove(output_file)
        print(f"[ERROR] Failed to extract {provider.upper()} logits: {e}")
        raise


def compute_kl_divergence_from_logits(log_probs_ref, log_probs_tar):
    """
    Compute KL divergence between two log probability distributions.
    Same logic as in compute_kl_divergence.py
    """
    debug_print(
        f"Computing KL divergence - log_probs shapes: ref={log_probs_ref.shape}, tar={log_probs_tar.shape}"
    )
    kl_divergence = 0.0
    for i in range(log_probs_ref.shape[0]):
        log_probs_ref_i = np.array(log_probs_ref[i])
        log_probs_tar_i = np.array(log_probs_tar[i])
        prob_ref_i = np.exp(log_probs_ref_i)
        kl_divergence += np.sum(prob_ref_i * np.abs(log_probs_ref_i - log_probs_tar_i))
    kl_divergence = kl_divergence / log_probs_ref.shape[0]
    debug_print(f"KL divergence computed: {kl_divergence}")
    return kl_divergence


def to_serializable(obj):
    """
    Recursively convert numpy types and torch types to native Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_serializable(v) for v in obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (torch.Tensor,)):
        return obj.detach().cpu().tolist()
    else:
        return obj


def compute_unified_comparison(model_logits_list, output_file):
    """
    Compute KL divergence comparison between all models in a unified way
    model_logits_list: List of tuples (model_name, model_data)
    """
    print("\n[INFO] Computing unified KL divergence comparison...")
    debug_print(f"Number of models to compare: {len(model_logits_list)}")
    debug_print(f"Model names: {[name for name, _ in model_logits_list]}")

    # Validate compatibility
    reference_data = model_logits_list[0][1]  # Use first model as reference for validation
    total_chunks = reference_data["total_chunks"]
    seq_len = reference_data["seq_len"]

    debug_print(f"Reference model: {model_logits_list[0][0]}")
    debug_print(f"Reference total_chunks: {total_chunks}, seq_len: {seq_len}")

    for model_name, data in model_logits_list:
        debug_print(
            f"Validating {model_name}: chunks={data['total_chunks']}, seq_len={data['seq_len']}"
        )
        if data["total_chunks"] != total_chunks:
            debug_print(
                f"[WARNING] Chunk count mismatch in {model_name}: {data['total_chunks']} vs {total_chunks}"
            )
        if data["seq_len"] != seq_len:
            debug_print(
                f"[WARNING] Sequence length mismatch in {model_name}: {data['seq_len']} vs {seq_len}"
            )

    print(f"[INFO] Computing KL divergences for {total_chunks} chunks...")

    # Process each chunk and compute all pairwise KL divergences
    chunk_results = []
    pairwise_totals = {}

    # Initialize totals for all pairs
    debug_print("Initializing pairwise comparisons:")
    for i in range(len(model_logits_list)):
        for j in range(i + 1, len(model_logits_list)):
            model1_name = model_logits_list[i][0]
            model2_name = model_logits_list[j][0]
            pair_key = f"{model1_name}_vs_{model2_name}"
            pairwise_totals[pair_key] = 0.0
            debug_print(f"  Pair: {pair_key}")

    for chunk_idx in range(total_chunks):
        debug_print(f"[PROGRESS] Processing chunk {chunk_idx + 1}/{total_chunks}...")

        chunk_result = {
            "chunk_id": int(chunk_idx + 1),
            "begin_loc": int(reference_data["chunk_info"][chunk_idx]["begin_loc"]),
            "end_loc": int(reference_data["chunk_info"][chunk_idx]["end_loc"]),
            "kl_divergences": {},
        }

        # Get logits for this chunk from all models
        chunk_logits = []
        for model_name, model_data in model_logits_list:
            logits = model_data["logits"][chunk_idx]
            debug_print(f"  {model_name} logits shape: {getattr(logits, 'shape', type(logits))}")
            chunk_logits.append((model_name, logits))

        # Find minimum sequence length for this chunk
        min_seq_len = min(getattr(logits, "shape", [None, 0])[1] for _, logits in chunk_logits)
        # Assume all have same vocab size
        vocab_size = getattr(chunk_logits[0][1], "shape", [None, None, 0])[2]
        debug_print(f"  Min seq len: {min_seq_len}, Vocab size: {vocab_size}")

        # Trim all logits to matching dimensions
        trimmed_logits = []
        for model_name, logits in chunk_logits:
            # Ensure logits is a numpy array
            arr = np.array(logits)
            trimmed = arr[:, :min_seq_len, :vocab_size]
            debug_print(f"  Trimmed {model_name} from {arr.shape} to {trimmed.shape}")
            trimmed_logits.append((model_name, trimmed))

        # Convert logits to log probabilities for all models
        log_probs_list = []
        for model_name, logits in trimmed_logits:
            logits_tensor = torch.tensor(logits, dtype=torch.float32)
            log_probs = torch.nn.functional.log_softmax(logits_tensor, dim=2).cpu().numpy()
            log_probs = np.squeeze(log_probs, axis=0)
            log_probs_list.append((model_name, log_probs))

        # Compute all pairwise KL divergences for this chunk
        for i in range(len(log_probs_list)):
            for j in range(i + 1, len(log_probs_list)):
                model1_name, log_probs1 = log_probs_list[i]
                model2_name, log_probs2 = log_probs_list[j]

                chunk_kl = compute_kl_divergence_from_logits(log_probs1, log_probs2)
                pair_key = f"{model1_name}_vs_{model2_name}"

                # Instead of assigning to an object (which is not type-checked as dict), use dict update
                kl_divergences: dict = chunk_result.get("kl_divergences", {})
                kl_divergences[pair_key] = float(chunk_kl)
                chunk_result["kl_divergences"] = kl_divergences
                pairwise_totals[pair_key] += chunk_kl

                debug_print(f"    {pair_key}: {chunk_kl:.6f}")

        chunk_results.append(chunk_result)

    # Calculate average KL divergences
    num_chunks = len(chunk_results)
    pairwise_averages = {pair: total / num_chunks for pair, total in pairwise_totals.items()}

    debug_print("\nFinal KL divergence totals:")
    for pair, total in pairwise_totals.items():
        debug_print(f"  {pair}: total={total:.6f}, avg={pairwise_averages[pair]:.6f}")

    # Prepare results
    results = {
        "models": {model_name: str(data["model_path"]) for model_name, data in model_logits_list},
        "total_chunks": int(num_chunks),
        "sequence_length": int(seq_len),
        "max_context_length": int(reference_data["max_context_length"]),
        "kl_divergences": {
            pair: {"total": float(pairwise_totals[pair]), "average": float(pairwise_averages[pair])}
            for pair in pairwise_totals
        },
        "chunk_results": chunk_results,
        "computation_timestamp": datetime.now().isoformat(),
        "summary": {
            "interpretation": "Lower KL divergence indicates more similar model outputs",
            "baseline_reference": model_logits_list[0][0],
            "pairwise_averages": {pair: float(avg) for pair, avg in pairwise_averages.items()},
            "chunks_processed": int(num_chunks),
        },
    }

    return results


def validate_inputs(hf_model, ep_path_pairs):
    """Validate that all input paths exist and EPs are supported"""
    # Check HF model path (only if provided)
    if hf_model and not os.path.exists(hf_model):
        print(f"[ERROR] Hugging Face model path does not exist: {hf_model}")
        return False

    # Check execution providers and paths
    for ep, path in ep_path_pairs:
        if ep not in EP_PACKAGE_MAP:
            print(f"[ERROR] Unsupported execution provider: {ep}")
            print(f"[ERROR] Supported providers: {list(EP_PACKAGE_MAP.keys())}")
            return False

        if not os.path.exists(path):
            print(f"[ERROR] Model path for {ep} does not exist: {path}")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generic model comparison: HF baseline vs ONNX Runtime GenAI models (or ONNX-only comparison)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare HF vs CUDA model
  python compute_kl_divergence.py --hf_model "F:\\shared\\Llama-3.1-8B-Instruct"
      --ep cuda --path "G:\\models\\cuda_model" --output "hf_vs_cuda.json"

  # Compare HF vs CUDA vs DirectML models
  python compute_kl_divergence.py --hf_model "F:\\shared\\Llama-3.1-8B-Instruct"
      --ep cuda --path "G:\\models\\cuda_model"
      --ep directml --path "G:\\models\\directml_model"
      --output "hf_vs_cuda_vs_directml.json"

  # Compare multiple models with same EP (e.g., different CUDA model variants)
  python compute_kl_divergence.py --hf_model "F:\\shared\\Llama-3.1-8B-Instruct"
      --ep cuda --path "G:\\models\\cuda_fp16"
      --ep cuda --path "G:\\models\\cuda_int4"
      --ep directml --path "G:\\models\\directml_model"
      --output "multi_model_comparison.json"

  # Compare two ONNX models with same EP (no HF model needed, uses optimized same_ep script)
  python compute_kl_divergence.py --ep cuda --path "G:\\models\\cuda_fp16"
      --ep cuda --path "G:\\models\\cuda_int4"
      --output "cuda_fp16_vs_int4.json"

  # Compare ONNX models with different EPs (no HF model needed)
  python compute_kl_divergence.py --ep cuda --path "G:\\models\\cuda_model"
      --ep directml --path "G:\\models\\directml_model"
      --output "cuda_vs_directml.json"

  # Compare multiple ONNX models with mixed EPs (no HF model needed)
  python compute_kl_divergence.py --ep cuda --path "G:\\models\\cuda_fp16"
      --ep cuda --path "G:\\models\\cuda_int4"
      --ep directml --path "G:\\models\\directml_model"
      --output "multi_onnx_comparison.json"

Supported execution providers: cuda, directml, cpu
Note: Multiple models with same EP are supported and will be named ep_1, ep_2, etc.
Note: HF model is now optional - you can compare ONNX models directly
Note: If no HF model is provided and exactly 2 models with same EP are given,
      the script will automatically use KL_divergence_metrics_same_ep.py for optimal performance
        """,
    )

    parser.add_argument(
        "--hf_model",
        required=False,
        default=None,
        help="Path to Hugging Face baseline model (optional - can compare ONNX models directly)",
    )
    parser.add_argument(
        "--ep",
        action="append",
        required=True,
        help="Execution provider (can be specified multiple times)",
    )
    parser.add_argument(
        "--path",
        action="append",
        required=True,
        help="Model path (must match order of --ep arguments)",
    )
    parser.add_argument("--output", required=True, help="Output JSON file for results")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for HF model inference (default: cuda)",
    )
    parser.add_argument(
        "--keep_logits", action="store_true", help="Keep extracted logits files after comparison"
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug output")

    args = parser.parse_args()

    # Set global debug flag
    global DEBUG
    DEBUG = args.debug

    # Validate arguments
    if len(args.ep) != len(args.path):
        print("[ERROR] Number of --ep arguments must match number of --path arguments")
        return 1

    debug_print(f"Execution providers: {args.ep}")
    debug_print(f"Model paths: {args.path}")

    # No limit on number of models - support any number of EP/path combinations

    ep_path_pairs = list(zip(args.ep, args.path))
    debug_print(f"EP-Path pairs: {ep_path_pairs}")

    # Check if we should delegate to KL_divergence_metrics_same_ep.py
    # Condition: No HF model provided, exactly 2 models, and same execution provider
    same_ep = ep_path_pairs[0][0] == ep_path_pairs[1][0] if len(ep_path_pairs) == 2 else "N/A"
    debug_print(
        f"Checking delegation conditions: hf_model={args.hf_model}, "
        f"num_models={len(ep_path_pairs)}, same_ep={same_ep}"
    )
    if (
        args.hf_model is None
        and len(ep_path_pairs) == 2
        and ep_path_pairs[0][0] == ep_path_pairs[1][0]
    ):
        print("=" * 80)
        print("DETECTED: Two ONNX models with same EP, no HF model")
        print("Delegating to KL_divergence_metrics_same_ep.py")
        print("=" * 80)
        debug_print("Delegation conditions met - calling KL_divergence_metrics_same_ep.py")
        print(f"Reference Model: {ep_path_pairs[0][1]}")
        print(f"Target Model: {ep_path_pairs[1][1]}")
        print(f"Execution Provider: {ep_path_pairs[0][0].upper()}")
        print("=" * 80)

        # Install the correct ONNX Runtime package for this EP
        ep = ep_path_pairs[0][0]
        print(f"\n[INFO] Ensuring {ep.upper()} environment is set up...")
        debug_print(f"Installing {EP_PACKAGE_MAP[ep]} for same_ep script")
        uninstall_onnxruntime_packages()
        if not install_package(EP_PACKAGE_MAP[ep]):
            print(f"[ERROR] Failed to install {EP_PACKAGE_MAP[ep]}")
            return 1
        debug_print(f"Successfully set up {ep.upper()} environment")

        # Call KL_divergence_metrics_same_ep.py
        python_exe = get_python_executable()
        cmd = [
            python_exe,
            "KL_divergence_metrics_same_ep.py",
            "--reference_model",
            ep_path_pairs[0][1],
            "--target_model",
            ep_path_pairs[1][1],
        ]

        print("\n[INFO] Running KL_divergence_metrics_same_ep.py...")
        result = subprocess.run(cmd, shell=True)

        if result.returncode == 0:
            print("\n[SUCCESS] KL divergence computation completed successfully")
        else:
            print("\n[ERROR] KL divergence computation failed")

        return result.returncode

    # Validate inputs
    if not validate_inputs(args.hf_model, ep_path_pairs):
        return 1

    print("=" * 80)
    if args.hf_model:
        print("GENERIC MODEL COMPARISON (with HF baseline)")
    else:
        print("ONNX MODEL COMPARISON (no HF baseline)")
    print("=" * 80)
    if args.hf_model:
        print(f"Hugging Face Model: {args.hf_model}")
    for ep, path in ep_path_pairs:
        print(f"{ep.upper()} Model: {path}")
    print(f"Output: {args.output}")
    if args.hf_model:
        print(f"Device for HF: {args.device}")
    print("=" * 80)

    start_time = time.time()

    try:
        # Store all model logits data
        model_logits_list = []

        # Step 1: Extract logits from HF model (if provided)
        if args.hf_model:
            debug_print("\n Hugging Face Baseline Extraction")
            hf_logits_data = extract_hf_logits_subprocess(args.hf_model, args.device)
            model_logits_list.append(("huggingface", hf_logits_data))

        # Step 2: Extract logits from each ONNX model
        current_ep = None  # Track current installed EP to avoid unnecessary reinstalls

        for i, (ep, path) in enumerate(ep_path_pairs):
            # Create unique model name for same EP models
            model_name = (
                f"{ep}_{i + 1}"
                if ep_path_pairs.count((ep, path)) > 1
                or sum(1 for x in ep_path_pairs if x[0] == ep) > 1
                else ep
            )

            debug_print(
                f"Processing model {i + 1}/{len(ep_path_pairs)}: ep={ep}, path={path}, model_name={model_name}"
            )
            print(f"\n[INFO] Processing {ep.upper()} model ({i + 1}/{len(ep_path_pairs)})")
            debug_print(f"Path: {path}")

            # Package management - only reinstall if EP changed
            if current_ep != ep:
                print(f"[INFO] Switching to {ep.upper()} environment...")
                debug_print(f"EP changed from {current_ep} to {ep}")
                debug_print("Uninstalling existing ONNX Runtime packages")
                uninstall_onnxruntime_packages()

                debug_print(f"Installing {EP_PACKAGE_MAP[ep]}")
                if not install_package(EP_PACKAGE_MAP[ep]):
                    print(f"[ERROR] Failed to install {EP_PACKAGE_MAP[ep]}")
                    return 1
                current_ep = ep
                debug_print(f"Successfully switched to {ep} environment")
            else:
                debug_print(f"Reusing {ep.upper()} environment (already installed)")
                debug_print(f"EP unchanged: {ep}")

            # Extract logits
            debug_print(f"Extracting logits for {model_name}")
            onnx_logits_data = extract_onnx_logits_subprocess(path, ep)
            model_logits_list.append((model_name, onnx_logits_data))
            debug_print(
                f"Added {model_name} to model_logits_list (total models: {len(model_logits_list)})"
            )

        # Step 3: Compute unified comparison
        print("\n[INFO] Computing KL Divergences...")
        debug_print(f"Total models for comparison: {len(model_logits_list)}")
        debug_print(f"Model list: {[name for name, _ in model_logits_list]}")
        results = compute_unified_comparison(model_logits_list, args.output)

        end_time = time.time()

        # Add timing information
        results["computation_time_seconds"] = float(end_time - start_time)

        # Step 4: Save results
        print(f"\n[INFO] Saving results to: {args.output}")
        debug_print(f"Results keys: {results.keys()}")
        debug_print("Serializing results to JSON")
        with open(args.output, "w") as f:
            json.dump(to_serializable(results), f, indent=2)
        debug_print(f"Results saved successfully to {args.output}")

        # Step 5: Generate summary
        print("\n" + "=" * 80)
        print("COMPARISON COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
        print(f"Results saved to: {args.output}")
        print()
        print("MODELS COMPARED:")
        for model_name, model_path in results["models"].items():
            print(f"  {model_name.upper()}: {model_path}")
        print()

        # Display KL divergence results
        if "kl_divergences" in results:
            print("KL DIVERGENCE RESULTS:")
            kl_divs = results["kl_divergences"]
            for comparison, values in kl_divs.items():
                comp_name = comparison.replace("_", " ").upper()
                print(f"  {comp_name}: {values['average']:.6f}")

        print()
        debug_print("INTERPRETATION:")
        debug_print("- Lower KL divergence = more similar model outputs")
        debug_print("- All comparisons are pairwise between models")
        debug_print("- Values show how much models differ from each other")
        print("=" * 80)

        # Optional: Save logits to files if requested
        if args.keep_logits:
            debug_print("\n[SAVE] Saving logits to individual files...")
            for model_name, model_data in model_logits_list:
                logits_file = f"logits_{model_name}.pkl"
                with open(logits_file, "wb") as f:
                    pickle.dump(model_data, f)
                debug_print(f"  Saved: {logits_file}")
            debug_print("Note: Logits files saved for future reuse.")

        return 0

    except KeyboardInterrupt:
        print("\n[INFO] Comparison interrupted by user")
        return 1
    except Exception as e:
        debug_print(f"[ERROR] Unexpected error: {e}")
        import traceback

        if DEBUG:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
