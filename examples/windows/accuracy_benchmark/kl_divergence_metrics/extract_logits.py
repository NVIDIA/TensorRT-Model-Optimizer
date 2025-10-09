#!/usr/bin/env python3
"""
Extract logits from ONNX Runtime GenAI models and save them to file.
Follows the same logic as KL_divergence_metrics.py but saves logits instead of computing KL divergence.
"""

import argparse
import os
import pickle
import time

import numpy as np
import torch
from datasets import load_dataset

# Lazy import for onnxruntime_genai - will be imported when needed
og = None

DEBUG = False


def debug_print(message):
    """
    Print debug message only if DEBUG flag is enabled.

    Args:
        message (str): Debug message to print.
    """
    if DEBUG:
        print(f"[DEBUG] {message}")


def get_wikitext2():
    """
    Load and concatenate the WikiText-2 test dataset.

    Returns:
        str: Concatenated text from all samples, separated by double newlines.

    Note:
        Requires HuggingFace CLI authentication.
    """
    print("\n[INFO] Loading Wikitext-2 'test' split ...")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    print(f"[DATASET] Number of raw samples: {len(test)}")

    # Concatenate all text samples into a single string, separated by double newlines
    result = "\n\n".join(text for text in test["text"])
    print(f"[DATASET] Total text length: {len(result)} characters")
    return result


def extract_logits_from_model(model_path, provider="cuda"):
    """
    Extract logits from an ONNX Runtime GenAI model on WikiText-2 dataset.

    Uses a sliding window approach to process the dataset in chunks and extract
    model logits for each chunk.

    Args:
        model_path (str): Path to the ONNX Runtime GenAI model directory.
        provider (str, optional): Execution provider hint ('cuda', 'directml', 'cpu').
                                 The actual provider is determined by the installed package.
                                 Defaults to "cuda".

    """
    print(f"\n[INFO] Loading model from: {model_path}")
    print(f"[INFO] Using provider: {provider}")

    try:
        # Import onnxruntime_genai when needed
        global og
        try:
            import onnxruntime_genai as og_module

            og = og_module
            print("[INFO] Successfully imported onnxruntime_genai")
        except ImportError as e:
            raise ImportError(
                f"Failed to import onnxruntime_genai: {e}. "
                f"Make sure the correct package is installed for provider '{provider}'"
            )

        # Load model and tokenizer with explicit provider configuration
        print(f"[INFO] Creating model with provider: {provider}")

        # For ONNX Runtime GenAI, the execution provider is determined by the installed package
        # We don't need to explicitly set it in the model creation
        model = og.Model(model_path)
        tokenizer = og.Tokenizer(model)
        print("[INFO] Successfully loaded model and tokenizer")

        # Print available providers for debugging
        try:
            import onnxruntime as ort

            available_providers = ort.get_available_providers()
            print(f"[INFO] Available ONNX Runtime providers: {available_providers}")
        except Exception:
            print("[INFO] Could not check available ONNX Runtime providers")

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print(
            f"[ERROR] Make sure the correct onnxruntime-genai package is installed for provider '{provider}'"
        )

        # Add more detailed error information
        try:
            import onnxruntime as ort

            available_providers = ort.get_available_providers()
            print(f"[ERROR] Currently available providers: {available_providers}")

            if provider == "cuda" and "CUDAExecutionProvider" not in available_providers:
                print("[ERROR] CUDA provider not available. Install onnxruntime-genai-cuda")
            elif provider == "directml" and "DmlExecutionProvider" not in available_providers:
                print("[ERROR] DirectML provider not available. Install onnxruntime-genai-directml")
        except Exception:
            pass

        raise

    # Parameters (same as KL_divergence_metrics.py)
    max_context_length = 1024

    print(f"[INFO] Max context length: {max_context_length}")

    # Load dataset
    dataset = get_wikitext2()

    # Tokenize
    print("[INFO] Tokenizing dataset...")
    input_ids = tokenizer.encode_batch([dataset])
    if isinstance(input_ids, dict) and "input_ids" in input_ids:
        input_ids = input_ids["input_ids"]
    if hasattr(input_ids, "as_numpy"):
        input_ids = input_ids.as_numpy()
    input_ids = np.array(input_ids)

    # Ensure input_ids is 2D (batch, seq_len)
    if input_ids.ndim == 1:
        input_ids = np.expand_dims(input_ids, 0)

    # Convert to torch tensor
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    seq_len = int(input_ids.shape[1])

    print(f"[INFO] Input sequence length: {seq_len}")

    # Store all logits
    all_logits = []
    chunk_info = []

    # Process chunks following the same logic as KL_divergence_metrics.py
    for chunk_count, begin_loc in enumerate(
        range(0, min(50 * max_context_length, seq_len), max_context_length), 1
    ):
        if DEBUG:
            print(f"[PROGRESS] Processing chunk {chunk_count}...")

        end_loc = min(begin_loc + max_context_length, seq_len)

        # Extract the current chunk of input tokens
        input_ids_chunk = input_ids[:, begin_loc:end_loc].clone()
        if DEBUG:
            print(f"  Chunk range: {begin_loc} to {end_loc}")
            print(f"  Chunk shape: {input_ids_chunk.shape}")

        # Set up generator parameters for deterministic generation (no sampling)
        params = og.GeneratorParams(model)
        params.set_search_options(
            max_length=int(input_ids_chunk.shape[1]), do_sample=False, early_stopping=False
        )

        # Create generator and append input tokens
        generator = og.Generator(model, params)
        generator.append_tokens(input_ids_chunk.numpy())

        # Run the model forward pass
        with torch.no_grad():
            try:
                generator.generate_next_token()
            except Exception as e:
                print(f"[ERROR] generate_next_token() failed: {e}")
                break

            # Get logits output from the model
            logits = generator.get_output("logits")
            if hasattr(logits, "as_numpy"):
                logits = logits.as_numpy()
            if DEBUG:
                print(f"  Logits shape: {logits.shape}")

            # Convert to torch tensor and store
            logits_tensor = torch.tensor(logits, dtype=torch.float32)
            all_logits.append(logits_tensor.cpu().numpy())

            # Store chunk information
            chunk_info.append(
                {
                    "chunk_id": chunk_count,
                    "begin_loc": begin_loc,
                    "end_loc": end_loc,
                    "shape": logits_tensor.shape,
                }
            )

    print(f"[INFO] Extracted logits from {len(all_logits)} chunks")

    return {
        "logits": all_logits,
        "chunk_info": chunk_info,
        "model_path": model_path,
        "provider": provider,
        "seq_len": seq_len,
        "max_context_length": max_context_length,
        "total_chunks": len(all_logits),
    }


def main():
    """
    Command-line entry point for extracting logits from ONNX Runtime GenAI models.

    Extracts model logits on WikiText-2 dataset and saves them to a pickle file
    for later KL divergence comparison.

    """
    parser = argparse.ArgumentParser(description="Extract logits from ONNX Runtime GenAI model")
    parser.add_argument("--model_path", required=True, help="Path to model directory")
    parser.add_argument("--output_file", required=True, help="Output pickle file path")
    parser.add_argument(
        "--provider",
        default="cuda",
        choices=["cuda", "directml", "cpu"],
        help="Execution provider (cuda, directml, or cpu)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug output")

    args = parser.parse_args()

    # Set global debug flag
    global DEBUG
    DEBUG = args.debug

    # Validate model directory exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model directory does not exist: {args.model_path}")
        return 1

    try:
        # Extract logits
        start_time = time.time()
        logits_data = extract_logits_from_model(args.model_path, args.provider)
        end_time = time.time()

        # Save to file
        print(f"\n[INFO] Saving logits to: {args.output_file}")
        with open(args.output_file, "wb") as f:
            pickle.dump(logits_data, f)

        print(f"[INFO] Extraction completed in {end_time - start_time:.2f} seconds")
        print(f"[INFO] Total chunks processed: {logits_data['total_chunks']}")
        print(f"[INFO] Model: {args.model_path}")
        print(f"[INFO] Provider: {args.provider}")
        print(f"[INFO] Output: {args.output_file}")

        return 0

    except Exception as e:
        print(f"[ERROR] Failed to extract logits: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
