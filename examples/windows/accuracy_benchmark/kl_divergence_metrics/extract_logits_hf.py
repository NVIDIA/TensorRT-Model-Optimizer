#!/usr/bin/env python3
"""
Extract logits from Hugging Face model using transformers library.
Follows the same logic as extract_logits.py but uses transformers instead of ONNX Runtime GenAI.
"""

import argparse
import os
import pickle
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    print("Loading Wikitext-2 dataset...")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    debug_print(f"Number of raw samples: {len(test)}")

    # Concatenate all text samples into a single string, separated by double newlines
    result = "\n\n".join(text for text in test["text"])
    debug_print(f"Total text length: {len(result)} characters")
    print(f"Dataset loaded ({len(result):,} characters)")
    return result


def extract_logits_from_hf_model(model_path, device="cuda"):
    """
    Extract logits from a Hugging Face transformer model on WikiText-2 dataset.

    Uses a sliding window approach to process the dataset in chunks and extract
    model logits for each chunk using the transformers library.

    Args:
        model_path (str): Path to the Hugging Face model directory or model name.
        device (str, optional): Device for inference ('cuda' or 'cpu'). Defaults to "cuda".

    """
    print("Loading Hugging Face model...")
    debug_print(f"Model path: {model_path}")
    debug_print(f"Device: {device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        model = model.to("cpu")

    model.eval()

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(getattr(model, "config", None), "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    # Parameters (same as extract_logits.py)
    max_context_length = 1024

    print(f"[INFO] Max context length: {max_context_length}")

    # Load dataset
    dataset = get_wikitext2()

    # Tokenize
    print("[INFO] Tokenizing dataset...")
    inputs = tokenizer(dataset, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"]

    seq_len = int(input_ids.shape[1])

    print(f"[INFO] Input sequence length: {seq_len}")

    # Store all logits
    all_logits = []
    chunk_info = []

    # Process chunks following the same logic as extract_logits.py
    for chunk_count, begin_loc in enumerate(
        range(0, min(50 * max_context_length, seq_len), max_context_length), 1
    ):
        if DEBUG:
            print(f"[PROGRESS] Processing chunk {chunk_count}...")

        end_loc = min(begin_loc + max_context_length, seq_len)

        # Extract the current chunk of input tokens
        input_ids_chunk = input_ids[:, begin_loc:end_loc]
        if DEBUG:
            print(f"  Chunk range: {begin_loc} to {end_loc}")
            print(f"  Chunk shape: {input_ids_chunk.shape}")

        # Move to device
        input_ids_chunk = input_ids_chunk.to(model.device)
        # Run the model forward pass
        with torch.no_grad():
            try:
                outputs = model(input_ids_chunk)
                logits = outputs.logits
                if DEBUG:
                    print(f"  Logits shape: {logits.shape}")

                # Store logits (convert to CPU and numpy)
                logits_numpy = logits.cpu().numpy()
                all_logits.append(logits_numpy)

                # Store chunk information
                chunk_info.append(
                    {
                        "chunk_id": chunk_count,
                        "begin_loc": begin_loc,
                        "end_loc": end_loc,
                        "shape": logits_numpy.shape,
                    }
                )

            except Exception as e:
                print(f"[ERROR] Model forward pass failed: {e}")
                break

    print(f"[INFO] Extracted logits from {len(all_logits)} chunks")

    return {
        "logits": all_logits,
        "chunk_info": chunk_info,
        "model_path": model_path,
        "provider": "huggingface_transformers",
        "device": device,
        "seq_len": seq_len,
        "max_context_length": max_context_length,
        "total_chunks": len(all_logits),
    }


def main():
    """
    Command-line entry point for extracting logits from Hugging Face models.

    Extracts model logits on WikiText-2 dataset and saves them to a pickle file
    for later KL divergence comparison with ONNX Runtime GenAI models.

    """
    parser = argparse.ArgumentParser(description="Extract logits from Hugging Face model")
    parser.add_argument("--model_path", required=True, help="Path to Hugging Face model directory")
    parser.add_argument("--output_file", required=True, help="Output pickle file path")
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="Device to use (cuda or cpu)"
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
        logits_data = extract_logits_from_hf_model(args.model_path, args.device)
        end_time = time.time()

        # Save to file
        print(f"\n[INFO] Saving logits to: {args.output_file}")
        with open(args.output_file, "wb") as f:
            pickle.dump(logits_data, f)

        print(f"[INFO] Extraction completed in {end_time - start_time:.2f} seconds")
        print(f"[INFO] Total chunks processed: {logits_data['total_chunks']}")
        print(f"[INFO] Model: {args.model_path}")
        print(f"[INFO] Device: {args.device}")
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
