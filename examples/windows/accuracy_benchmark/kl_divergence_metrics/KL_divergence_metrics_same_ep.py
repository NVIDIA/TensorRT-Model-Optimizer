"""
Optimized KL divergence comparison for ONNX Runtime GenAI models with the same execution provider.

This script efficiently compares two ONNX Runtime GenAI models by computing KL divergence
between their output distributions without package switching overhead.

Usage:
    python KL_divergence_metrics_same_ep.py \\
        --reference_model "path/to/reference/model" \\
        --target_model "path/to/target/model"
"""

import argparse
import os

import numpy as np
import onnxruntime_genai as og
import torch
from datasets import load_dataset

DEBUG = False


def get_kl_divergence(log_probs_ref, log_probs_tar):
    """
    Compute Kullback-Leibler divergence between two log probability distributions.

    KL divergence measures how one probability distribution diverges from a reference
    distribution. Lower values indicate more similar distributions.

    Args:
        log_probs_ref (np.ndarray): Reference log probabilities with shape (seq_len, vocab_size).
        log_probs_tar (np.ndarray): Target log probabilities with shape (seq_len, vocab_size).

    Returns:
        float: Average KL divergence across all positions.

    Note:
        Formula: KL(P||Q) = sum(P(x) * |log(P(x)) - log(Q(x))|) averaged over sequence length
    """
    kl_divergence = 0.0
    for i in range(log_probs_ref.shape[0]):
        log_probs_ref[i] = np.array(log_probs_ref[i])
        log_probs_tar[i] = np.array(log_probs_tar[i])
        prob_ref = np.exp(log_probs_ref[i])
        kl_divergence += np.sum(prob_ref * abs(log_probs_ref[i] - log_probs_tar[i]))
    kl_divergence = kl_divergence / log_probs_ref.shape[0]
    return kl_divergence


def get_wikitext2():
    """
    Load and concatenate the WikiText-2 test dataset.

    Returns:
        str: Concatenated text from all samples in the WikiText-2 test split,
             with samples separated by double newlines.

    Note:
        Requires HuggingFace CLI authentication to access the dataset.
    """
    # Load the Wikitext-2 test split using HuggingFace datasets
    print("\n[INFO] Loading Wikitext-2 'test' split ...")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    if DEBUG:
        print(f"[DATASET] Number of raw samples: {len(test)}")
        for i in range(3):
            print(f"[DATASET] Sample[{i}]: {repr(test[i]['text'])[:200]} ...")
    # Concatenate all text samples into a single string, separated by double newlines
    result = "\n\n".join(text for text in test["text"])
    if DEBUG:
        print(
            f"[DATASET] Concatenated text preview: {result[:512]!r} ... [total chars: {len(result)}]"
        )
    return result


def run_kl_divergence_on_models(reference_model, target_model):
    """
    Compute KL divergence between two ONNX Runtime GenAI models on WikiText-2 dataset.

    This function loads both models, processes the WikiText-2 dataset in chunks, and
    computes the KL divergence between their output distributions for each chunk.
    The results are averaged across all chunks.

    Args:
        reference_model (str): Path to the reference ONNX Runtime GenAI model directory.
        target_model (str): Path to the target ONNX Runtime GenAI model directory.

    """
    ref_model = og.Model(reference_model)
    tar_model = og.Model(target_model)
    tokenizer_ref = og.Tokenizer(ref_model)
    tokenizer_tar = og.Tokenizer(tar_model)
    max_context_length = 1024
    dataset = get_wikitext2()

    input_ids_ref = tokenizer_ref.encode_batch([dataset])
    input_ids_tar = tokenizer_tar.encode_batch([dataset])
    # Handle possible dict output from tokenizer
    if isinstance(input_ids_ref, dict) and "input_ids" in input_ids_ref:
        input_ids_ref = input_ids_ref["input_ids"]
    # Convert to numpy if needed
    if hasattr(input_ids_ref, "as_numpy"):
        input_ids_ref = input_ids_ref.as_numpy()
        if DEBUG:
            print("[TOKENIZER] Used as_numpy()")
    if isinstance(input_ids_tar, dict) and "input_ids" in input_ids_tar:
        input_ids_tar = input_ids_tar["input_ids"]
    if hasattr(input_ids_tar, "as_numpy"):
        input_ids_tar = input_ids_tar.as_numpy()
        if DEBUG:
            print("[TOKENIZER] Used as_numpy()")
    input_ids_ref = np.array(input_ids_ref)
    input_ids_tar = np.array(input_ids_tar)

    # Ensure input_ids is 2D (batch, seq_len)
    if input_ids_ref.ndim == 1:
        input_ids_ref = np.expand_dims(input_ids_ref, 0)
        if DEBUG:
            print(f"[SHAPE] Expanded dims, now: {input_ids_ref.shape}")
    if input_ids_tar.ndim == 1:
        input_ids_tar = np.expand_dims(input_ids_tar, 0)
        if DEBUG:
            print(f"[SHAPE] Expanded dims, now: {input_ids_tar.shape}")
    # Convert input_ids to torch tensor
    input_ids_ref = torch.tensor(input_ids_ref, dtype=torch.long)
    input_ids_tar = torch.tensor(input_ids_tar, dtype=torch.long)
    seq_len_ref = int(input_ids_ref.shape[1])
    seq_len_tar = int(input_ids_tar.shape[1])
    if DEBUG:
        print(f"[INFO] Ref input length: {seq_len_ref}")
        print(f"[INFO] Tar input length: {seq_len_tar}")

    if seq_len_ref != seq_len_tar:
        print(
            f"Error: Input tokenizer lengths for reference and target models do not match: "
            f"{seq_len_ref} != {seq_len_tar}"
        )
        return
    if DEBUG:
        print(f"[INFO] Input lengths match: {seq_len_ref}")
    # Slide a window over the input to compute perplexity in chunks
    total_kl_divergence = 0.0
    total_batch = 0
    for begin_loc in range(0, seq_len_ref, max_context_length):
        end_loc = min(begin_loc + max_context_length, seq_len_ref)
        # Extract the current chunk of input tokens
        input_ids_chunk_ref = input_ids_ref[:, begin_loc:end_loc].clone()
        input_ids_chunk_tar = input_ids_tar[:, begin_loc:end_loc].clone()
        if DEBUG:
            print(f"input_ids_chunk_ref.shape: {input_ids_chunk_ref.shape}")
            print(f"input_ids_chunk_tar.shape: {input_ids_chunk_tar.shape}")
        # Set up generator parameters for deterministic generation (no sampling)
        params_ref = og.GeneratorParams(ref_model)
        params_tar = og.GeneratorParams(tar_model)
        params_ref.set_search_options(
            max_length=int(input_ids_chunk_ref.shape[1]), do_sample=False, early_stopping=False
        )
        params_tar.set_search_options(
            max_length=int(input_ids_chunk_tar.shape[1]), do_sample=False, early_stopping=False
        )
        # Create generator and append input tokens
        generator_ref = og.Generator(ref_model, params_ref)
        generator_ref.append_tokens(input_ids_chunk_ref.numpy())
        generator_tar = og.Generator(tar_model, params_tar)
        generator_tar.append_tokens(input_ids_chunk_tar.numpy())

        # Run the model forward pass without gradient calculation
        with torch.no_grad():
            if DEBUG:
                print("[INFER] Running model forward pass ...")
            try:
                generator_ref.generate_next_token()
                generator_tar.generate_next_token()
            except Exception as e:
                print(f"[INFER] .generate_next_token() failed: {e}")
                break  # Fatal error
            # Get logits output from the model
            logits_ref = generator_ref.get_output("logits")
            logits_tar = generator_tar.get_output("logits")
            if DEBUG:
                print(f"logits_ref.shape: {logits_ref.shape}")
                print(f"logits_tar.shape: {logits_tar.shape}")
            # Convert numpy arrays to torch tensors
            logits_ref = torch.tensor(logits_ref, dtype=torch.float32)
            logits_tar = torch.tensor(logits_tar, dtype=torch.float32)
        # Compute log probabilities over vocabulary for each position
        log_probs_ref = torch.nn.functional.log_softmax(logits_ref, dim=2).cpu().numpy()
        log_probs_tar = torch.nn.functional.log_softmax(logits_tar, dim=2).cpu().numpy()
        if DEBUG:
            print(f"log_probs_ref.shape: {log_probs_ref.shape}")
            print(f"log_probs_tar.shape: {log_probs_tar.shape}")
        # Compute KL divergence
        kl_divergence = 0.0
        # Reshape log_probs_ref and log_probs_tar from (1, 1024, 128256) to (1024, 128256)
        log_probs_ref = log_probs_ref.squeeze(0)
        log_probs_tar = log_probs_tar.squeeze(0)

        # log_probs_ref = torch.tensor(log_probs_ref, dtype=torch.float32)
        # log_probs_tar = torch.tensor(log_probs_tar, dtype=torch.float32)
        # kl_divergence = torch.nn.functional.kl_div(
        #     log_probs_ref, log_probs_tar, reduction='batchmean', log_target=True
        # )
        kl_divergence = get_kl_divergence(log_probs_ref, log_probs_tar)
        total_kl_divergence += kl_divergence
        total_batch += 1
        if DEBUG:
            print(f"KL divergence: {kl_divergence}")
    avg_kl_divergence = total_kl_divergence / total_batch
    if DEBUG:
        print(f"Average KL divergence: {avg_kl_divergence}")
    print(f"Total KL divergence: {total_kl_divergence}")
    print(f"Total batch: {total_batch}")
    print(f"Average KL divergence: {avg_kl_divergence}")


def main():
    """
    Command-line entry point for optimized KL divergence comparison of same-EP models.

    This script is optimized for comparing two ONNX Runtime GenAI models that use
    the same execution provider, avoiding package switching overhead. It computes
    KL divergence between model outputs on the WikiText-2 dataset.

    Command-line Arguments:
        --reference_model: Path to reference model directory (required)
        --target_model: Path to target model directory (required)

    Example:
        $ python KL_divergence_metrics_same_ep.py \\
              --reference_model "G:\\models\\cuda_fp16" \\
              --target_model "G:\\models\\cuda_int4"
    """
    parser = argparse.ArgumentParser(
        description="Run KL divergence evaluation on ONNX Runtime GenAI models"
    )
    parser.add_argument(
        "--reference_model", required=True, help="Path to reference model directory"
    )
    parser.add_argument("--target_model", required=True, help="Path to target model directory")
    args = parser.parse_args()

    # Validate that all model directories exist
    valid_models = []
    if os.path.exists(args.reference_model):
        valid_models.append(args.reference_model)
    else:
        print(f"Warning: Reference Model directory does not exist: {args.reference_model}")
    if os.path.exists(args.target_model):
        valid_models.append(args.target_model)
    else:
        print(f"Warning: Target Model directory does not exist: {args.target_model}")
    if len(valid_models) != 2:
        print("Error: No valid model directories provided")
        return

    print(
        f"Running KL divergence evaluation on reference model={valid_models[0]} and target model={valid_models[1]}"
    )
    run_kl_divergence_on_models(valid_models[0], valid_models[1])


if __name__ == "__main__":
    main()
