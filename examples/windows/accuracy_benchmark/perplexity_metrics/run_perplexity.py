import argparse
import os
import sys

import pandas as pd

# Ensure this directory is on sys.path for local imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from perplexity_metrics import perplexity_eval  # noqa: E402


def run_perplexity_on_models(
    model_dirs, output_file="perplexity_results.csv", i="1024", chunk_size=None
):
    """
    Run perplexity evaluation on multiple ONNX Runtime GenAI models.

    This function evaluates one or more models at different input sequence lengths,
    saves results to a CSV file, and prints a summary report. Each model-length
    combination is evaluated independently, with errors handled gracefully.

    Args:
        model_dirs (list[str]): List of model directory paths to evaluate.
                               Each directory must contain a valid ONNX Runtime GenAI model.
        output_file (str, optional): Path for the output CSV file containing results.
                                    Defaults to "perplexity_results.csv".
        i (str or list, optional): Input sequence lengths to evaluate. Can be:
                                  - String: comma-separated values (e.g., "1024,2048,4096")
                                  - List/tuple: sequence of integers
                                  - Single int: one length to evaluate
                                  Defaults to "1024".
        chunk_size (int, optional): Prefill chunk size for KV cache chunking.
                                   Required for input lengths > 1024.
                                   Overrides chunk_size in model config if provided.
                                   Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing evaluation results with columns:
                     - Model Path: Full path to model directory
                     - Input Length: Sequence length used for evaluation
                     - Perplexity: Computed perplexity score (or "N/A" if failed)
                     - Status: "Success" or "Failed"
                     - Error: Error message if failed, "None" if successful

    """
    results = []

    for model_dir in model_dirs:
        print(f"\n{'=' * 60}")
        print(f"Evaluating perplexity for: {model_dir}")
        print(f"{'=' * 60}")

        try:
            # Check if model directory exists
            if not os.path.exists(model_dir):
                print(f"Error: Model directory does not exist: {model_dir}")
                results.append(
                    {
                        "Model Path": model_dir,
                        "Perplexity": "N/A",
                        "Status": "Directory not found",
                        "Error": "Directory does not exist",
                    }
                )
                continue

            # Check if genai_config.json exists
            config_path = os.path.join(model_dir, "genai_config.json")
            if not os.path.exists(config_path):
                print(f"Error: genai_config.json not found in: {model_dir}")
                results.append(
                    {
                        "Model Path": model_dir,
                        "Perplexity": "N/A",
                        "Status": "Invalid model format",
                        "Error": "genai_config.json not found",
                    }
                )
                continue
            # Parse i as a comma-separated list of integers, pass each to perplexity_eval
            if isinstance(i, str):
                i_list = [int(x.strip()) for x in i.split(",") if x.strip()]
            elif isinstance(i, (list, tuple)):
                i_list = [int(x) for x in i]
            else:
                i_list = [int(i)]
            # For each input length, run perplexity_eval and record results
            for input_len in i_list:
                try:
                    print(f"  Evaluating with input length: {input_len}")
                    if chunk_size is None:
                        print(
                            "  Note: input length is ignored unless chunk_size is set or "
                            "config.search.chunk_size is present."
                        )
                    if chunk_size is not None:
                        print(f"  Using chunk_size: {chunk_size}")
                        perplexity = perplexity_eval(model_dir, str(input_len), chunk_size)
                    else:
                        perplexity = perplexity_eval(model_dir, str(input_len))
                    results.append(
                        {
                            "Model Path": model_dir,
                            "Input Length": int(input_len),
                            "Perplexity": float(perplexity),
                            "Status": "Success",
                            "Error": "None",
                        }
                    )
                except Exception as e:  # noqa: PERF203
                    print(f"  Error for input length {input_len}: {e!s}")
                    results.append(
                        {
                            "Model Path": model_dir,
                            "Input Length": int(input_len),
                            "Perplexity": "N/A",
                            "Status": "Failed",
                            "Error": str(e),
                        }
                    )

            print(" Perplexity evaluation completed successfully")

        except Exception as e:
            print(f"Error during perplexity evaluation: {e!s}")
            results.append(
                {"Model Path": model_dir, "Perplexity": "N/A", "Status": "Failed", "Error": str(e)}
            )

    # Create results DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 60}")

    # Print summary
    successful = df[df["Status"] == "Success"]
    failed = df[df["Status"] != "Success"]

    print("\nSummary:")
    print(f"  Successful evaluations: {len(successful)}")
    print(f"  Failed evaluations: {len(failed)}")

    if len(successful) > 0:
        print("\nPerplexity Results:")
        for _, row in successful.iterrows():
            print(
                f"  {os.path.basename(row['Model Path'])} [i={row.get('Input Length', '?')}]: "
                f"{row['Perplexity']:.4f}"
                if isinstance(row["Perplexity"], (int, float))
                else row["Perplexity"]
            )

    return df


def main():
    """
    Command-line entry point for perplexity evaluation.

    Parses command-line arguments and runs perplexity evaluation on specified
    ONNX Runtime GenAI models. Results are saved to a CSV file.

    Command-line Arguments:
        --models: One or more model directory paths (required)
        --i: Comma-separated input sequence lengths (default: "1024")
        --output: Output CSV file path (default: "perplexity_results.csv")
        --chunk_size: Prefill chunk size for prefill chunking (optional)

    Example:
        $ python run_perplexity.py --models /path/to/model
        $ python run_perplexity.py --models /path/to/model1 /path/to/model2 \\
              --i 1024,2048,4096 --chunk_size 1024 --output results.csv
    """
    parser = argparse.ArgumentParser(
        description="Run perplexity evaluation on multiple ONNX Runtime GenAI models"
    )
    parser.add_argument(
        "--models", nargs="+", required=True, help="List of model directory paths to evaluate"
    )
    parser.add_argument(
        "--i",
        default="1024",
        help="Comma-separated input seq lengths to be evaluated (e.g. 1024,2048) please enter number >= 1024",
    )
    parser.add_argument(
        "--output",
        default="perplexity_results.csv",
        help="Output CSV file name (default: perplexity_results.csv)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Chunk size for KV caching optimization (optional)",
    )

    args = parser.parse_args()

    # Validate that all model directories exist
    valid_models = []
    for model_dir in args.models:
        if os.path.exists(model_dir):
            valid_models.append(model_dir)
        else:
            print(f"Warning: Model directory does not exist: {model_dir}")

    if not valid_models:
        print("Error: No valid model directories provided")
        return

    print(f"Running perplexity evaluation on {len(valid_models)} models...")
    if args.chunk_size is not None:
        print(f"Using chunk_size: {args.chunk_size}")
    run_perplexity_on_models(valid_models, args.output, args.i, args.chunk_size)


if __name__ == "__main__":
    main()
