import argparse
import os

import pandas as pd
from perplexity_metrics import perplexity_eval


def run_perplexity_on_models(
    model_dirs, output_file="perplexity_results.csv", i="1024", chunk_size=None
):
    """
    Run perplexity evaluation on multiple model directories

    Args:
        model_dirs: List of model directory paths
        output_file: Output CSV file name
        i: Comma-separated input sequence lengths to evaluate
        chunk_size: Optional chunk size for KV caching optimization
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
                print(f"  Evaluating with input length: {input_len}")
                if chunk_size is not None:
                    print(f"  Using chunk_size: {chunk_size}")
                    perplexity = perplexity_eval(model_dir, str(input_len), chunk_size)
                else:
                    perplexity = perplexity_eval(model_dir, str(input_len))
                results.append(
                    {
                        "Model Path": model_dir,
                        "Input Length": input_len,
                        "Perplexity": f"{perplexity:.4f}",
                        "Status": "Success",
                        "Error": "None",
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
            print(f"  {os.path.basename(row['Model Path'])}: {row['Perplexity']}")

    return df


def main():
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
