import argparse

import torch
import transformers

import modelopt.torch.quantization as mtq


def main():
    parser = argparse.ArgumentParser(
        description="Apply rotation to a transformer model for improved quantization."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or HuggingFace model ID to load the model from (e.g., meta-llama/Meta-Llama-3-8B)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="transformer_universal",
        help="Config name (e.g., transformer_universal) or path to a config yaml file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the rotated model. If not provided, model will not be saved.",
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", trust_remote_code=True, torch_dtype=torch.float32
    )
    # model.cuda()

    # Run inference before rotation
    input_ids = torch.randint(0, 100, (1, 10)).cuda()
    output = model(input_ids)[0]

    # Apply rotation
    print(f"Applying rotation with config: {args.config}...")
    model = mtq.rotate(model, args.config)

    # Run inference after rotation
    output_after = model(input_ids)[0]

    # Verify outputs
    max_diff = torch.abs(output - output_after).max().item()
    max_rel_diff = max_diff / output.abs().max().item()
    print(f"Output max diff: {max_diff}, max rel diff: {max_rel_diff}")

    try:
        assert torch.allclose(output, output_after, atol=1e-3, rtol=1e-3)
        print("✓ Rotation successful - outputs match within tolerance")
    except AssertionError:
        print("⚠ Warning: Outputs differ more than expected tolerance")

    # Save model if output path is provided
    if args.output:
        print(f"Saving rotated model to {args.output}...")
        model.save_pretrained(args.output)
        print("✓ Model saved successfully")


if __name__ == "__main__":
    main()
