from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import modelopt.torch.quantization as mtq

MAX_SEQ_LEN = 2048
MAX_OUTPUT_LEN = 512


def get_tokenizer(ckpt_path, max_seq_len=MAX_SEQ_LEN):
    """Returns the tokenizer from the model ckpt_path."""
    print(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path, model_max_length=max_seq_len, padding_side="left", trust_remote_code=True
    )
    if type(tokenizer).__name__ == "QWenTokenizer":
        # qwen use token id 151643 as pad and eos tokens
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(151643)

    # can't set attribute 'pad_token' for "<unk>"
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def _get_calib_dataloader(
    data="cnn_dailymail", tokenizer=None, batch_size=1, calib_size=512, block_size=512, device=None
):
    print("Loading calibration dataset")
    if data == "pileval":
        dataset = load_dataset(
            "json", data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst", split="train"
        )
        dataset = dataset["text"][:calib_size]
    elif data == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        dataset = dataset["article"][:calib_size]
    else:
        raise NotImplementedError

    batch_encoded = tokenizer.batch_encode_plus(
        dataset, return_tensors="pt", padding=True, truncation=True, max_length=block_size
    )
    if device:
        batch_encoded = batch_encoded.to(device)
    batch_encoded = batch_encoded["input_ids"]

    calib_dataloader = DataLoader(batch_encoded, batch_size=batch_size, shuffle=False)

    return calib_dataloader


def _quantize_model_with_dataset(lm, quant_cfg: str, calib_dataset):
    atq_cfg = getattr(mtq, quant_cfg)

    if hasattr(lm, "gpt2"):
        net = lm.gpt2
    else:
        net = lm.model

    def calibrate_loop(model):
        print("Calibrating model...")
        for data in calib_dataset:
            model(data)
        print("Calibration complete.")

    net = mtq.quantize(net, atq_cfg, calibrate_loop)
    # Fold weights for faster evaluation.
    mtq.fold_weight(net)

    if hasattr(lm, "gpt2"):
        lm.gpt2 = net
    else:
        lm.model = net


def quantize_model(model, quant_cfg: str, tokenizer, batch_size, calib_size, data="cnn_dailymail"):
    """Quantizes the model with the provided calibration dataset.

    Args:
        model: the model to be quantized.
        quant_cfg: the quantization algorithm config name.
        tokenizer: the tokenizer.
        batch_size: the calibration batch size for each calibration inference run.
        calib_size: the total calibration dataset size.
        data: the name of the calibration dataset.
    """
    if "AWQ" in quant_cfg:
        if calib_size > 32:
            print(
                f"AWQ calibration could take longer with calib_size = {calib_size}, Using"
                " calib_size=32 instead"
            )
            calib_size = 32
        print(
            "\nAWQ calibration could take longer than other calibration methods. Please"
            " increase the batch size to speed up the calibration process. Batch size can"
            " be set by adding the argument --batch_size <batch_size> to the command line."
        )

    device = model.device
    if hasattr(model, "model"):
        device = model.model.device

    calib_dataloader = _get_calib_dataloader(
        data=data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        calib_size=calib_size,
        device=device,
    )

    _quantize_model_with_dataset(model, quant_cfg, calib_dataloader)
