# Speculative Decoding

Large Language Models (LLMs) have demonstrated remarkable capabilities and are increasingly applied in various domains. However, their text generation process is costly and slow. This inefficiency is attributed to the nature of auto-regressive decoding: each token generation necessitates a forward pass, requiring access to the entire parameter set of the LLM. This results in a memory-bound limitation for auto-regressive decoding. To accelerate auto-regressive decoding, speculative decoding methods use a draft model (either a smaller model or the LLM itself) to guess the next γ tokens through standard auto-regressive generation. Subsequently, the original LLM validates these guessed tokens, necessitating only a single forward pass for verification. If the draft model accurately predicts α tokens, a single forward pass of the original LLM can generate α+1 tokens.

In this example, Medusa, one of the speculative decoding methods, end-to-end workflow is demonstrated for a pretrained HF text generation model.

## End-to-end Medusa Fine-tuning

### Prepare data

In Medusa fine-tuning, multiple medusa heads are added to the base model to predict the next γ tokens. These tokens will then be validated by the original LLM. In order for these predicted tokens to be accepted by the original LLM, their prediction distributions should be similar to that of the base model. Therefore, we need to prepare fine-tuning data generated from the original LLM. Start by lanuching an inference server that will run the base model. Let us use TinyLlama/TinyLlama-1.1B-Chat-v1.0 as an example.

```sh
bash prepare_data.sh --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --server True
```

We use Daring-Anteater dataset in this example. We adapt the fine-tuning data by calling this server:

```sh
git clone https://huggingface.co/datasets/nvidia/Daring-Anteater
bash prepare_data.sh --data_path Daring-Anteater/train.jsonl --output_path finetuning/data.jsonl
```

#### End-to-end Medusa Example Workflow

Here is the recommended End-to-end Medusa workflow:

```python
import os
import torch
import transformers
import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp

# Create a Medusa model
model = transformers.AutoModelForCausalLM.from_pretrained(
    "<path to your pretrained model>",
)
config = {
    "medusa_num_heads": 2,
    "medusa_num_layers": 1,
}
mtsp.convert(model, [("medusa", config)])
model.generation_config.do_sample = True

tokenizer = transformers.AutoTokenizer.from_pretrained(ckpt_path)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Create a trainer and update the compute_loss for Medusa
trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
trainer._move_model_to_device(model, trainer.args.device)
mtsp.plugins.transformers.replace_medusa_compute_loss(trainer, medusa_only_heads=True)

mto.enable_huggingface_checkpointing()

trainer.train(resume_from_checkpoint=checkpoint)
trainer.save_state()
trainer.save_model("<path to the output directory>")
```

#### End-to-end Medusa Example

This folder contains end-to-end runnable medusa fine-tuning pipeline where TinyLlama from huggingface is trained on ShareGPT dataset.

First, we need to download the data:

```sh
git clone https://huggingface.co/datasets/nvidia/Daring-Anteater
```

Then, we need to fine-tune both base model and medusa heads. Here is the command for that:

```sh
./launch.sh --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
            --data Daring-Anteater/train.jsonl \
            --mode medusa \
            --num_epochs 0.1 --lr 1e-5 --save_steps 100 \
            --output_dir medusa-tinyllama --only_medusa_heads False \
            --heads 2 --layers 1
```

This will generate a fine-tuned checkpoint in `output_dir` specified above.

#### Deployment

The final model after End-to-end Medusa fine-tuning is similar in architecture to HF models. It can be further optimized through **ModelOpt**, e.g., PTQ and QAT. It can be deployed to TensorRT-LLM (TRTLLM) or to TensorRT just like a regular **ModelOpt** model. See more details on deployment of quantized model to TRTLLM [here](../llm_ptq/README.md).
