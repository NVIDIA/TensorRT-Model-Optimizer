# Speculative Decoding

Large Language Models (LLMs) have demonstrated remarkable capabilities and are increasingly applied in various domains. However, their text generation process is costly and slow. This inefficiency is attributed to the nature of auto-regressive decoding: each token generation necessitates a forward pass, requiring access to the entire parameter set of the LLM. This results in a memory-bound limitation for auto-regressive decoding. To accelerate auto-regressive decoding, speculative decoding methods use a draft model (either a smaller model or the LLM itself) to guess the next γ tokens through standard auto-regressive generation. Subsequently, the original LLM validates these guessed tokens, necessitating only a single forward pass for verification. If the draft model accurately predicts α tokens, a single forward pass of the original LLM can generate α+1 tokens.

To learn more about the speculative decoding feature, please refer to the [documentation](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/7_speculative_decoding.html).

In this example, the end-to-end workflow of speculative decoding is demonstrated for a pretrained HF text generation model.

## End-to-end Speculative Decoding Fine-tuning

### Prepare Data

In speculative decoding fine-tuning, extra speculative decoding module, like Medusa heads or EAGLE module, are added to the base model to predict the next γ tokens. These tokens will then be validated by the original LLM. In order for these predicted tokens to be accepted by the original LLM, their prediction distributions should be similar to that of the base model. Therefore, we need to prepare fine-tuning data generated from the original LLM. Start by launching an inference server that will run the base model. Let us use TinyLlama/TinyLlama-1.1B-Chat-v1.0 as an example.

First, set up a vllm server with TinyLlama. Make sure to use a different docker container other than the one for training as installing vllm may cause version conflicts with modelopt. Note: for quantized models by ModelOpt, you need to add --quantization=modelopt flag.

```sh
pip install vllm
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --api-key token-abc123 --port 8000  --tensor-parallel-size 1
```

Then, we adapt the fine-tuning data by calling this server. In this example, we use Daring-Anteater dataset.

```sh
git clone https://huggingface.co/datasets/nvidia/Daring-Anteater
python3 server_generate.py --data_path Daring-Anteater/train.jsonl --output_path finetune/data.jsonl --max_token 512 --chat
```

To add a system prompt, use the `--system_prompt` argument:

```sh
python3 server_generate.py --data_path Daring-Anteater/train.jsonl --output_path finetune/data.jsonl --max_token 512 --chat --system_prompt <system_prompt_text>
```

#### SLURM Prepare Data

For basic parallelization of synthetic data generation we provide some SLURM support.
Assuming a `$SLURM_JOB_ID` is present and nodes, n1, n2, n3, n4 are selected the following is achieveable.

Example of allocating 4 nodes for 120 minutes

```sh
salloc  -N4 -A <account> -p <partition>  -J <account>-synthetic:data-gen -t 120
```

Create shards of some given size

```sh
python3 distributed_generate/sharding_utils.py --input_path /data/train.jsonl --output_dir /data/train/ --max_lines_per_shard 10000
```

Run workers on SLURM

```sh
bash distributed_generate/launch.sh $SLURM_JOB_ID vllm TinyLlama/TinyLlama-1.1B-Chat-v1.0 /data/train/ /data/output /scripts/ 0 10 n1,n2,n3,n4 "\"You are a helpful assistant.\""
```

`/scripts/` is the absolute path to `modelopt/examples/speculative_decoding` which contains `server_generate.py` and `distributed_generate`.
This will launch a vllm server (sglang is also available) on each node. Each node will work through 10 shards of data (10\*max_lines_per_shard number of samples).
In this case, the first 40 shards of data will be processed.
To process the next 40 shards

```sh
bash distributed_generate/launch.sh $SLURM_JOB_ID vllm TinyLlama/TinyLlama-1.1B-Chat-v1.0 /data/train/ /data/output /scripts/ 40 10 n1,n2,n3,n4
```

To combine the shards back

```sh
python3 distributed_generate/sharding_utils.py --input_dir /data/output/ --output_path /data/output.jsonl --combine
```

### Speculative Decoding Example Training Workflow

Here is the recommended end-to-end speculative decoding training workflow:

```python
import os
import torch
import transformers
import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp

# Create a base model
model = transformers.AutoModelForCausalLM.from_pretrained(
    "<path to your pretrained model>",
)

if mode == "medusa":
    config = {
        "medusa_num_heads": 2,
        "medusa_num_layers": 1,
    }
elif mode == "eagle":
    config = {
        "eagle_num_layers": 1,
        "use_input_layernorm_in_first_layer": True,
        "use_last_layernorm": False
    }
mtsp.convert(model, [(mode, config)])

for name, param in model.named_parameters():
    if not name.startswith(("medusa", "eagle")):
        param.requires_grad = False

tokenizer = transformers.AutoTokenizer.from_pretrained(ckpt_path)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Create a trainer
trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
trainer._move_model_to_device(model, trainer.args.device)

# Enable HF checkpointing so that the saved model will contain the speculative decoding module
mto.enable_huggingface_checkpointing()

trainer.train(resume_from_checkpoint=checkpoint)
trainer.save_state()
trainer.save_model("<path to the output directory>")
```

### End-to-end Speculative Decoding Example

This folder contains end-to-end runnable speculative decoding fine-tuning pipeline where TinyLlama from huggingface is trained on Daring-Anteater dataset.

First, download the data:

```sh
git clone https://huggingface.co/datasets/nvidia/Daring-Anteater
```

Then, prepare the synthesized data from the base model. Please refer to the **Prepare Data** section.

Next, we fine-tune the speculative decoding models with the base model frozen. Here is the command for Medusa and EAGLE:

```sh
./launch.sh --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
            --data finetune/data.jsonl \
            --mode medusa \
            --num_epochs 1 --lr 1e-5 --save_steps 1000 \
            --output_dir medusa-tinyllama \
            --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
            --num_gpu 1 --train_bs 4 \
            --medusa_num_heads 2 --medusa_num_layers 1 \
            --freeze_base_model True

./launch.sh --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
            --data finetune/data.jsonl \
            --mode eagle \
            --num_epochs 1 --lr 1e-5 --save_steps 1000 \
            --output_dir eagle-tinyllama \
            --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
            --num_gpu 1 --train_bs 4 \
            --eagle_num_layers 1 \
            --freeze_base_model True
```

This will generate fine-tuned checkpoints in `output_dir` specified above.

Alternatively, you can refer to this [notebook](example.ipynb).

### Deployment

The final model after end-to-end speculative decoding fine-tuning is similar in architecture to HF models. It can be further optimized through **ModelOpt**, e.g., PTQ and QAT. It can be deployed to TensorRT-LLM (TRTLLM) or to TensorRT just like a regular **ModelOpt** model. See more details on deployment of quantized model to TRTLLM [here](../llm_ptq/README.md).

## Model Support List

Model | Medusa | EAGLE
--- | --- | ---
LLAMA 2 | Yes | Yes
LLAMA 3, 3.1 | Yes | Yes
Mistral | Yes | Yes
Phi 3 | Yes | Yes
QWen 1.5,2,2.5 | Yes | Yes
