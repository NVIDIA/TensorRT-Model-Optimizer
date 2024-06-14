# Quantization Aware Training (QAT) for Hugging Face (HF) Models

QAT helps to improve the model accuracy beyond post training quantization (PTQ).

In this example, QAT workflow is demonstrated for a HF text generation model supervised fine-tuning (SFT) on a summarization task.

## System Requirements

The Llama2-7B fine-tuning and QAT below requires a minimum of 2 80GB GPUs per machine.

## Setup

Install necessary modules

```sh
pip install -r requirements.txt
```

## QAT

In QAT, a model quantized using `mtq.quantize()` can be directly fine-tuned with the original training pipeline. During QAT, the scaling factors inside quantizers are frozen and the model weights are fine-tuned.

### QAT Example Workflow

Here is the recommended QAT workflow:

1. Un-quantized training/fine-tuning, e.g. BF16 fine-tuning without any quantization.
1. QAT for 1-10% original training duration and a small learning rate, e.g. 1e-5 for Adam optimizer

Doing QAT without the original un-quantized training/fine-tuning is found to have worser accuracy. Therefore, we recommend un-quantized training/fine-tuning followed by QAT for best convergence.

Here is an example code for performing QAT:

```python
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

...

# [Not shown] load model, tokenizer, data loaders etc
trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)


def forward_loop(model):
    for i, data in enumerate(calib_dataloader):
        model(data)


# Quantize the model in-place; The model should be unwrapped from any distributed wrapper
# The model may be wrapped in a DataParallel or DistributedDataParallel after `mtq.quantize`
model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop)

# Save the modelopt quantizer states
torch.save(mto.modelopt_state(model), "modelopt_quantizer_states.pt")

# To resume training from a checkpoint or load the final QAT model,
# load the quantizer states before loading the model weights
# mto.restore_from_modelopt_state(model, torch.load("modelopt_quantizer_states.pt"))

trainer.train()  # Train the quantized model (i.e, QAT)

# Save the final model weights; An example usage
trainer.save_model()
```

### End-to-end QAT Example

This folder contains end-to-end runnable fine-tuning/QAT pipeline where Llama2-7B from huggingface is trained on
SAMSum dataset.

First, we need to run un-quantized fine-tuning. Here is the command for that:

```sh
./launch.sh --model meta-llama/Llama-2-7b-hf \
   --num_epochs 5.0 \
   --lr 1e-5 \
   --do_train True \
   --output_dir llama2-finetune
```

This will generate a fine-tuned checkpoint in `output_dir` specified above. You can load this checkpoint, quantize the model, evaluate PTQ results or run additional QAT.
This can be accomplished by specifying the quantization format to the `launch.sh` script.
In this example, we are quantizing the model with INT4 block-wise weights and INT8 per-tensor activation quantization.

To perform PTQ evaluation, run:

```sh
# Load the checkpoint from previous fine-tuning stage, quantize the model and evaluate without additional training
./launch.sh --model llama2-finetune \
   --do_train False \
   --quant_cfg 'INT4_WEIGHT_INT8_ACTIVATIONS'
```

To perform QAT, run:

```sh
# Load the checkpoint from previous fine-tuning stage, quantize the model and run additional training (QAT)
./launch.sh --model llama2-finetune \
   --num_epochs 0.5 \
   --lr 1e-5 \
   --do_train True \
   --output_dir llama2-qat \
   --quant_cfg 'INT4_WEIGHT_INT8_ACTIVATIONS'
```

You may alternatively perform QAT with any other quantization formats from **ModelOpt**. Please see more details on the supported quantization formats and how to use them as shown below:

```python
import modelopt.torch.quantization as mtq

# To learn about the quantization formats and quantization config from modelopt
help(mtq.config)
```

> **_NOTE:_**  QAT requires higher memory than the full-precision fine-tuning. A solution to avoid this extra memory usage is to use [activation checkpointing](https://pytorch.org/docs/stable/checkpoint.html) or gradient checkpointing. Activation checkpointing can be enabled easily with training frameworks such as Huggingface by adding an additional argument `gradient_checkpointing True`. Learn more [here](https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#gradient-checkpointing). Activation checkpointing or gradient checkpointing is enabled by default in this example.

> **_NOTE:_** Like any other model training, the QAT model accuracy can be further improved by optimizing the training
> hyper-parameters such as learning rate, training duration etc. For example in the fine-tuning example above, QAT for training duration longer than 10% of the original training duration improves the model accuracy further.

#### Results

The result from performing the above experiments is tabulated below (You could get slightly different numbers). As we can see below, QAT has significantly improved the validation perplexity over PTQ alone (lower perplexity is better).

|                 |  Validation perplexity   |
|-----------------|--------------------|
| Fine-tuned BF16 (No quantization) | 2.71  |
| PTQ with INT4 weights & INT8 activations on the Fine-tuned BF16 model    | 188.48  |
| QAT with INT4 weights & INT8 activations on Fine-tuned BF16 model        | 4.90  |

#### Deployment

The final model after QAT is similar in architecture to that of PTQ model. QAT model simply have updated weights as compared to the PTQ model. It can be deployed to TensorRT-LLM (TRTLLM) or to TensorRT just like a regular **ModelOpt** PTQ model if the quantization format is supported for deployment. See more details on deployment of quantized model to TRTLLM [here](../llm_ptq/README.md).
