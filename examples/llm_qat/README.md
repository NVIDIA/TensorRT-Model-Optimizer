# Quantization Aware Training (QAT)

QAT helps to improve the model accuracy beyond post training quantization (PTQ).
To learn more about the QAT feature, please refer to the [documentation](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_pytorch_quantization.html#quantization-aware-training-qat).

In this example, QAT workflow is demonstrated for NVIDIA NeMo and Huggingface text generation model for supervised fine-tuning (SFT).

## NVIDIA NeMo QAT

Please refer to the NeMo [QAT playbook](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html#deployment) for an example on how to do QAT on a NeMo Supervised Fine-Tuned (SFT) model.

## Hugging Face QAT

### System Requirements

The Llama2-7B fine-tuning and QAT below requires a minimum of 2 x 80GB GPUs per machine.

### QAT

In QAT, a model quantized using [mtq.quantize()](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html#modelopt.torch.quantization.model_quant.quantize) can be directly fine-tuned with the original training pipeline. During QAT, the scaling factors inside quantizers are frozen and the model weights are fine-tuned.

#### QAT Example Workflow

Here is the recommended QAT workflow:

Step 1: Train/fine-tune the model in the original precision without quantization.

Step 2: Quantize the model from step 1 with `mtq.quantize()`

Step 3: Train/fine-tune the quantized model with a small learning rate, e.g. 1e-5 for Adam optimizer.

> **_NOTE:_** `Step 3` listed above is the actual 'Quantization Aware Training' step. The optimal hyperparameter setting for QAT can vary depending on the model and training dataset. The optimal QAT duration depends on the dataset, model etc.
> In general, QAT with much shorter than the original pre-training steps is sufficient to recover the accuracy. For LLMs in particular, we find that QAT with number of tokens in the order of 100s of millions are often sufficient to recover most of the accuracy.

> **_TIP:_** We find QAT without the original precision training/fine-tuning (i.e skipping `Step 1` of the QAT workflow from above) to give worser accuracy. Therefore, we recommend un-quantized original precision training/fine-tuning followed by QAT for best accuracy.

> **_NOTE:_** Huggingface models trained with modelopt.torch.speculative (mtsp) can be used in QAT directly like regular Huggingface models.

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

# To resume training from a checkpoint or load the final QAT model for evaluation,
# load the quantizer states before loading the model weights
# mto.restore_from_modelopt_state(model, torch.load("modelopt_quantizer_states.pt", weights_only=False))
# After loading the quantizer states, load the model weights
# model.load_state_dict(state_dict_from_last_checkpoint)

trainer.train()  # Train the quantized model (i.e, QAT)

# Save the final model weights; An example usage
trainer.save_model()
```

> **_NOTE:_** The example above uses [mto.modelopt_state](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.opt.conversion.html#modelopt.torch.opt.conversion.modelopt_state) and [mto.restore_from_modelopt_state](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.opt.conversion.html#modelopt.torch.opt.conversion.restore_from_modelopt_state) for saving and restoring of ModelOpt
> modified model. ModelOpt provides additional methods/workflows for saving and restoring ModelOpt modified model. Please see [saving & restoring](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/6_save_load.html) to learn more.

> **_NOTE:_** ModelOpt provides accelerated quantization kernels using Triton that significantly speed up NVFP4 format QAT. For details, see the [installation guide](https://nvidia.github.io/TensorRT-Model-Optimizer/getting_started/_installation_for_Linux.html#accelerated-quantization-with-triton-kernels).

#### End-to-end QAT Example

This folder contains end-to-end runnable fine-tuning/QAT pipeline where Llama2-7B from huggingface is trained on
[Samsung/samsum](https://huggingface.co/datasets/Samsung/samsum) dataset.

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
   --quant_cfg INT4_WEIGHT_INT8_ACTIVATIONS
```

To perform QAT, run:

```sh
# Load the quantized checkpoint from previous fine-tuning stage and run additional training (QAT)
./launch.sh --model llama2-finetune \
   --num_epochs 0.5 \
   --lr 1e-5 \
   --do_train True \
   --output_dir llama2-qat
```

You may alternatively perform QAT with any other quantization formats from **ModelOpt**. Please see more details on the supported quantization formats and how to use them as shown below:

```python
import modelopt.torch.quantization as mtq

# To learn about the quantization formats and quantization config from modelopt
help(mtq.config)
```

> **_NOTE:_** QAT requires higher memory than the full-precision fine-tuning. A solution to avoid this extra memory usage is to use [activation checkpointing](https://pytorch.org/docs/stable/checkpoint.html) or gradient checkpointing. Activation checkpointing can be enabled easily with training frameworks such as Huggingface by adding an additional argument `gradient_checkpointing True`. Learn more [here](https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#gradient-checkpointing). Activation checkpointing or gradient checkpointing is enabled by default in this example.

> **_NOTE:_** Like any other model training, the QAT model accuracy can be further improved by optimizing the training
> hyper-parameters such as learning rate, training duration etc.

> **_NOTE:_** `launch.sh` defaults to use `LlamaDecoderLayer` as the transformer layer class. If your model uses a different class, you can pass `--fsdp_transformer_layer_cls_to_wrap <your_layer_class>` to the `launch.sh` script.

#### Results

The result from performing the above experiments is tabulated below (You could get slightly different numbers). As we can see below, QAT has improved the validation perplexity and MMLU over PTQ (lower perplexity & higher MMLU are better).

| | Validation perplexity on `Samsung/samsum` dataset |
|-----------------|--------------------|
| Fine-tuned BF16 (No quantization) | 2.71 |
| PTQ with INT4 weights & INT8 activations on the Fine-tuned BF16 model | 188.48 |
| QAT with INT4 weights & INT8 activations on Fine-tuned BF16 model | 4.90 |

To illustrate the efficiency of QAT, we share another experiment of [`meta-llama/Meta-Llama-3-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B) with [`nvidia/Daring-Anteater`](https://huggingface.co/datasets/nvidia/Daring-Anteater) dataset and `NVFP4` format. The result is as below:

| | Validation perplexity on `nvidia/Daring-Anteater` dataset |
|-----------------|--------------------|
| Fine-tuned BF16 (No quantization) (1 stage) | 1.56 |
| Fine-tuned BF16 (No quantization) (2 stages) | 1.53 |
| PTQ with NVFP4 weights & NVFP4 activations on the Fine-tuned BF16 model (2 stages) | 1.62 |
| QAT with NVFP4 weights & NVFP4 activations | 1.58 |

To fairly compare the PTQ and QAT, the fine-tuning in this experiment also has two stages so that the training steps are consistent between PTQ and QAT, the detailed recipe is:

- Finetuning (1 stage): "1000 steps fine-tuning with `lr = 1e-5`"
- Finetuning (2 stages): "1000 steps fine-tuning with `lr = 1e-5`" + "1000 steps fine-tuning with `lr = 3e-6`"
- QAT: "1000 steps fine-tuning with `lr = 1e-5`" + "1000 steps QAT with `lr = 3e-6`"

> **_NOTE:_** From our experience, the QAT performs better with a larger batch size, so we recommend using a larger batch size if your hardware allows it.

> **_NOTE:_** If you only use part of the dataset for fine-tuning/QAT, we recommend to use different data samples for fine-tuning and QAT, otherwise there may appear overfitting issues during the QAT stage.

#### Deployment

The final model after QAT is similar in architecture to that of PTQ model. QAT model simply have updated weights as compared to the PTQ model. It can be deployed to TensorRT-LLM (TRTLLM) or to TensorRT just like a regular **ModelOpt** PTQ model if the quantization format is supported for deployment. See more details on deployment of quantized model to TRTLLM [here](../llm_ptq/README.md).

## Other Examples

### End-to-end QLoRA with Real Quantization

[QLoRA](https://arxiv.org/pdf/2305.14314) is a technique mainly intended for further reducing the training memory requirement of LoRA. In QLoRA, the LoRA backbone weights are quantized to reduce the model footprint. Unlike QAT which uses simulated quantization, QLoRA requires real quantization. To compress the model weights after quantization, we use the `mtq.compress()` function, which currently supports FP8, FP4, and INT4 formats. This feature can be enabled by passing `--compress True` to the `launch.sh` script. For detailed configuration options and patterns, please refer to the `modelopt.torch.quantization.compress` documentation.

In this example, the model is trained on [Samsung/samsum](https://huggingface.co/datasets/Samsung/samsum) dataset.

To evaluate QLoRA quantized model before training, run:

```sh
# Load the HF checkpoint, quantize the model and evaluate without additional training
# Also compress the model after quantization
./launch.sh --model meta-llama/Llama-2-7b-hf \
   --do_train False \
   --quant_cfg NVFP4_DEFAULT_CFG \
   --compress True
```

To perform QLoRA training, run:

```sh
# Load the HF checkpoint, quantize the model, add LoRA adapter, and run additional training
# Also compress the model after quantization
./launch.sh --model meta-llama/Llama-2-7b-hf \
   --num_epochs 0.5 \
   --lr 1e-3 \
   --do_train True \
   --output_dir llama2-fp4-qlora \
   --quant_cfg NVFP4_DEFAULT_CFG \
   --compress True \
   --lora True
```

> **_NOTE:_** QLoRA is currently an experimental feature designed to reduce the memory footprint during training. Deployment functionality is not yet available.

##### Results

The result from performing the above experiments is tabulated below (You could get slightly different numbers). As we can see below, QLoRA achieves
similar validation perplexity as that of BF16 finetuning while consuming significantly lower GPU memory.

| | Validation perplexity on `Samsung/samsum` dataset |
|-----------------|--------------------|
| Fine-tuned BF16 (No quantization) | 2.71 |
| NF4 quantization | 4.28 |
| NF4 QLoRA after training | 2.90 |
