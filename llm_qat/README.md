# Quantization Aware Training (QAT)

QAT helps to improve the model accuracy beyond post training quantization (PTQ).

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
# mto.restore_from_modelopt_state(model, torch.load("modelopt_quantizer_states.pt"))
# After loading the quantizer states, load the model weights
# model.load_state_dict(state_dict_from_last_checkpoint)

trainer.train()  # Train the quantized model (i.e, QAT)

# Save the final model weights; An example usage
trainer.save_model()
```

> **_NOTE:_** The example above uses [mto.modelopt_state](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.opt.conversion.html#modelopt.torch.opt.conversion.modelopt_state) and [mto.restore_from_modelopt_state](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.opt.conversion.html#modelopt.torch.opt.conversion.restore_from_modelopt_state) for saving and restoring of ModelOpt
> modified model. ModelOpt provides additional methods/workflows for saving and restoring ModelOpt modified model. Please see [saving & restoring](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/6_save_load.html) to learn more.

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
> hyper-parameters such as learning rate, training duration etc.

#### Results

The result from performing the above experiments is tabulated below (You could get slightly different numbers). As we can see below, QAT has improved the validation perplexity and MMLU over PTQ (lower perplexity & higher MMLU are better).

|                 |  Validation perplexity on `Samsung/samsum` dataset |
|-----------------|--------------------|
| Fine-tuned BF16 (No quantization) | 2.71  |
| PTQ with INT4 weights & INT8 activations on the Fine-tuned BF16 model    | 188.48  |
| QAT with INT4 weights & INT8 activations on Fine-tuned BF16 model         | 4.90  |

#### Deployment

The final model after QAT is similar in architecture to that of PTQ model. QAT model simply have updated weights as compared to the PTQ model. It can be deployed to TensorRT-LLM (TRTLLM) or to TensorRT just like a regular **ModelOpt** PTQ model if the quantization format is supported for deployment. See more details on deployment of quantized model to TRTLLM [here](../llm_ptq/README.md).

## Other Examples

### End-to-end QLoRA with Real Quantization

[QLoRA](https://arxiv.org/pdf/2305.14314) is a technique mainly intended for further reducing the training memory requirement of LoRA. In QLoRA, the LoRA backbone weights are quantized to reduce the model footprint. Unlike QAT which uses simulated quantization, QLoRA requires real quantization. Currently, only NF4_REAL_QUANT_CFG and INT4_AWQ_REAL_QUANT_CFG are supported.

In this example, the model is trained on [Samsung/samsum](https://huggingface.co/datasets/Samsung/samsum) dataset.

To evaluate QLoRA quantized model before training, run:

```sh
# Load the HF checkpoint, quantize the model and evaluate without additional training
./launch.sh --model meta-llama/Llama-2-7b-hf \
   --do_train False \
   --quant_cfg 'NF4_REAL_QUANT_CFG'
```

To perform QLoRA training, run:

```sh
# Load the HF checkpoint, quantize the model, add LoRA adapter, and run additional training
./launch.sh --model meta-llama/Llama-2-7b-hf \
   --num_epochs 0.5 \
   --lr 1e-3 \
   --do_train True \
   --output_dir llama2-nf4-qlora \
   --quant_cfg 'NF4_REAL_QUANT_CFG' \
   --lora True
```

> **_NOTE:_** QLoRA is currently an experimental feature designed to reduce the memory footprint during training. Deployment functionality is not yet available.

##### Results

The result from performing the above experiments is tabulated below (You could get slightly different numbers). As we can see below, QLoRA achieves
similar validation perplexity as that of BF16 finetuning while consuming significantly lower GPU memory.

|                 |  Validation perplexity on `Samsung/samsum` dataset |
|-----------------|--------------------|
| Fine-tuned BF16 (No quantization) | 2.71  |
| NF4 quantization    | 4.28  |
| NF4 QLoRA after training    | 2.90  |

### End-to-end Medusa QAT Example

[Medusa](https://github.com/FasterDecoding/Medusa) is a simple framework that democratizes the acceleration techniques for LLM generation with multiple decoding heads. It adds extra "heads" to LLMs to predict multiple future tokens simultaneously. During generation, these heads each produce multiple likely words for the corresponding position. These options are then combined and processed using a tree-based attention mechanism. Finally, a typical acceptance scheme is employed to pick the longest plausible prefix from the candidates for further decoding. ModelOpt supports quantizing the medusa model to further speed up the inference. This PTQ medusa model can be fine-tuned to improve accuracy as well as performance since medusa speedup depends on its heads' accuracy.

First, we need to run un-quantized fine-tuning of both base model and medusa heads. Here is the command for that:

```sh
./launch.sh --model meta-llama/Llama-2-7b-hf \
   --num_epochs 5.0 \
   --lr 1e-5 \
   --do_train True \
   --output_dir llama2-medusa-finetune \
   --medusa True \
   --only_medusa_heads False \
   --num_medusa_heads 2 --num_medusa_layers 1
```

This will generate a fine-tuned medusa checkpoint in `output_dir` specified above. Next we perform PTQ and fine-tune the quantized medusa model. Note, by setting the --only_medusa_heads flag to True, we will freeze the base model and only fine-tune the medusa heads. This may be helpful to maintain the base model distribution.

```sh
# Load the checkpoint from previous fine-tuning stage, quantize the model and run additional training (QAT)
./launch.sh --model llama2-medusa-finetune \
   --num_epochs 0.5 \
   --lr 1e-5 \
   --do_train True \
   --output_dir llama2-medusa-qat \
   --quant_cfg 'INT4_WEIGHT_INT8_ACTIVATIONS' \
   --medusa True \
   --only_medusa_heads True \
   --num_medusa_heads 2 --num_medusa_layers 1
```
