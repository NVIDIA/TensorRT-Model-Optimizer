# Quantization Aware Training (QAT)

QAT helps to improve the model accuracy beyond post training quantization (PTQ).

In QAT, a model quantized using [mtq.quantize()](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html#modelopt.torch.quantization.model_quant.quantize) can be directly fine-tuned with the original training pipeline. During QAT, the scaling factors inside quantizers are frozen and the model weights are fine-tuned.

To learn more about the QAT feature, please refer to the [documentation](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_pytorch_quantization.html#quantization-aware-training-qat).

> **_NOTE:_** In this example, a QAT workflow is demonstrated for Huggingface text generation model for supervised fine-tuning (SFT). However, the workflow is general and
> be extended to frameworks such as [NeMo](https://docs.nvidia.com/nemo-framework/user-guide/latest/model-optimization/quantization/quantization.html) and models beyond LLMs such as CNN based vision models.

### System Requirements

The Llama3-8B fine-tuning and QAT below requires a minimum of 2 x 80GB GPUs per machine.

### QAT

Here is a recommended QAT workflow:

Step 1: Train/fine-tune the model in the original precision without quantization.

Step 2: Quantize the model from step 1 with `mtq.quantize()`

Step 3: Train/fine-tune the quantized model with a small learning rate, e.g. 1e-5 for Adam optimizer.

> **_NOTE:_** `Step 3` listed above is the actual 'Quantization Aware Training' step. The optimal hyperparameter setting for QAT can vary depending on the model and training dataset. The optimal QAT duration depends on the dataset, model etc.

> **_NOTE:_** We find QAT without the original precision training/fine-tuning (i.e skipping `Step 1` of the QAT workflow from above) to give worser accuracy. Therefore, we recommend un-quantized original precision training/fine-tuning followed by QAT for best accuracy.

> **_TIP:_** Huggingface models trained with modelopt.torch.speculative (mtsp) can be used in QAT directly like regular Huggingface models.

Here is an example code for performing QAT:

```python
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

...

# [Not shown] load model, tokenizer, data loaders etc
trainer = Trainer(model=model, processing_class=tokenizer, args=training_args, **data_module)


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

This folder contains end-to-end runnable fine-tuning/QAT pipeline where Llama3-8B from huggingface is trained on
[Daring-Anteater](https://huggingface.co/datasets/nvidia/Daring-Anteater) dataset.

First, we need to run un-quantized fine-tuning. Here is the command for that:

```sh
./launch.sh --model meta-llama/Meta-Llama-3-8B \
   --num_epochs 2.0 \
   --lr 1e-5 \
   --do_train True \
   --output_dir llama3-finetune
```

This will generate a fine-tuned checkpoint in `output_dir` specified above. You can load this checkpoint, quantize the model, evaluate PTQ results or run additional QAT.
This can be accomplished by specifying the quantization format to the `launch.sh` script.
In this example, we are quantizing the model with INT4 block-wise weights and INT8 per-tensor activation quantization.

To perform PTQ evaluation, run:

```sh
# Load the checkpoint from previous fine-tuning stage, quantize the model and evaluate without additional training
./launch.sh --model llama3-finetune \
   --do_train False \
   --quant_cfg NVFP4_DEFAULT_CFG
```

To perform QAT, run:

```sh
# Load the quantized checkpoint from previous fine-tuning stage and run additional training (QAT)
./launch.sh --model llama3-finetune \
   --num_epochs 2.0 \
   --lr 1e-5 \
   --do_train True \
   --quant_cfg NVFP4_DEFAULT_CFG \
   --output_dir llama3-qat
```

You may alternatively perform QAT with any other quantization formats from **ModelOpt**. Please see more details on the supported quantization formats and how to use them as shown below:

```python
import modelopt.torch.quantization as mtq

# To learn about the quantization formats and quantization config from modelopt
help(mtq.config)
```

You could also add your own customized quantization format to `CUSTOM_QUANT_CFG` from `main.py` and perform QAT.

> **_NOTE:_** QAT requires higher memory than the full-precision fine-tuning. A solution to avoid this extra memory usage is to use [activation checkpointing](https://pytorch.org/docs/stable/checkpoint.html) or gradient checkpointing. Activation checkpointing can be enabled easily with training frameworks such as Huggingface by adding an additional argument `gradient_checkpointing True`. Learn more [here](https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#gradient-checkpointing). Activation checkpointing or gradient checkpointing is enabled by default in this example.

> **_NOTE:_** Like any other model training, the QAT model accuracy can be further improved by optimizing the training
> hyper-parameters such as learning rate, training duration etc.

> **_NOTE:_** `launch.sh` defaults to use `LlamaDecoderLayer` as the transformer layer class. If your model uses a different class, you can pass `--fsdp_transformer_layer_cls_to_wrap <your_layer_class>` to the `launch.sh` script.

> **_NOTE:_** The script defaults to using FSDP1. To use FSDP2, pass "--use_fsdp2 True" to the `launch.sh` script. Note that FSDP2 is less stable than FSDP1 currently. Use it with caution.

#### Results

Here is an example result following the workflow above with slightly different hyper-parameters (We used an effective batch size of 128 by adjusting `--train_bs` and `--accum_steps` as per the available GPU memory).
As we can see below, QAT has improved the validation perplexity.

You could get slightly different numbers depending on your hyper-parameters - however you should be able to see consistent improvement
for QAT over PTQ alone.

| | Validation perplexity on `nvidia/Daring-Anteater` dataset |
|-----------------|--------------------|
| Fine-tuned BF16 (No quantization) | 1.45 |
| PTQ with NVFP4 weights & NVFP4 activations on the Fine-tuned BF16 model | 1.56 |
| QAT with NVFP4 weights & NVFP4 activations | 1.49 |

> **_NOTE:_** From our experience, the QAT performs better with a larger batch size, so we recommend using a larger batch size if your hardware allows it.

> **_NOTE:_** If you only use part of the dataset for fine-tuning/QAT, we recommend to use different data samples for fine-tuning and QAT, otherwise there may appear overfitting issues during the QAT stage.

#### NeMo QAT/QAD Simplified Flow Example

The [examples/nemo_run/qat](../nemo_run/qat) directory also contains an end-to-end NeMo QAT Simplified Flow example, which supports both QAT with cross-entropy loss and QAD (quantization-aware distillation) with knowledge-distillation loss between the full-precision teacher and quantized student models. Refer to [README](../nemo_run/qat/README.md) for more detail.

#### Testing QAT model with LLM benchmarks for accuracy evaluation

The model generated after QAT can be tested for LLM accuracy evaluation for various LLM benchmarks. After running the fine-tuning, following code can be used to run LLM evaluation for [supported tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks).

To run the llm_eval tasks on QAT model, run:

```sh
cd ../llm_eval

python lm_eval_hf.py --model hf \
    --tasks <comma separated tasks> \
    --model_args pretrained=../llm_qat/llama3-qat \
    --quant_cfg NVFP4_DEFAULT_CFG \
    --batch_size 4
```

See more details on running LLM evaluation benchmarks [here](../llm_eval/README.md).

#### Deployment

The final model after QAT is similar in architecture to that of PTQ model. QAT model simply have updated weights as compared to the PTQ model. It can be deployed to TensorRT-LLM (TRTLLM) or to TensorRT just like a regular **ModelOpt** PTQ model if the quantization format is supported for deployment.

To run QAT model with TRTLLM, run:

```sh
cd ../llm_ptq

./scripts/huggingface_example.sh --model ../llm_qat/llama3-qat --quant w4a8_awq
```

Note: The QAT checkpoint for `w4a8_awq` config can be created by using `--quant_cfg W4A8_AWQ_BETA_CFG` in [QAT example](#end-to-end-qat-example).

See more details on deployment of quantized model [here](../llm_ptq/README.md).

## Other Examples

### End-to-end QLoRA with Real Quantization

[QLoRA](https://arxiv.org/pdf/2305.14314) is a technique mainly intended for further reducing the training memory requirement of LoRA. In QLoRA, the LoRA backbone weights are quantized to reduce the model footprint. Unlike QAT which uses simulated quantization, QLoRA requires real quantization. To compress the model weights after quantization, we use the `mtq.compress()` function, which currently supports FP8, FP4, and INT4 formats. This feature can be enabled by passing `--compress True` to the `launch.sh` script. For detailed configuration options and patterns, please refer to the `modelopt.torch.quantization.compress` documentation.

To evaluate QLoRA quantized model before training, run:

```sh
# Load the HF checkpoint, quantize the model and evaluate without additional training
# Also compress the model after quantization
./launch.sh --model meta-llama/Meta-Llama-3-8B \
   --do_train False \
   --quant_cfg NVFP4_DEFAULT_CFG \
   --compress True
```

To perform QLoRA training, run:

```sh
# Load the HF checkpoint, quantize the model, add LoRA adapter, and run additional training
# Also compress the model after quantization
./launch.sh --model meta-llama/Meta-Llama-3-8B \
   --num_epochs 0.5 \
   --lr 1e-3 \
   --do_train True \
   --output_dir llama3-fp4-qlora \
   --quant_cfg NVFP4_DEFAULT_CFG \
   --compress True \
   --lora True
```

> **_NOTE:_** QLoRA is currently an experimental feature designed to reduce the memory footprint during training. Deployment functionality is not yet available.

## Quantization Aware Distillation (QAD)

Quantization aware distillation (QAD) can be used improve accuracy of the model using the original full precision model as a teacher model.

Here is an example workflow for performing QAD:

.. note::
QAD workflow is experimental and is subject to change.

```python
import modelopt.torch.opt as mto
import modelopt.torch.distill as mtd
import modelopt.torch.quantization as mtq
from modelopt.torch.distill.plugins.huggingface import LMLogitsLoss
from modelopt.torch.quantization.plugins.transformers_trainer import QADTrainer

...

# [Not shown] load model, tokenizer, data loaders etc
# Create the distillation config
distill_config = {
   "teacher_model": (
         _teacher_factory,
         (
            model_args.teacher_model,
            training_args.cache_dir,
         ),
         {},
   ),
   "criterion": LMLogitsLoss(),
   "expose_minimal_state_dict": False,
}

trainer = QADTrainer(
   model=model,
   processing_class=tokenizer,
   args=training_args,
   quant_args=quant_args,
   distill_config=distill_config,
   **data_module,
)

trainer.train()  # Train the quantized model using distillation (i.e, QAD)

# Save the final student model weights; An example usage
trainer.save_model(export_student=True)
```

#### End-to-end QAD example

To perform QAD with logits loss, run:

```sh
./launch.sh --model llama3-finetune \
   --num_epochs 3 \
   --lr 4e-5 \
   --quant_cfg NVFP4_DEFAULT_CFG \
   --do_train True \
   --output_dir llama-qad \
   --distill True
```

> **_NOTE:_** QAD currently requires quantization to be applied before the FSDP wrapper. Training is not supported for models that exceed single GPU memory capacity.

#### Testing and deploying QAD models

The model generated after QAD is similar to QAT model. Refer to [Testing](#testing-qat-model-with-llm-benchmarks-for-accuracy-evaluation) and [Deployment](#deployment) for more details.
