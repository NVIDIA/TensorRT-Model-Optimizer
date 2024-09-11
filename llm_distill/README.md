# Knowledge Distillation for HuggingFace Models

In this tutorial, we demonstrate how to use Model Optimizer to perform teacher-student distillation.

## Knowledge Distilltion (KD) for HuggingFace Models

Knowledge Distillation allows one to use a more powerful model's learned features to guide a student
model's objective function into imitating it.
Here we finetune Llama-2 models on the [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca)
question-answer dataset as a minimal example to demonstrate a simple way of integrating Model Optimizer's KD feature.

First we do supervised finetuning (SFT) of a Llama-2-7b on OpenOrca dataset as the teacher, then distill it into
a 1B-parameter model.

Keep in mind the training loss of the distillation run is not directly comparable to the training loss of the teacher run.

> NOTE: We can fit the following in memory using [FSDP](https://huggingface.co/docs/accelerate/en/usage_guides/fsdp)
> enabled on 8x RTX 6000 (total ~400GB VRAM)

### Train teacher

```bash
accelerate launch --multi_gpu --mixed_precision bf16  main.py \
    --single_model \
    --teacher_name_or_path 'meta-llama/Llama-2-7b-hf' \
    --output_dir ./llama2-7b-sft \
    --logging_steps 5 \
    --max_steps 400 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_checkpointing True \
    --fsdp 'full_shard auto_wrap' \
    --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer
```

### Distill teacher into student

```bash
accelerate launch --multi_gpu --mixed_precision bf16  main.py \
    --teacher_name_or_path ./llama2-7b-sft \
    --student_name_or_path 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T' \
    --output_dir ./llama2-distill \
    --logging_steps 5 \
    --max_steps 200 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_checkpointing False \
    --fsdp 'full_shard auto_wrap' \
    --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer
```

> NOTE: If you receive a `RuntimeError: unable to open file <...> in read-only mode: No such file or directory` simply re-run the command a second time.
