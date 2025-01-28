# Quantization-Aware Training (QAT)

Quantization-Aware Training (QAT) is a powerful technique for optimizing your models, particularly when post-training quantization (PTQ) methods fail to meet the requirements for your tasks. By simulating the effects of quantization during training, QAT allows the model to learn to minimize the quantization error, ultimately delivering better accuracy.

If PTQ methods on the market fall short of your needs, QAT should be your next step.

## Implementing QAT with ModelOPT

**Overview**

ModelOPT makes QAT straightforward and flexible. While the example below uses Hugging Face Accelerate for simplicity. You can integrate QAT into your workflow using your preferred training setup.

**Example:**

Here’s an example workflow using Hugging Face Accelerate to manage distributed training:

```python
# Restore the model in its quantized state using ModelOPT's API
mto.restore(transformer, args.restore_quantized_ckpt)

# Move the model to the appropriate device and set the desired weight precision
transformer.to(accelerator.device, dtype=weight_dtype)
transformer.requires_grad_(True)

transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    transformer, optimizer, train_dataloader, lr_scheduler
)

```

Once the model is loaded in its quantized state through ModelOPT, you can proceed with regular training. The QAT process will automatically take place during the forward pass.

**How QAT Works in ModelOPT**

With ModelOPT, the forward pass simulates quantization, allowing the model to adjust its weights to minimize training loss and reduce quantization errors. This enables the model to better handle the constraints of quantized hardware without significant loss of performance.

**Combining Other Techniques in ModelOPT**

ModelOPT enables seamless integration of additional techniques, such as distillation, into your workflow without requiring significant changes to your training code.

**Example: Combining Distillation with QAT**

Distillation is a powerful approach where a high-precision model (the teacher) guides the training of a quantized model (the student). ModelOPT simplifies the process of combining distillation with QAT by handling most of the complexity for you.

For more details about distillation, please refer to this [link](../../../docs/source/getting_started/6_distillation.rst).

Here’s an example of how to implement this:

```diff
# Restore the model in its quantized state using ModelOPT's API
mto.restore(transformer, args.restore_quantized_ckpt)

'''
After mtd.convert, the model structure becomes:

model:
    transformer_0
    transformer_1
    teacher_model:
        transformer_0
        transformer_1

And the forward pass is automatically monkey-patched to:

def forward(input):
    student_output = model(input)
    _ = teacher_model(input)
    return student_output
'''

+ # Configuration for knowledge distillation (KD)
+ kd_config = {
+     "teacher_model": teacher_model,
+     "criterion": distill_config["criterion"],
+     "loss_balancer": distill_config["loss_balancer"],
+     "expose_minimal_state_dict": False,
+ }
+ transformer = mtd.convert(transformer, mode=[("kd_loss", kd_config)])

# Move the model to the appropriate device and set the desired weight precision
transformer.to(accelerator.device, dtype=weight_dtype)
transformer.requires_grad_(True)

# Making sure to freeze the weights from model._teacher_model
transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    transformer, optimizer, train_dataloader, lr_scheduler
)

# Compute the distillation loss using ModelOPT's compute_kd_loss
+ ...
+ loss = transformer.compute_kd_loss(...)
+ ...

```

## Flexibility to Customize QAT

ModelOPT supports fine-tuning weights during QAT out of the box only. However, if you have specific QAT algorithms or techniques, ModelOPT’s API is designed to accommodate custom implementations.

**Get Involved**

If you’re interested in extending ModelOPT’s QAT capabilities, we encourage you to contribute by:

- Raising a feature request
- Opening an issue
- Submitting a merge request with your implementation

By leveraging ModelOPT for QAT, you can achieve superior performance for complex image and video generation tasks, tailored to your specific requirements.
