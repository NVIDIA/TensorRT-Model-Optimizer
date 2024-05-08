# Sparsity for HuggingFace Models

In this tutorial, we demonstrate how to use Nvidia TensorRT Model Optimizer to perform Post-Training Sparsification (PTS) and Sparsity Aware Training (SAT) on a HuggingFace [Llama2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) model.

## Post-Training Sparsification (PTS) for HuggingFace Models

Post-training sparsification (PTS) is a technique used to sparsify a pretrained model without additional retraining.
PTS usually incurs noticeable accuracy drop compared to the original model, thus it is recommended to finetune the sparsified model using sparsity-aware training (SAT) to recover the accuracy.

### System Requirements

The PTS examples provided below have been verified to work on a system with 1 x A100/H100(80GB) GPUs.
The GPU memory consumption is approximately 44GB for the Llama2-7B model.

### Run PTS on Llama2-7B

This example demonstrates how to use Model Optimizer to perform post-training sparsification (PTS) on the Llama2-7B model.

> *If you experience numerical stability issues during the calculation of the Hessian inverse, it's recommended to maintain the `fp32` format for both model weights and activations by specifying `--dtype fp32`.*

```sh
python hf_pts.py --model_name_or_path meta-llama/Llama-2-7b-hf \
    --device cuda \
    --model_max_length 1024 \
    --dtype fp16 \
    --sparsity_fmt sparsegpt \
    --calib_size 128 \
    --output_dir saved_models_Llama-2-7b-hf_sparsegpt_tp1_pp1
```

The above command will sparsify the Llama2-7B model and save the sparsified model to `saved_models_Llama-2-7b-hf_sparsegpt_tp1_pp1` directory.
The saved checkpoint, e.g. `pts_modelopt_state.pth`,  can be loaded using `modelopt.torch.opt.restore()` for inference.

For details on the TensorRT-LLM deployment of the PTS model, please refer to [link](../llm_ptq/README.md#Post-training-Sparsity).

## Sparsity Aware Training (SAT)

Sparsity-aware training (SAT) is a training method that allows the model to learn accommodating the sparsity patterns
introduced by PTS. SAT is used to recover the accuracy loss incurred after PTS.  To carry out the following SAT example,
the user must first complete the prerequisite PTS step above and generate a PTS model checkpoint.

### System Requirements

The SAT examples provided below have been verified to work on a system with 8 x A100/H100(80GB) GPUs.
Each GPU typically utilizes approximately 40GB for finetuning the Llama2-7B model. The peak RAM usage is approximately 220GB.

### Setup

Install dependencies

```sh
pip install -r requirements.txt
```

### Data Preparation

Download and preprocess data for training and evaluation.

```sh
python data_prep.py --save_path data
```

### Run SAT on Llama2-7B

The following command demonstrates how to perform SAT on the Llama2-7B model on 8 GPUs.
The model is finetuned on the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset for 3 epochs.
The input data is tokenized to a maximum length of 1024 tokens. The tokenized data is saved as a pickle file for faster data loading. The one-time process takes less than an hour to finish depending on the CPU. The resulting pickle file can be utilized for future training sessions.

```sh
bash launch_finetune.sh --model meta-llama/Llama-2-7b-hf \
    --max_length 1024 \
    --output_dir saved_models_Llama-2-7b-hf_sparsegpt_tp1_pp1/finetuned \
    --num_epochs 3 \
    --restore_path saved_models_Llama-2-7b-hf_sparsegpt_tp1_pp1/pts_modelopt_state.pth
```

The saved checkpoint, e.g. `finetuned_modelopt_state.pth`,  can be loaded using `modelopt.torch.opt.restore()` for inference.

> *The above commands are for demonstration purposes only. Users are encouraged to modify the hyperparameters based on their use case. Sparsity aware training is computationally expensive and may require a large number of GPUs to train the model in a reasonable amount of time. The default setting is 3 epochs, which typically yields optimal performance. Users are encouraged to increase the number of epochs to achieve better performance or decrease it to reduce training time. For example, you can replace `--num_epochs 3` with `--max_steps 1000` to train the model for 1000 iterations.*

> *Sparsity aware training requires higher GPU memory compared to the original model training. If you encounter out-of-memory issues, consider reducing the batch size, enable gradient checkpointing, or choose lower precision in training to reduce GPU memory footprint.*

> *`mto.modelopt_state()` is used to retrieve the modelopt state_dict, which is necessary for saving and restoring the sparsified model. When [FSDP](https://pytorch.org/docs/stable/fsdp.html) is enabled, `mto.modelopt_state()` should be executed immediately after the model has been restored using `mto.restore()` to avoid any model loading issues.*

## Evaluation

The following command demonstrates how to evaluate the finetuned model on the `cnn_dailymail` dataset.

```sh
accelerate launch --multi_gpu --num_processes=8 eval.py \
    --model_dir meta-llama/Llama-2-7b-hf \
    --data_path data/cnn_eval.json \
    --batch_size 1 \
    --beam_size 4 \
    --model_max_length 1024 \
    --modelopt_restore_path saved_models_Llama-2-7b-hf_sparsegpt_tp1_pp1/finetuned/finetuned_modelopt_state.pth
```

The output should be similar to the following:

```sh
ROUGE scores: {'rouge1': 42.174, 'rouge2': 19.2724, 'rougeL': 28.6989, 'rougeLsum': 39.1394}
```

Please refer to [link](../llm_eval/README.md#Evaluation-scripts-for-LLM-tasks) for more details of how to evaluate the sparsified models on other benchmarks, such as MMLU and HumanEval.
