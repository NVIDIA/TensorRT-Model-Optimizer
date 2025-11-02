# Compress Algorithm Tutorial

This tutorial demonstrates how to compress large language models using the compress algorithm based on the [Puzzle paper](https://arxiv.org/abs/2411.19146).

In this example, we compress the [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model by searching for the optimal `ffn_intermediate_size` across MLP layers and `attention op/noop`. This results in a heterogeneous architecture while reducing GPU memory usage from 113GB to 96GB (15% reduction) with less than 1% regression in the token_accuracy_top_10 metric.

## Compress the Model

1. Specify the `puzzle_dir`, `input_hf_model_path`, `dataset_path`, `intermediate_size_list`, and `target_memory` arguments in the [llama-3_1-8B_pruneffn_memory.yaml](./configs/llama-3_1-8B_pruneffn_memory/llama-3_1-8B_pruneffn_memory.yaml) configuration file.

   Let's first shoot for 32% GPU memory reduction setting `target_memory = 78_000` GiB.

2. Download and prepare the [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2).

   dataset split: "code", "math", "stem", "chat", excluding reasoning samples (2.62GB)

   ```bash
   python -m modelopt.torch._compress.dataset.prepare_dataset --dataset_name nvidia/Nemotron-Post-Training-Dataset-v2 --output_dir path/to/Nemotron-Post-Training-Dataset-v2
   ```

3. Run the compression script.

   ```bash
   torchrun --nproc_per_node 1 examples/compress/main.py --config path/to/llama-3_1-8B_pruneffn_memory.yaml 2>&1 | tee ./log.txt | grep "Compress Progress"
   ```

   This will save the full output to `log.txt` and display the following progress on screen:

   ```bash
   # Produced on a single NVIDIA H100 80GB HBM3 card
   [2025-11-01 14:01:10] Compress Progress 1/8: starting compression pipeline
   [2025-11-01 14:01:10] Compress Progress 2/8: converting model from HF to DeciLM
   [2025-11-01 14:01:29] Compress Progress 3/8: scoring pruning activations
   [2025-11-01 14:02:30] Compress Progress 4/8: pruning the model and saving pruned checkpoints
   [2025-11-01 14:03:18] Compress Progress 5/8: building replacement library and calculating subblock statistics
   [2025-11-01 14:03:19] Compress Progress 6/8: calculating one block scores
   [2025-11-01 14:13:35] Compress Progress 7/8: running MIP and realizing models
   [2025-11-01 14:13:52] Compress Progress 8/8: compression pipeline completed
   ```

   This will generate the following network architecture (see `log.txt`):

   ```bash
   ...
   block_0:   attention  gqa_4   ffn  intermediate_14336
   block_1:   attention  gqa_4   ffn  intermediate_14336
   block_2:   attention  gqa_4   ffn  intermediate_14336
   block_3:   attention  gqa_4   ffn  intermediate_14336
   block_4:   attention  gqa_4   ffn  intermediate_14336
   block_5:   attention  gqa_4   ffn  intermediate_14336
   block_6:   attention  gqa_4   ffn  intermediate_14336
   block_7:   attention  gqa_4   ffn  intermediate_14336
   block_8:   attention  gqa_4   ffn  intermediate_14336
   block_9:   attention  gqa_4   ffn  intermediate_14336
   block_10:  attention  gqa_4   ffn  intermediate_14336
   block_11:  attention  gqa_4   ffn  intermediate_14336
   block_12:  attention  gqa_4   ffn  intermediate_14336
   block_13:  attention  gqa_4   ffn  intermediate_14336
   block_14:  attention  gqa_4   ffn  intermediate_14336
   block_15:  attention  gqa_4   ffn  intermediate_14336
   block_16:  attention  gqa_4   ffn  intermediate_14336
   block_17:  attention  no_op   ffn  intermediate_14336
   block_18:  attention  no_op   ffn  intermediate_14336
   block_19:  attention  no_op   ffn  intermediate_14336
   block_20:  attention  no_op   ffn  intermediate_14336
   block_21:  attention  no_op   ffn  intermediate_14336
   block_22:  attention  no_op   ffn  intermediate_14336
   block_23:  attention  no_op   ffn  intermediate_14336
   block_24:  attention  no_op   ffn  intermediate_14336
   block_25:  attention  no_op   ffn  intermediate_14336
   block_26:  attention  no_op   ffn  intermediate_14336
   block_27:  attention  no_op   ffn  intermediate_14336
   block_28:  attention  no_op   ffn  intermediate_14336
   block_29:  attention  gqa_4   ffn  intermediate_14336
   block_30:  attention  gqa_4   ffn  intermediate_14336
   block_31:  attention  gqa_4   ffn  intermediate_14336
   
   [2025-11-02 04:53:11,332]^[[92m[rank-0]^[[0m[run_puzzle.py:295] Total costs: {'stats.memory_mib': 75796.4140625, 'stats.ffn_num_params': 5637275648, 'stats.num_kv_heads': 160, 'stats.kv_cache_memory_mib': 61440.0, 'stats.ffn_memory_mib': 10752.25, 'stats.attention_memory_mib': 63040.15625, 'stats.attention_num_params': 838942720, 'stats.num_params': 7526895616, 'stats.has_attention': 20, 'stats.has_ffn': 32}
   ...
   ################################################################
   validate_model_and_extract_token_probs(model_name='teacher')
   ################################################################
   ...
   Average losses = {'lm_loss': 1.118250765837729, 'token_accuracy_top_1': 0.7331905364990234, 'token_accuracy_top_5': 0.9094219207763672, 'token_accuracy_top_10': 0.9423646926879883,
   ...
   ################################################################
   validate_model_with_kl_div(model_name='solution_0', is_calc_kl_div=True)
   ################################################################
   ....
   Average losses = {'lm_loss': 1.7577573340386152, 'token_accuracy_top_1': 0.6225490570068359, 'token_accuracy_top_5': 0.846257209777832, 'token_accuracy_top_10': 0.8987817764282227} 

   ```

   30% GPU memory reduction leads to nearly 5% regression in token_accuracy_top_10 metric (0.898 / 0.942). Let's rerun MIP search aiming for 15% memory reduction.

## Re-run MIP Search with different memory constraints

If you want to try different memory constraints without re-running the expensive pruning and scoring steps, use the `--mip-only` flag.
This assumes pruning, replacement library building, NAS scoring, and subblock stats calculation have already been completed.

Set `target_memory: 96_000` in `llama-3_1-8B_pruneffn_memory.yaml`.

```bash
torchrun --nproc_per_node 1 examples/compress/main.py --config path/to/llama-3_1-8B_pruneffn_memory.yaml --mip-only 2>&1 | tee ./log.txt | grep "Compress Progress"
```

This will generate the following network architecture (see `log.txt`):

```bash
block_0:   attention  gqa_4   ffn  intermediate_14336
block_1:   attention  gqa_4   ffn  intermediate_14336
block_2:   attention  gqa_4   ffn  intermediate_14336
block_3:   attention  gqa_4   ffn  intermediate_14336
block_4:   attention  gqa_4   ffn  intermediate_14336
block_5:   attention  gqa_4   ffn  intermediate_14336
block_6:   attention  gqa_4   ffn  intermediate_14336
block_7:   attention  gqa_4   ffn  intermediate_14336
block_8:   attention  gqa_4   ffn  intermediate_14336
block_9:   attention  gqa_4   ffn  intermediate_14336
block_10:  attention  gqa_4   ffn  intermediate_14336
block_11:  attention  gqa_4   ffn  intermediate_14336
block_12:  attention  gqa_4   ffn  intermediate_14336
block_13:  attention  gqa_4   ffn  intermediate_14336
block_14:  attention  gqa_4   ffn  intermediate_14336
block_15:  attention  gqa_4   ffn  intermediate_14336
block_16:  attention  gqa_4   ffn  intermediate_14336
block_17:  attention  gqa_4   ffn  intermediate_14336
block_18:  attention  no_op   ffn  intermediate_14336
block_19:  attention  no_op   ffn  intermediate_14336
block_20:  attention  no_op   ffn  intermediate_14336
block_21:  attention  gqa_4   ffn  intermediate_14336
block_22:  attention  no_op   ffn  intermediate_14336
block_23:  attention  no_op   ffn  intermediate_14336
block_24:  attention  no_op   ffn  intermediate_14336
block_25:  attention  gqa_4   ffn  intermediate_14336
block_26:  attention  gqa_4   ffn  intermediate_14336
block_27:  attention  gqa_4   ffn  intermediate_14336
block_28:  attention  gqa_4   ffn  intermediate_14336
block_29:  attention  gqa_4   ffn  intermediate_14336
block_30:  attention  gqa_4   ffn  intermediate_14336
block_31:  attention  gqa_4   ffn  intermediate_14336

[2025-11-02 11:01:56,443]^[[92m[rank-0]^[[0m[run_puzzle.py:295] Total costs: {'stats.memory_mib': 94708.4609375, 'stats.attention_memory_mib': 81952.203125, 'stats.ffn_memory_mib': 10752.25, 'stats.has_ffn': 32, 'stats.ffn_num_params': 5637275648, 'stats.attention_num_params': 1090625536, 'stats.has_attention': 26, 'stats.kv_cache_memory_mib': 79872.0, 'stats.num_kv_heads': 208, 'stats.num_params': 7778578432}
...
################################################################
validate_model_with_kl_div(model_name='solution_0', is_calc_kl_div=True)
################################################################
Average losses = {'lm_loss': 1.2425934937782586, 'token_accuracy_top_1': 0.703862190246582, 'token_accuracy_top_5': 0.8954982757568359, 'token_accuracy_top_10': 0.9336576461791992,
```

On the other hand, if you set `target_memory: 28_000`, you would observe that for some layers the intermediate FFN size starts to reduce (see `log.txt`):

```bash
block_5:   attention  no_op   ffn  intermediate_11520
block_6:   attention  no_op   ffn  intermediate_14336
block_7:   attention  no_op   ffn  intermediate_8704
block_8:   attention  no_op   ffn  intermediate_14336
block_9:   attention  no_op   ffn  intermediate_3072
block_10:  attention  no_op   ffn  intermediate_11520
block_11:  attention  no_op   ffn  intermediate_11520
block_12:  attention  no_op   ffn  intermediate_11520
block_13:  attention  no_op   ffn  intermediate_11520
block_14:  attention  no_op   ffn  intermediate_3072
```

## Advanced usage

Modify `path/to/Llama-3_1-8B yaml` file for advanced compression scenarios.
