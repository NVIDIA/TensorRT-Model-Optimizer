<div align="center">

# Megatron-LM Integration Advanced Usage

[Kimi-K2-Instruct Slurm Examples](#slurm-examples) |
[Advanced Configuration](#advanced-configuration) |
[Checkpoint Resume](#checkpoint-resume) |
[Megatron-LM Integration](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt)

</div>

## Slurm Examples

For models that require multi-node, our scripts in Megatron-LM examples also support `slurm` with a sbatch wrapper.
Start with the example `slurm/sbatch.sh` with some minor modification or use your existing `sbatch`
script.

<br>

## ⭐ BF16 Kimi-K2-Instruct EAGLE3 Training

Different from local environment, we only allow passing variables through a shell script (default: `.env_setup_template.sh`).
Commandline variable passthrough is not supported. `config/moonshotai/kimi_k2_instruct.sh` is a config that has been tested
with 8 nodes of DGX H100 (TP=8, ETP=1, EP=64, overall 64 H100 GPUs in total). Update `HF_MODEL_CKPT` to the exact
checkpoint path in the container to start:

```sh
export USER_FSW=<path_to_scratch_space>
export CONTAINER_IMAGE=<path_to_container_image>
export SANDBOX_ENV_SETUP=./config/moonshotai/kimi_k2_instruct.sh
sbatch --nodes=8 slurm/sbatch.sh "/workspace/Megatron-LM/examples/post_training/modelopt/eagle3.sh moonshotai/Kimi-K2-Instruct"
```

To export the trained EAGLE3 model, switch to `kimi_k2_instruct_export.sh`.
**We only support pipeline-parallel (PP) export.** In this case, 2 nodes are used (PP=16).

```sh
export USER_FSW=<path_to_scratch_space>
export CONTAINER_IMAGE=<path_to_container_image>
export SANDBOX_ENV_SETUP=./config/moonshotai/kimi_k2_instruct_export.sh
sbatch --nodes=2 slurm/sbatch.sh "/workspace/Megatron-LM/examples/post_training/modelopt/export.sh moonshotai/Kimi-K2-Instruct"
```

## Advanced Configuration

WIP

## Checkpoint Resume

WIP
