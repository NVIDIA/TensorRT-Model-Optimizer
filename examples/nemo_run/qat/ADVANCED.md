# NeMo QAT/QAD Flow: Advanced Topics

If you need to run QAT/QAD on a Slurm cluster (for example to use more than 1 node)

To run the example on slurm, edit the `SLURM_CONFIG` at the bottom of `nemo_qat_flow.py` with the appropriate credentials, container, cluster name (host), and container mounts. Make sure you are mounting the NeMo and Megatron-LM repositories above in the Slurm cluster and that you've checked out the correct commits.

## Running the Flow on Slurm

To launch the Flow on a Slurm cluster, modify your Slurm credentials at the bottom of `nemo_qat_flow.py` and add the `--use-slurm` flag to the command. On a different server (e.g. your local server), launch the NeMo container as described in the [README](README.md) then run `python qat/nemo_qat_flow.py --use-slurm --log-dir /slurm/log/dir`, which will `ssh` into the Slurm cluster, `rsync` your files over, and launch the tasks. The log directory on the Slurm cluster should look like this after an experiment is run (assuming your experiment name is `qat_flow_ckpts`)

**NOTE:** `rsync` may not currently be available in the NeMo container and will be added as a dependency.

```
qat_flow_ckpts qat_flow_ckpts_1755708286
```

If you `cd` into the experiment itself, e.g. `cd qat_flow_ckpts_1755708286`, you'll find a directory structure like the following. Each folder is for a stage of the Simplified Flow, and in each stage you can see the logs for that stage as well as the sbatch command that was run. You can `cd` into each stage and `tail -f` the log file to see the logs while the stage is running.

```
├── 00_openscience_data
│   ├── code
│   ├── configs
│   ├── log-coreai_dlalgo_modelopt-modelopt.00_openscience_data_5345664_0.out
│   └── sbatch_coreai_dlalgo_modelopt-modelopt.00_openscience_data_5345664.out
├── 01_import_model
│   ├── code
│   ├── configs
│   ├── log-coreai_dlalgo_modelopt-modelopt.01_import_model_5345665_0.out
│   └── sbatch_coreai_dlalgo_modelopt-modelopt.01_import_model_5345665.out
├── 02_mmlu_bf16
│   ├── code
│   ├── configs
│   ├── log-coreai_dlalgo_modelopt-modelopt.02_mmlu_bf16_5345666_0.out
│   └── sbatch_coreai_dlalgo_modelopt-modelopt.02_mmlu_bf16_5345666.out
├── 03_ptq
│   ├── code
│   ├── configs
│   ├── log-coreai_dlalgo_modelopt-modelopt.03_ptq_5345667_0.out
│   └── sbatch_coreai_dlalgo_modelopt-modelopt.03_ptq_5345667.out
├── 04_mmlu_ptq
│   ├── code
│   ├── configs
│   ├── log-coreai_dlalgo_modelopt-modelopt.04_mmlu_ptq_5345668_0.out
│   └── sbatch_coreai_dlalgo_modelopt-modelopt.04_mmlu_ptq_5345668.out
├── 05_train
│   ├── code
│   ├── configs
│   ├── log-coreai_dlalgo_modelopt-modelopt.05_train_5345669_0.out
│   └── sbatch_coreai_dlalgo_modelopt-modelopt.05_train_5345669.out
├── 06_mmlu_sft
│   ├── code
│   └── configs
├── 07_export_hf
│   ├── code
│   └── configs
```
