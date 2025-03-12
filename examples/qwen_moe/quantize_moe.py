import copy
import json
import logging
import random
import time
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from modelopt.torch.utils import create_forward_loop
import modelopt.torch.quantization as mtq
from accelerate.hooks import remove_hook_from_module
from modelopt.torch.export import export_tensorrt_llm_checkpoint
logger = logging.getLogger(__name__)
# python3.10
# pip install https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-0.17.0.post1-cp310-cp310-linux_x86_64.whl
# pip install -e .[torch,onnx] accelerate transformers datasets hf_transfer

KV_CACHE_CFG = {
    "*.query_key_value.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.Wqkv.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.W_pack.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.c_attn.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.k_proj.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.v_proj.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
}


def quantize_model(model, quant_cfg,
                    tokenizer,
                   ):
    

    # NOTE: for ModelOpt v0.19 release
    calibrate_loop = create_forward_loop(
        dataset_name="cnn_dailymail",
        tokenizer=tokenizer,
        model=model,
        batch_size=0, num_samples=64
    )


    logger.info("Starting quantization...")
    start_time = time.time()
    
    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    end_time = time.time()
    logger.info(
        "Quantization done. Total time used: {:.2f} s.".format(end_time -
                                                               start_time))
    return model

def quant_cfg_choices():
    import modelopt.torch.quantization as mtq
    QUANT_CFG_CHOICES = {
        "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
        "fp8": mtq.FP8_DEFAULT_CFG,
        "int4_awq": mtq.INT4_AWQ_CFG,
        "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
    }
    if hasattr(mtq, "NVFP4_DEFAULT_CFG"):
        QUANT_CFG_CHOICES["nvfp4"] = mtq.NVFP4_DEFAULT_CFG
    return QUANT_CFG_CHOICES

def get_tokenizer(ckpt_path, max_seq_length=2048, model_type=None):
    logger.info(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=max_seq_length,
        padding_side="left",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        if model_type and model_type == "qwen":
            # qwen use token id 151643 as pad and eos tokens
            tokenizer.eos_token = tokenizer.convert_ids_to_tokens(151643)
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
        else:
            tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token is not None, f"Pad token for {model_type} cannot be set!"

    return tokenizer

def get_model(ckpt_path: str,
              dtype: str = 'bfloat16',
              device: str = 'cuda',
              device_map: str = "auto"):
    logger.info(f"Initializing model from {ckpt_path}")
    # Note: VILA model is not in public HF model zoo yet. We need to explicitly import from the git repo
    hf_config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
    torch_dtype = getattr(hf_config, 'torch_dtype', None)

    model_cls = AutoModelForCausalLM
    if hf_config.model_type == "llava":
        from transformers import LlavaForConditionalGeneration
        model_cls = LlavaForConditionalGeneration
    elif hf_config.model_type == "mpt":
        from transformers import MptForCausalLM
        model_cls = MptForCausalLM
    elif hf_config.model_type == 'mllama':
        from transformers import MllamaForConditionalGeneration
        model_cls = MllamaForConditionalGeneration

    elif hf_config.model_type == "glm":
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path,
                                                      device_map="cuda",
                                                      torch_dtype=torch_dtype,
                                                      trust_remote_code=True)
    else:
        model = model_cls.from_pretrained(
            ckpt_path,
            device_map=device_map if device != "cpu" else "cpu",
            torch_dtype="auto",
            trust_remote_code=True)
        if hf_config.model_type in ["llava", "internvl_chat"]:
            model = model.language_model

    model.eval()

    model_dtype = next(model.parameters()).dtype
    if torch_dtype != model_dtype:
        logger.info(
            f"[TensorRT-LLM][WARNING] The manually set model data type is {dtype}, "
            f"but the data type of the HuggingFace model is {model_dtype}.")

    return model

def quantize_and_export(*,
                        model_name,
                        device,
                        calib_dataset,
                        dtype,
                        qformat,
                        kv_cache_dtype,
                        calib_size,
                        batch_size,
                        calib_max_seq_length,
                        awq_block_size,
                        output_dir,
                        tp_size,
                        pp_size,
                        cp_size,
                        seed,
                        tokenizer_max_seq_length,
                        num_medusa_heads=None,
                        num_medusa_layers=None,
                        max_draft_len=None,
                        medusa_hidden_act=None,
                        medusa_model_dir=None,
                        quant_medusa_head=None,
                        auto_quantize_bits=None,
                        device_map="auto",
                        quantize_lm_head=False):
    '''
        Load model from the model_dir, call Modelopt to quantize the model, and then export
        the quantized model as TRT-LLM checkpoint
    '''
    try:
        import modelopt  # noqa
    except ImportError as e:
        logger.error(
            "Failed to import modelopt, pls check the Modelopt installation. Currently it is known to be unsupported on Windows OS"
        )
        raise e

    


    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for inference.")

    random.seed(seed)
    np.random.seed(seed)

    # Check that only one quantization format is provided for non auto_quant case
    if not auto_quantize_bits:
        assert (len(qformat.split(",")) == 1
                ), "Quantization supports only one quantization format."

    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    dtype = getattr(hf_config, 'torch_dtype', None)

    model = get_model(model_name, dtype, device=device, device_map=device_map)
    
    is_enc_dec = False
    model_type = "qwen" if "qwen" in getattr(hf_config, "model_type", None) else "error"
    if "error" in model_type:
        raise ValueError(f"Unsupported model type: {model_type}")
    tokenizer = get_tokenizer(model_name,
                                max_seq_length=tokenizer_max_seq_length,
                                model_type=model_type)
    if qformat in ["full_prec", "int8_wo", "int4_wo"
                   ] and kv_cache_dtype is None:
        logger.info(f"No quantization applied, export {dtype} model")
    else:
        if "awq" in qformat:
            if calib_size > 32:
                logger.info(
                    f"AWQ calibration could take longer with calib_size = {calib_size}, Using"
                    " calib_size=32 instead")
                calib_size = 32
            logger.info(
                "\nAWQ calibration could take longer than other calibration methods. Please"
                " increase the batch size to speed up the calibration process. Batch size can be"
                " set by adding the argument --batch_size <batch_size> to the command line.\n"
            )

        quant_cfg = None
        if auto_quantize_bits:
            raise
        else:
            if qformat in quant_cfg_choices():
                quant_cfg = quant_cfg_choices()[qformat]
            else:
                raise ValueError(f"Unsupported quantization format: {qformat}")

            if "awq" in qformat:
                quant_cfg = copy.deepcopy(quant_cfg_choices()[qformat])
                weight_quantizer = quant_cfg["quant_cfg"]["*weight_quantizer"]
                if isinstance(weight_quantizer, list):
                    weight_quantizer = weight_quantizer[0]
                if awq_block_size:
                    weight_quantizer["block_sizes"][-1] = awq_block_size

                # Coarser optimal scale search seems to resolve the overflow in TRT-LLM for some models
                if "w4a8_awq" == qformat and model_type in ["gemma", "mpt"]:
                    quant_cfg["algorithm"] = {
                        "method": "awq_lite",
                        "alpha_step": 1
                    }

            if kv_cache_dtype is not None:
                if kv_cache_dtype == "fp8":
                    for value in KV_CACHE_CFG.values():
                        value.update({"num_bits": (4, 3)})  # type: ignore
                quant_cfg["quant_cfg"].update(KV_CACHE_CFG)  # type: ignore

            # Gemma 7B has accuracy regression using alpha 1. We set 0.5 instead.
            if model_type == "gemma" and "int8_sq" in qformat:
                quant_cfg["algorithm"] = {"method": "smoothquant", "alpha": 0.5}

            if qformat == 'fp8' and quantize_lm_head:
                print("Quantizing lm_head layer")
                del quant_cfg["quant_cfg"]["*lm_head*"]



        model = quantize_model(model, quant_cfg, tokenizer,)

    with torch.inference_mode():
        if model_type is None:
            logger.info(
                f"Unknown model type {type(model).__name__}. Continue exporting..."
            )
            model_type = f"unknown:{type(model).__name__}"

        architecture = type(model).__name__

        export_path = output_dir
        start_time = time.time()

        # Move meta tensor back to device before exporting.
        remove_hook_from_module(model, recurse=True)
        
        export_tensorrt_llm_checkpoint(
            model,
            model_type,
            getattr(torch, dtype) if isinstance(dtype, str) else dtype,
            export_dir=export_path,
            inference_tensor_parallel=tp_size,
            inference_pipeline_parallel=pp_size,
        )

        export_paths = []
        tensorrt_llm_configs = []
        if not is_enc_dec:
            with open(f"{export_path}/config.json", "r") as f:
                tensorrt_llm_config = json.load(f)
            tensorrt_llm_configs.append(tensorrt_llm_config)
            export_paths.append(export_path)
        else:
            for component in ["encoder", "decoder"]:
                with open(f"{export_path}/{component}/config.json", "r") as f:
                    tensorrt_llm_config = json.load(f)
                tensorrt_llm_configs.append(tensorrt_llm_config)
                export_paths.append(f"{export_path}/{component}")

        for export_path, tensorrt_llm_config in zip(export_paths,
                                                    tensorrt_llm_configs):

            tensorrt_llm_config["model_type"] = model_type
            if not is_enc_dec:
                tensorrt_llm_config["architecture"] = architecture

            # Workaround for wo quantization
            if qformat in ["int8_wo", "int4_wo", "full_prec"]:
                from tensorrt_llm.quantization import QuantAlgo
                if qformat == "int8_wo":
                    tensorrt_llm_config["quantization"][
                        "quant_algo"] = QuantAlgo.W8A16
                elif qformat == "int4_wo":
                    tensorrt_llm_config["quantization"][
                        "quant_algo"] = QuantAlgo.W4A16
                else:
                    tensorrt_llm_config["quantization"]["quant_algo"] = None

            # HF uses rope_scaling while tensorrt_llm uses rotary_scaling
            if hasattr(model.config, "rope_scaling"
                       ) and "rotary_scaling" not in tensorrt_llm_config:
                tensorrt_llm_config["rotary_scaling"] = getattr(
                    model.config, "rope_scaling")
            with open(f"{export_path}/config.json", "w") as f:
                json.dump(tensorrt_llm_config, f, indent=4)

            # Workaround for Modelopt 0.9.x fp8_kv_cache knob issue
            if qformat in ['fp8', 'nvfp4'] and kv_cache_dtype is None:
                with open(f"{export_path}/config.json", "r") as f:
                    tensorrt_llm_config = json.load(f)
                tensorrt_llm_config["quantization"][
                    "kv_cache_quant_algo"] = None
                with open(f"{export_path}/config.json", "w") as f:
                    json.dump(tensorrt_llm_config, f, indent=4)

            # Workaround for qwen version
            if model_type == 'qwen' or model_type == 'qwen2_vl':
                with open(f"{export_path}/config.json", "r") as f:
                    tensorrt_llm_config = json.load(f)
                qwen_config = AutoConfig.from_pretrained(model_name,
                                                         trust_remote_code=True)
                try:
                    from transformers import LlavaOnevisionConfig
                    if isinstance(qwen_config, LlavaOnevisionConfig):
                        qwen_config = qwen_config.text_config
                except:
                    pass
                tensorrt_llm_config["qwen_type"] = qwen_config.model_type
                if qwen_config.model_type == "qwen2":
                    tensorrt_llm_config[
                        "norm_epsilon"] = qwen_config.rms_norm_eps
                    tensorrt_llm_config["rotary_base"] = qwen_config.rope_theta
                tensorrt_llm_config[
                    "intermediate_size"] = qwen_config.intermediate_size
                with open(f"{export_path}/config.json", "w") as f:
                    json.dump(tensorrt_llm_config, f, indent=4)

            # Set rotary parameters correctly for chatglm.
            if model_type == 'chatglm':
                rotary_base = 10000.0
                rotary_embedding_scaling = None
                chatglm_config = AutoConfig.from_pretrained(
                    model_name, trust_remote_code=True)
                chatglm_version = tensorrt_llm_config['chatglm_version']
                rope_ratio = tensorrt_llm_config.get('rope_ratio', 1.0)
                if chatglm_version == 'chatglm2':
                    if rope_ratio > 1:
                        rotary_embedding_scaling = {
                            'type': 'linear',
                            'factor': rope_ratio
                        }
                elif chatglm_version == 'chatglm3':
                    rotary_base *= rope_ratio

                with open(f"{export_path}/config.json", "r") as f:
                    tensorrt_llm_config = json.load(f)
                tensorrt_llm_config['rotary_base'] = rotary_base
                tensorrt_llm_config['rotary_scaling'] = rotary_embedding_scaling
                tensorrt_llm_config['rotary_pct'] = 0.5
                with open(f"{export_path}/config.json", "w") as f:
                    json.dump(tensorrt_llm_config, f, indent=4)

            # context parallel
            if cp_size > 1:
                with open(f"{export_path}/config.json", "r") as f:
                    tensorrt_llm_config = json.load(f)
                tensorrt_llm_config["mapping"]["cp_size"] = cp_size
                tensorrt_llm_config["mapping"]["world_size"] *= cp_size
                with open(f"{export_path}/config.json", "w") as f:
                    json.dump(tensorrt_llm_config, f, indent=4)

            if model_type == 'gptnext':
                with open(f"{export_path}/config.json", "r") as f:
                    tensorrt_llm_config = json.load(f)
                if tensorrt_llm_config['max_position_embeddings'] is None:
                    tensorrt_llm_config['max_position_embeddings'] = getattr(
                        model.config, "n_positions", None)
                with open(f"{export_path}/config.json", "w") as f:
                    json.dump(tensorrt_llm_config, f, indent=4)

         

            # Workaround for lm_head quantization
            # Can be removed after modelopt version is > 0.23
            if quantize_lm_head:
                with open(f"{export_path}/config.json", "r") as f:
                    tensorrt_llm_config = json.load(f)
                if 'lm_head' in tensorrt_llm_config['quantization'][
                        'exclude_modules']:
                    tensorrt_llm_config['quantization'][
                        'exclude_modules'].remove('lm_head')
                with open(f"{export_path}/config.json", "w") as f:
                    json.dump(tensorrt_llm_config, f, indent=4)

        end_time = time.time()
        print(
            "Quantized model exported to {} \nTotal time used {:.2f} s.".format(
                export_path, end_time - start_time))

        # Need to delete the model and release memory explicitly;
        # otherwise torch may retain its GPU memory until a delayed GC running,
        # which reduces the available GPU memory for subsequent stages.
        del model

if __name__ == "__main__":
    # model_name = "Qwen/Qwen2-57B-A14B-Instruct"
    model_name = "Qwen/Qwen1.5-MoE-A2.7B"
    qformat = "fp8"
    # Select the quantization config, for example, INT8 Smooth Quant
    configs = {
        "int4_awq": mtq.INT4_AWQ_CFG,
        "fp8": mtq.FP8_DEFAULT_CFG,
        "full_prec": None
    }
    config = configs[qformat]
    dtype = "bfloat16"
    
    export_path_short = f"./{model_name}_{qformat}"
    export_path = f"{export_path_short}/tllm-checkpoint"
    kv_cache_dtype = None

    quantize_and_export(
        model_name=model_name,
        device="cuda",
        calib_dataset="cnn_dailymail",
        dtype="bfloat16",
        qformat=qformat,
        kv_cache_dtype=kv_cache_dtype,
        calib_size=1024,
        batch_size=0,
        calib_max_seq_length=512,
        awq_block_size=None,
        output_dir=export_path,
        tp_size=2,
        pp_size=1,
        cp_size=1,
        seed=0,
        tokenizer_max_seq_length=512,
        num_medusa_heads=None,
        num_medusa_layers=None, 
    )
    # copy from the model repo name to the via snapshot_donwload.py
    from huggingface_hub import snapshot_download, upload_large_folder
    snapshot_download(
        model_name,
        revision="main",
        local_dir=export_path_short,
        max_workers=8,
        allow_patterns=["*.json", "*.txt", "tokenizer.model", "*.py"]
    )
    
    model_name_up = "michaelfeil/"+model_name.split("/")[1]+f"{qformat}_tllm"
    print(f"uplaoding to {model_name_up}")
    upload_large_folder(
        model_name_up,
        export_path_short,
        repo_type="model",
        private=False,
    )