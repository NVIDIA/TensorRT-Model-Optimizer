=================================================================
Unified HuggingFace Checkpoint
=================================================================

We support exporting modelopt-optimized Huggingface models and Megatron Core models to a unified checkpoint format that can be deployed in various inference frameworks such as TensorRT-LLM, vLLM, and SGLang.

The workflow is as follows:

#. Load the Huggingface models or Megatron Core models, `quantize with modelopt <https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/llm_ptq#ptq-post-training-quantization>`_ , and export to the unified checkpoint format, where the layer structures and tensor names are aligned with the original checkpoint.
#. Load the unified checkpoint in the supported inference framework for accelerated inference.


Export Quantized Model
======================

The modelopt quantized model can be exported to the unified checkpoint format stored as

#. A group of safetensors files, containing quantized model weights and scaling factors.
#. A ``hf_quant_config.json`` file containing quantization configurations.
#. Other json files that store the model structure information, tokenizer information, and metadata.


The export API (:meth:`export_hf_checkpoint <modelopt.torch.export.unified_export_hf.export_hf_checkpoint>`) can be used as follows:

.. code-block:: python

    from modelopt.torch.export import export_hf_checkpoint

    with torch.inference_mode():
        export_hf_checkpoint(
            model,  # The quantized model.
            export_dir,  # The directory where the exported files will be stored.
        )

Deployment Support Matrix
==============================================

Currently, we support the following quantization formats with the unified HF export API:
#. FP8
#. FP8_PB
#. NVFP4
#. NVFP4_AWQ
#. INT4_AWQ
#. W4A8_AWQ

For deployment with TensorRT-LLM, we support llama 3.1, 3.3, Mixtral 8x7B, with FP8 and NVFP4 checkpoints; Medusa and Eagle FP8 checkpoints are also tested.

For deployment with vLLM, we support llama 3.1, 3.3, Mixtral 8x7B, with FP8 checkpoints.

For deployment with SGLang, we support llama 3.1, 3.3, with FP8 checkpoints.

Other models and quantization formats may work, but they are not thoroughly tested.


Deployment with Selected Inference Frameworks
==============================================

.. tab:: TensorRT-LLM

    Follow the `TensorRT-LLM installation instructions. <https://nvidia.github.io/TensorRT-LLM/installation/linux.html>`_

    Currently we support fp8 and nvfp4 quantized models for TensorRT-LLM deployment, you need v0.17.0 or later version of TensorRT-LLM.

    To run modelopt quantized model from Huggingface model hub, e.g., `nvidia/Llama-3.1-8B-Instruct-FP8`_, refer to the sample code below:

    .. code-block:: python

        from tensorrt_llm import LLM, SamplingParams

        def main():

            prompts = [
                "Hello, my name is",
                "The president of the United States is",
                "The capital of France is",
                "The future of AI is",
            ]
            sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

            llm = LLM(model="nvidia/Llama-3.1-8B-Instruct-FP8")

            outputs = llm.generate(prompts, sampling_params)

            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        if __name__ == '__main__':
            main()

.. tab:: vLLM

    Follow `vLLM installation instructions. <https://github.com/vllm-project/vllm?tab=readme-ov-file#getting-started>`_

    Currently we support fp8 quantized models (without fp8 kv cache) for vLLM deployment, you need v0.6.5 or later version of vLLM.

    To run modelopt quantized model from Huggingface model hub, e.g., `nvidia/Llama-3.1-8B-Instruct-FP8`_, refer to the sample code below:

    .. code-block:: python

        from vllm import LLM, SamplingParams

        def main():

            model_id = "nvidia/Llama-3.1-8B-Instruct-FP8"
            sampling_params = SamplingParams(temperature=0.8, top_p=0.9)

            prompts = [
                "Hello, my name is",
                "The president of the United States is",
                "The capital of France is",
                "The future of AI is",
            ]

            llm = LLM(model=model_id, quantization="modelopt")
            outputs = llm.generate(prompts, sampling_params)

            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        if __name__ == "__main__":
            main()

.. tab:: SGLang

    Follow the `SGLang installation instructions. <https://docs.sglang.ai/start/install.html>`_

    Currently we support fp8 quantized models (without fp8 kv cache) for SGLang deployment, you need to use the main branch of SGLang (since Jan 6, 2025) and build it from source.

    To run modelopt quantized model from Huggingface model hub, e.g., `nvidia/Llama-3.1-8B-Instruct-FP8`_, refer to the sample code below:

    .. code-block:: python

        import sglang as sgl

        def main():

            prompts = [
                "Hello, my name is",
                "The president of the United States is",
                "The capital of France is",
                "The future of AI is",
            ]
            sampling_params = {"temperature": 0.8, "top_p": 0.95}
            llm = sgl.Engine(model_path="nvidia/Llama-3.1-8B-Instruct-FP8", quantization="modelopt")

            outputs = llm.generate(prompts, sampling_params)
            for prompt, output in zip(prompts, outputs):
                print("===============================")
                print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

        if __name__ == "__main__":
            main()

.. _nvidia/Llama-3.1-8B-Instruct-FP8: https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8

.. =================================================================
.. TODO: Add sample usage for Autodeploy when it's public
.. =================================================================
