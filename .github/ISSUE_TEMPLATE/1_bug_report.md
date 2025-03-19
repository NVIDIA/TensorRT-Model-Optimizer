---
name: Bug report
about: Submit a bug report to help us improve ModelOpt
title: ''
labels: bug
assignees: ''
---

## Describe the bug
<!-- Description of what the bug is, its impact (blocker, should have, nice to have) and any stack traces or error messages. -->


### Steps/Code to reproduce bug
<!-- Please list *minimal* steps or code snippet for us to be able to reproduce the bug. -->
<!-- A helpful guide on on how to craft a minimal bug report http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports. -->


### Expected behavior


## System information

- Container used (if applicable): ?
- OS (e.g., Ubuntu 22.04, CentOS 7, Windows 10): ? <!-- If Windows, please add the `windows` label to the issue. -->
- CPU architecture (x86_64, aarch64): ?
- GPU name (e.g. H100, A100, L40S): ?
- GPU memory size: ?
- Number of GPUs: ?
- Library versions (if applicable):
  - Python: ?
  - ModelOpt version or commit hash: ?
  - CUDA: ?
  - PyTorch: ?
  - Transformers: ?
  - TensorRT-LLM: ?
  - ONNXRuntime: ?
  - TensorRT: ?
- Any other details that may help: ?


<details>
<summary><b>Click to expand: Python script to automatically collect system information</b></summary>

```python
import platform
import re
import subprocess


def get_nvidia_gpu_info():
    try:
        nvidia_smi = (
            subprocess.check_output(
                "nvidia-smi --query-gpu=name,memory.total,count --format=csv,noheader,nounits",
                shell=True,
            )
            .decode("utf-8")
            .strip()
            .split("\n")
        )
        if len(nvidia_smi) > 0:
            gpu_name = nvidia_smi[0].split(",")[0].strip()
            gpu_memory = round(float(nvidia_smi[0].split(",")[1].strip()) / 1024, 1)
            gpu_count = len(nvidia_smi)
            return gpu_name, f"{gpu_memory} GB", gpu_count
    except Exception:
        return "?", "?", "?"


def get_cuda_version():
    try:
        nvcc_output = subprocess.check_output("nvcc --version", shell=True).decode("utf-8")
        match = re.search(r"release (\d+\.\d+)", nvcc_output)
        if match:
            return match.group(1)
    except Exception:
        return "?"


def get_package_version(package):
    try:
        return getattr(__import__(package), "__version__", "?")
    except Exception:
        return "?"


# Get system info
os_info = f"{platform.system()} {platform.release()}"
if platform.system() == "Linux":
    try:
        os_info = (
            subprocess.check_output("cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2", shell=True)
            .decode("utf-8")
            .strip()
            .strip('"')
        )
    except Exception:
        pass
elif platform.system() == "Windows":
    print("Please add the `windows` label to the issue.")

cpu_arch = platform.machine()
gpu_name, gpu_memory, gpu_count = get_nvidia_gpu_info()
cuda_version = get_cuda_version()

# Print system information in the format required for the issue template
print("=" * 70)
print("- Container used (if applicable): " + "?")
print("- OS (e.g., Ubuntu 22.04, CentOS 7, Windows 10): " + os_info)
print("- CPU architecture (x86_64, aarch64): " + cpu_arch)
print("- GPU name (e.g. H100, A100, L40S): " + gpu_name)
print("- GPU memory size: " + gpu_memory)
print("- Number of GPUs: " + str(gpu_count))
print("- Library versions (if applicable):")
print("  - Python: " + platform.python_version())
print("  - ModelOpt version or commit hash: " + get_package_version("modelopt"))
print("  - CUDA: " + cuda_version)
print("  - PyTorch: " + get_package_version("torch"))
print("  - Transformers: " + get_package_version("transformers"))
print("  - TensorRT-LLM: " + get_package_version("tensorrt_llm"))
print("  - ONNXRuntime: " + get_package_version("onnxruntime"))
print("  - TensorRT: " + get_package_version("tensorrt"))
print("- Any other details that may help: " + "?")
print("=" * 70)
```

</details>
