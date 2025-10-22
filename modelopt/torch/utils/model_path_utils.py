# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for handling model paths, supporting both local filesystem paths and HuggingFace Hub model IDs."""

import glob
import json
import os
import warnings
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download

    HF_HUB_AVAILABLE = True
except ImportError:
    hf_hub_download = None
    snapshot_download = None
    HF_HUB_AVAILABLE = False

try:
    from transformers import AutoConfig
    from transformers.utils import TRANSFORMERS_CACHE

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoConfig = None
    TRANSFORMERS_CACHE = None
    TRANSFORMERS_AVAILABLE = False

__all__ = [
    "ModelPathResolver",
    "fetch_model_config",
    "is_huggingface_model_id",
    "resolve_model_path",
]


def is_huggingface_model_id(model_path: str) -> bool:
    """Check if the given path is a HuggingFace Hub model ID rather than a local path.

    Args:
        model_path: The model path to check

    Returns:
        True if it appears to be a HuggingFace model ID, False if it's a local path

    Examples:
        >>> is_huggingface_model_id("microsoft/DialoGPT-medium")
        True
        >>> is_huggingface_model_id("/path/to/local/model")
        False
        >>> is_huggingface_model_id("./local_model")
        False
    """
    # If it's a valid local directory, it's not a Hub model ID
    if os.path.isdir(model_path):
        return False

    # Check for obvious local path indicators
    local_path_indicators = ["./", "../", "~/", "\\", ":", "C:", "D:", "/home/", "/usr/", "/opt/"]
    if any(model_path.startswith(indicator) for indicator in local_path_indicators):
        return False

    # If it contains OS-specific path separators, it's likely a local path
    if os.path.sep in model_path or (os.path.altsep and os.path.altsep in model_path):
        # Exception: if it doesn't exist locally and looks like org/model format, might be Hub ID
        return (
            not os.path.exists(model_path)
            and "/" in model_path
            and model_path.count("/") == 1
            and not model_path.startswith("/")
        )

    # If it contains exactly one forward slash and looks like org/model format, likely a Hub ID
    if "/" in model_path and model_path.count("/") == 1 and not model_path.startswith("/"):
        # Additional check: Hub model IDs typically don't contain certain characters
        invalid_chars = ["\\", ":", "*", "?", '"', "<", ">", "|"]
        if not any(char in model_path for char in invalid_chars):
            # Make sure it doesn't look like a local relative path
            return not model_path.startswith(("./", "../"))

    return False


def resolve_model_path(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    download_files: bool = True,
    allow_patterns: list[str] | None = None,
) -> str:
    """Resolve a model name or path to a local directory path.

    If the input is already a local directory, returns it as-is.
    If the input is a HuggingFace model ID, attempts to resolve it to the local cache path.

    Args:
        model_name_or_path: Either a local directory path or HuggingFace model ID
        trust_remote_code: Whether to trust remote code when loading the model
        download_files: Whether to download files if not found in cache
        allow_patterns: List of file patterns to download (e.g., ["*.py", "*.json"])
                       If None, downloads common model files

    Returns:
        Local directory path to the model files

    Raises:
        ValueError: If the model path cannot be resolved and download_files is False
        ImportError: If required packages (transformers, huggingface_hub) are not available

    Examples:
        >>> # Local path (returned as-is)
        >>> resolve_model_path("/path/to/local/model")
        '/path/to/local/model'

        >>> # HuggingFace model ID (resolved to cache)
        >>> resolve_model_path("microsoft/DialoGPT-medium")
        '/home/user/.cache/huggingface/hub/models--microsoft--DialoGPT-medium/snapshots/abc123'
    """
    # If it's already a local directory, return as-is
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    # If it's not a HuggingFace model ID, return as-is (might be a local path that doesn't exist yet)
    if not is_huggingface_model_id(model_name_or_path):
        return model_name_or_path

    # Handle HuggingFace model ID
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers package is required for resolving HuggingFace model IDs. "
            "Install it with: pip install transformers"
        )

    try:
        # First try to load the config to trigger caching
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

        # The config object should have the local path information
        if hasattr(config, "_name_or_path") and os.path.isdir(config._name_or_path):
            return config._name_or_path

    except Exception as e:
        warnings.warn(f"Could not load config for {model_name_or_path}: {e}")

    # Try to use snapshot_download if available and download_files is True
    if download_files and HF_HUB_AVAILABLE and snapshot_download is not None:
        try:
            if allow_patterns is None:
                allow_patterns = ["*.py", "*.json", "*.txt", "*.md"]  # Common model files

            local_path = snapshot_download(
                repo_id=model_name_or_path,
                allow_patterns=allow_patterns,
            )
            return local_path
        except Exception as e:
            warnings.warn(f"Could not download model files using snapshot_download: {e}")

    # Fallback: try to find in HuggingFace cache
    if TRANSFORMERS_CACHE:
        try:
            # Look for the model in the cache directory
            cache_pattern = os.path.join(TRANSFORMERS_CACHE, "models--*")
            cache_dirs = glob.glob(cache_pattern)

            # Convert model name to cache directory format
            model_cache_name = model_name_or_path.replace("/", "--")
            for cache_dir in cache_dirs:
                if model_cache_name in cache_dir:
                    # Look for the snapshots directory
                    snapshots_dir = os.path.join(cache_dir, "snapshots")
                    if os.path.exists(snapshots_dir):
                        # Get the latest snapshot
                        snapshot_dirs = [
                            d
                            for d in os.listdir(snapshots_dir)
                            if os.path.isdir(os.path.join(snapshots_dir, d))
                        ]
                        if snapshot_dirs:
                            latest_snapshot = max(snapshot_dirs)  # Use lexicographically latest
                            snapshot_path = os.path.join(snapshots_dir, latest_snapshot)
                            return snapshot_path

        except Exception as e:
            warnings.warn(f"Could not search HuggingFace cache for {model_name_or_path}: {e}")

    # If all else fails and we're not supposed to download, raise an error
    if not download_files:
        raise ValueError(
            f"Could not resolve model path for {model_name_or_path} and download_files=False"
        )

    # Last resort: return the original path (might work for some use cases)
    warnings.warn(f"Could not resolve model path for {model_name_or_path}, returning original path")
    return model_name_or_path


def fetch_model_config(
    model_id: str,
    filename: str = "config.json",
    trust_remote_code: bool = False,
) -> dict | None:
    """Fetch a configuration file from either a local path or HuggingFace Hub.

    Args:
        model_id: Either a local directory path or HuggingFace model ID
        filename: Name of the config file to fetch (default: "config.json")
        trust_remote_code: Whether to trust remote code when loading

    Returns:
        The configuration dictionary if successful, None otherwise

    Examples:
        >>> # Fetch from local path
        >>> config = fetch_model_config("/path/to/local/model")

        >>> # Fetch from HuggingFace Hub
        >>> config = fetch_model_config("microsoft/DialoGPT-medium")
    """
    # Try local path first
    if not is_huggingface_model_id(model_id):
        config_file = Path(model_id) / filename
        if config_file.exists():
            try:
                with open(config_file) as f:
                    return json.load(f)
            except Exception as e:
                warnings.warn(f"Could not load config from {config_file}: {e}")
                return None
        else:
            warnings.warn(f"Config file not found: {config_file}")
            return None

    # Handle HuggingFace model ID
    if not HF_HUB_AVAILABLE:
        warnings.warn(
            "huggingface_hub is not available. Cannot fetch config from Hub. "
            "Install it with: pip install huggingface_hub"
        )
        return None

    try:
        # Download only the specific config file
        config_path = hf_hub_download(repo_id=model_id, filename=filename, repo_type="model")

        with open(config_path) as f:
            return json.load(f)

    except Exception as e:
        warnings.warn(f"Could not fetch {filename} from HuggingFace Hub for {model_id}: {e}")
        return None


class ModelPathResolver:
    """A context manager and utility class for resolving model paths consistently.

    This class provides a convenient interface for handling model paths throughout
    a workflow, with caching and consistent behavior.

    Args:
        model_name_or_path: Either a local directory path or HuggingFace model ID
        trust_remote_code: Whether to trust remote code when loading
        download_files: Whether to download files if not found in cache
        allow_patterns: List of file patterns to download

    Examples:
        >>> # Use as context manager
        >>> with ModelPathResolver("microsoft/DialoGPT-medium") as resolver:
        ...     local_path = resolver.local_path
        ...     config = resolver.get_config()

        >>> # Use as regular class
        >>> resolver = ModelPathResolver("microsoft/DialoGPT-medium")
        >>> local_path = resolver.resolve()
    """

    def __init__(
        self,
        model_name_or_path: str,
        trust_remote_code: bool = False,
        download_files: bool = True,
        allow_patterns: list[str] | None = None,
    ):
        """Initialize the ModelPathResolver.

        Args:
            model_name_or_path: Either a local directory path or HuggingFace model ID
            trust_remote_code: Whether to trust remote code when loading
            download_files: Whether to download files if not found in cache
            allow_patterns: List of file patterns to download
        """
        self.model_name_or_path = model_name_or_path
        self.trust_remote_code = trust_remote_code
        self.download_files = download_files
        self.allow_patterns = allow_patterns
        self._local_path: str | None = None
        self._is_hub_id: bool | None = None

    @property
    def is_huggingface_model_id(self) -> bool:
        """Check if the model path is a HuggingFace Hub model ID."""
        if self._is_hub_id is None:
            self._is_hub_id = is_huggingface_model_id(self.model_name_or_path)
        return self._is_hub_id

    @property
    def local_path(self) -> str:
        """Get the resolved local path."""
        if self._local_path is None:
            self._local_path = self.resolve()
        return self._local_path

    def resolve(self) -> str:
        """Resolve the model path to a local directory."""
        return resolve_model_path(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            download_files=self.download_files,
            allow_patterns=self.allow_patterns,
        )

    def get_config(self, filename: str = "config.json") -> dict | None:
        """Fetch a configuration file."""
        return fetch_model_config(
            self.model_name_or_path,
            filename=filename,
            trust_remote_code=self.trust_remote_code,
        )

    def get_file_path(self, filename: str) -> Path:
        """Get the path to a specific file in the model directory."""
        return Path(self.local_path) / filename

    def file_exists(self, filename: str) -> bool:
        """Check if a specific file exists in the model directory."""
        return self.get_file_path(filename).exists()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""

    def __str__(self) -> str:
        """String representation."""
        return f"ModelPathResolver({self.model_name_or_path})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"ModelPathResolver(model_name_or_path='{self.model_name_or_path}', "
            f"trust_remote_code={self.trust_remote_code}, "
            f"download_files={self.download_files})"
        )
