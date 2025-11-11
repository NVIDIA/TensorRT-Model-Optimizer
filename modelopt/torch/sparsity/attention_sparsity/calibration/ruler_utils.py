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

# Copied and Adapted from https://github.com/NVIDIA/RULER
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""Official RULER dataset generation utilities adapted for Model Optimizer.

This module contains core logic from the RULER benchmark (https://github.com/NVIDIA/RULER)
adapted to work as a library for calibration purposes. The generation logic closely follows
the official RULER implementation to ensure dataset consistency.

Key adaptations from official RULER:
- Converted from CLI scripts to library functions
- Works with HuggingFace tokenizers directly
- Removed file I/O, returns data structures
- Simplified for calibration use case (primarily NIAH tasks)
"""

import json
import logging
import random
import re
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import nltk
import wonderwords
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

logger = logging.getLogger(__name__)


# Needle/Haystack template from official RULER
NEEDLE_TEMPLATE = "One of the special magic {type_needle_v} for {key} is: {value}."

# Depth positions for needle insertion (from official RULER)
DEPTHS = [
    0,
    2,
    5,
    7,
    10,
    12,
    15,
    18,
    20,
    23,
    25,
    28,
    30,
    33,
    35,
    38,
    40,
    43,
    45,
    48,
    50,
    53,
    55,
    58,
    60,
    62,
    65,
    67,
    70,
    72,
    75,
    77,
    80,
    82,
    85,
    87,
    90,
    92,
    95,
    97,
    100,
]

# Official RULER URL list
RULER_URLS_FILE = "https://raw.githubusercontent.com/NVIDIA/RULER/main/scripts/data/synthetic/json/PaulGrahamEssays_URLs.txt"


def _get_data_dir() -> Path:
    """Get data directory for RULER data.

    Returns:
        Path to data directory under calibration/ (created if doesn't exist)
    """
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _download_paul_graham_essays() -> str:
    """Download Paul Graham essays using official RULER URL list.

    Downloads the URL list from RULER repo, then fetches all essays.
    Uses only the GitHub .txt files for simplicity (no HTML parsing needed).

    Returns:
        Combined essay text
    """
    logger.info("Downloading Paul Graham essays for RULER calibration...")
    logger.info("This is a one-time download (~1-2MB), data will be cached.")

    # Step 1: Download URL list from official RULER
    try:
        with urllib.request.urlopen(RULER_URLS_FILE, timeout=10) as response:
            url_list_content = response.read().decode("utf-8")
            all_urls = [line.strip() for line in url_list_content.split("\n") if line.strip()]
    except Exception as e:
        raise RuntimeError(
            f"Failed to download RULER URL list from {RULER_URLS_FILE}: {e}\n"
            "Please check your internet connection."
        ) from e

    # Step 2: Filter to only GitHub .txt files (no HTML parsing needed)
    txt_urls = [
        url for url in all_urls if url.startswith("https://github.com") and url.endswith(".txt")
    ]

    logger.info(f"Found {len(txt_urls)} essay URLs from RULER")

    # Step 3: Download essays
    all_essays = []
    failed_urls = []

    for url in tqdm(txt_urls, desc="Downloading essays"):
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                text = response.read().decode("utf-8")
                all_essays.append(text)
        except Exception as e:
            logger.debug(f"Failed to download {url}: {e}")
            failed_urls.append(url)
            continue

    if not all_essays:
        raise RuntimeError(
            f"Failed to download any essays. Tried {len(txt_urls)} URLs.\n"
            "Please check your internet connection."
        )

    if failed_urls:
        logger.warning(f"Failed to download {len(failed_urls)}/{len(txt_urls)} essays")

    combined_text = " ".join(all_essays)
    logger.info(f"✓ Downloaded {len(all_essays)}/{len(txt_urls)} essays successfully")

    return combined_text


def _load_paul_graham_essays() -> str:
    """Load Paul Graham essays (auto-downloads on first use).

    Checks data directory, downloads if not present.

    Returns:
        Essay text as string
    """
    data_dir = _get_data_dir()
    essay_file = data_dir / "PaulGrahamEssays.json"

    # Check if already downloaded
    if essay_file.exists():
        with open(essay_file) as f:
            essay_data = json.load(f)
            return re.sub(r"\s+", " ", essay_data["text"])

    # Auto-download
    logger.info("First-time setup: downloading Paul Graham essays...")
    essay_text = _download_paul_graham_essays()

    # Save for future use
    with open(essay_file, "w") as f:
        json.dump({"text": essay_text}, f)
    logger.info(f"✓ Essays saved to {essay_file}")

    return re.sub(r"\s+", " ", essay_text)


def _load_word_lists():
    """Load word lists for random word generation.

    Returns:
        List of words (adj-noun combinations)
    """
    # Load wonderwords lists (same as official RULER)
    nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
    adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
    words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
    words = sorted(set(words))
    return words


# Global word list (loaded once)
_WORD_LIST = None


def generate_random_number(num_digits=7) -> str:
    """Generate random number (from official RULER)."""
    lower_bound = 10 ** (num_digits - 1)
    upper_bound = 10**num_digits - 1
    return str(random.randint(lower_bound, upper_bound))


def generate_random_word() -> str:
    """Generate random word (from official RULER)."""
    global _WORD_LIST
    if _WORD_LIST is None:
        _WORD_LIST = _load_word_lists()
    return random.choice(_WORD_LIST)


def generate_random_uuid() -> str:
    """Generate random UUID (from official RULER)."""
    return str(uuid.UUID(int=random.getrandbits(128), version=4))


def generate_random(type_needle: str) -> str:
    """Generate random needle value based on type (from official RULER).

    Args:
        type_needle: Type of needle ('numbers', 'words', 'uuids')

    Returns:
        Random value as string
    """
    if type_needle == "numbers":
        return generate_random_number()
    elif type_needle == "words":
        return generate_random_word()
    elif type_needle == "uuids":
        return generate_random_uuid()
    else:
        raise ValueError(f"Unknown needle type: {type_needle}")


def generate_niah_sample(
    num_haystack: int,
    tokenizer,
    template: str,
    answer_prefix: str,
    tokens_to_generate: int = 128,
    type_haystack: str = "essay",
    type_needle_k: str = "words",
    type_needle_v: str = "numbers",
    num_needle_k: int = 1,
    num_needle_v: int = 1,
    num_needle_q: int = 1,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Generate a single NIAH (Needle in a Haystack) sample.

    This function implements the core generation logic from official RULER's niah.py,
    adapted to work as a library function.

    Args:
        num_haystack: Number of haystack items/words
        tokenizer: HuggingFace tokenizer (AutoTokenizer instance)
        template: NIAH question template
        answer_prefix: Answer prefix template
        tokens_to_generate: Expected number of generation tokens
        type_haystack: Type of haystack ('essay', 'noise', 'needle')
        type_needle_k: Type of needle keys ('numbers', 'words', 'uuids')
        type_needle_v: Type of needle values ('numbers', 'words', 'uuids')
        num_needle_k: Number of needle keys
        num_needle_v: Number of needle values per key
        num_needle_q: Number of needles to query
        random_seed: Random seed for this sample

    Returns:
        Dictionary with 'input', 'outputs', 'length' keys
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Ensure num_needle_k >= num_needle_q
    num_needle_k = max(num_needle_k, num_needle_q)

    # Generate needles (keys and values)
    keys, values, needles = [], [], []
    for _ in range(num_needle_k):
        keys.append(generate_random(type_needle_k))
        value = []
        for _ in range(num_needle_v):
            value.append(generate_random(type_needle_v))
            needles.append(
                NEEDLE_TEMPLATE.format(
                    type_needle_v=type_needle_v,
                    key=keys[-1],
                    value=value[-1],
                )
            )
        values.append(value)

    random.shuffle(needles)

    # Generate context based on haystack type
    if type_haystack == "essay":
        # Load essay corpus
        essay_text = _load_paul_graham_essays()
        haystack = essay_text.split(" ")

        # Create text from haystack
        if num_haystack <= len(haystack):
            text = " ".join(haystack[:num_haystack])
        else:
            # Repeat haystack as needed
            repeats = (num_haystack + len(haystack) - 1) // len(haystack)
            text = " ".join((haystack * repeats)[:num_haystack])

        # Insert needles at various depths
        document_sents = sent_tokenize(text.strip())
        insertion_positions = [
            0,
            *sorted(
                int(len(document_sents) * (depth / 100))
                for depth in random.sample(DEPTHS, len(needles))
            ),
            len(document_sents),
        ]

        document_sents_list = []
        for i in range(1, len(insertion_positions)):
            last_pos = insertion_positions[i - 1]
            next_pos = insertion_positions[i]
            document_sents_list.append(" ".join(document_sents[last_pos:next_pos]))
            if i - 1 < len(needles):
                document_sents_list.append(needles[i - 1])

        context = " ".join(document_sents_list)

    if type_haystack == "noise":
        haystack_sent = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
        sentences = [haystack_sent] * num_haystack
        indexes = sorted(random.sample(range(num_haystack), len(needles)), reverse=True)
        for index, element in zip(indexes, needles):
            sentences.insert(index, element)
        context = "\n".join(sentences)

    elif type_haystack == "needle":
        sentences = [
            NEEDLE_TEMPLATE.format(
                type_needle_v=type_needle_v,
                key=generate_random(type_needle_k),
                value=generate_random(type_needle_v),
            )
            for _ in range(num_haystack)
        ]

        indexes = sorted(random.sample(range(num_haystack), len(needles)), reverse=True)
        for index, element in zip(indexes, needles):
            sentences.insert(index, element)
        context = "\n".join(sentences)

    # Generate query and answer
    indices = random.sample(range(num_needle_k), num_needle_q)
    queries = [keys[i] for i in indices]
    answers = [a for i in indices for a in values[i]]
    query = ", ".join(queries[:-1]) + ", and " + queries[-1] if len(queries) > 1 else queries[0]

    # Format template (adjust for singular vs plural)
    type_needle_v_display = type_needle_v
    formatted_template = template
    if num_needle_q * num_needle_v == 1:
        formatted_template = formatted_template.replace("Some", "A")
        formatted_template = formatted_template.replace("are all", "is")
        formatted_template = formatted_template.replace("are", "is")
        formatted_template = formatted_template.replace("answers", "answer")
        type_needle_v_display = type_needle_v[:-1]  # remove "s"

    input_text = formatted_template.format(
        type_needle_v=type_needle_v_display,
        context=context,
        query=query,
    )

    # Add answer prefix
    formatted_answer_prefix = answer_prefix.format(
        type_needle_v=type_needle_v_display,
        query=query,
    )
    input_text = input_text + formatted_answer_prefix

    # Calculate actual length
    if hasattr(tokenizer, "encode"):
        # HuggingFace tokenizer
        tokens = tokenizer.encode(input_text, add_special_tokens=False)
        length = len(tokens) + tokens_to_generate
    else:
        # Fallback
        length = len(input_text.split()) + tokens_to_generate

    return {
        "input": input_text,
        "outputs": answers,
        "length": length,
    }


def find_optimal_haystack_size(
    tokenizer,
    max_seq_length: int,
    template: str,
    answer_prefix: str,
    tokens_to_generate: int = 128,
    type_haystack: str = "essay",
    **kwargs,
) -> int:
    """Find optimal haystack size using binary search (from official RULER).

    Args:
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length
        tokens_to_generate: Expected generation tokens
        type_haystack: Type of haystack
        template: NIAH question template
        answer_prefix: Answer prefix template
        **kwargs: Additional arguments for generate_niah_sample

    Returns:
        Optimal number of haystack items
    """
    # Determine incremental step based on haystack type
    if type_haystack == "essay":
        incremental = 500
    elif type_haystack in ["noise", "needle"]:
        incremental = 25
    else:
        incremental = 100

    if max_seq_length < 4096 and type_haystack != "essay":
        incremental = 5

    # Estimate tokens per haystack item
    sample = generate_niah_sample(
        incremental,
        tokenizer,
        template,
        answer_prefix,
        tokens_to_generate,
        type_haystack=type_haystack,
        **kwargs,
    )

    if hasattr(tokenizer, "encode"):
        sample_tokens = len(tokenizer.encode(sample["input"], add_special_tokens=False))
    else:
        sample_tokens = len(sample["input"].split())

    tokens_per_haystack = sample_tokens / incremental
    estimated_max = int((max_seq_length / tokens_per_haystack) * 3)

    # Binary search for optimal size
    lower_bound = incremental
    upper_bound = max(estimated_max, incremental * 2)
    optimal_num_haystack = None

    logger.info(f"Estimated {tokens_per_haystack:.1f} tokens per haystack")
    logger.info(f"Binary search bounds: {lower_bound} to {upper_bound}")

    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        sample = generate_niah_sample(
            mid,
            tokenizer,
            template,
            answer_prefix,
            tokens_to_generate,
            type_haystack=type_haystack,
            **kwargs,
        )
        total_tokens = sample["length"]

        logger.debug(f"Testing haystack size: {mid}, tokens: {total_tokens}/{max_seq_length}")

        if total_tokens <= max_seq_length:
            optimal_num_haystack = mid
            lower_bound = mid + 1
        else:
            upper_bound = mid - 1

    final_size = optimal_num_haystack if optimal_num_haystack is not None else incremental
    logger.info(f"Optimal haystack size: {final_size}")

    return final_size
