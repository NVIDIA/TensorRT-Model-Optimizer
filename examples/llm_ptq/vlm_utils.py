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

"""Utility functions for Vision-Language Model (VLM) inference and testing."""

import os

from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor


def run_vl_preview_generation(model, tokenizer, model_path, stage_name):
    """Run preview generation for VL models using sample images.

    Args:
        model: The VL model
        tokenizer: The tokenizer
        model_path: Path to the model (for loading image processor)
        stage_name: Description of the stage (e.g., "before quantization")

    Returns:
        Generated response text for logging/comparison
    """
    try:
        print(f"Loading sample images for {stage_name} preview...")

        # Load sample images from the images directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        images_dir = os.path.join(script_dir, "images")

        # Use single image for VL preview to avoid shape mismatch issues
        image_files = ["example1a.jpeg", "example1b.jpeg"]
        image = None
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            if os.path.exists(img_path):
                image = Image.open(img_path)
                print(f"  Loaded: {img_file}")
                break  # Use the first available image
            else:
                print(f"  Warning: {img_file} not found")

        if image is None:
            print("No sample images found - skipping VL preview generation")
            return None

        # Generate response
        question = "Describe this image briefly."  # Updated for single image
        generation_config = {
            "max_new_tokens": 50,
            "do_sample": False,
            "eos_token_id": tokenizer.eos_token_id,
        }

        print(f"Generating VL response ({stage_name})...")

        # Try to detect if this is a v1 model (has chat method) or v2 model (uses generate)
        if hasattr(model, "chat"):
            print("  Using v1 model.chat() method...")
            # Load image processor for v1 models
            image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)

            # Process single image for v1 models
            image_features = image_processor([image])  # Pass as list with single image

            # Move image features to the same device as the model
            model_device = model.device
            for key, value in image_features.items():
                if hasattr(value, "to"):  # Check if it's a tensor
                    image_features[key] = value.to(model_device)
                    print(f"    Moved {key} to {model_device}")

            response = model.chat(
                tokenizer=tokenizer,
                question=question,
                generation_config=generation_config,
                **image_features,
            )
        else:
            print("  Using v2 model.generate() method...")
            # Load processor for v2 models
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

            # Create messages in the format expected by v2 models
            messages = [
                {"role": "system", "content": "/no_think"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "",
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ]

            # Apply chat template
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process inputs using the processor with single image
            inputs = processor(
                text=[prompt],
                images=[image],  # Pass single image as list
                return_tensors="pt",
            )

            # Move inputs to the same device as the model
            model_device = model.device
            inputs = inputs.to(model_device)
            print(f"    Moved inputs to {model_device}")

            # Generate response using model.generate
            generated_ids = model.generate(
                pixel_values=inputs.pixel_values,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **generation_config,
            )

            # Decode the response (trim input tokens like in the working example)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = output_text[0]

        print(f"✅ VL generation {stage_name} successful!")
        print(f"Question: {question}")
        print(f"Response: {response}")

        # Return the response for comparison/logging
        return response

    except Exception as e:
        print(f"❌ VL preview generation {stage_name} failed: {e}")
        print("This may indicate issues with the quantized model")
        return None


def run_text_only_generation(model, tokenizer, question, generation_config, model_path):
    """Run text-only generation for VL models, supporting both v1 (chat) and v2 (generate) models.

    Args:
        model: The VL model
        tokenizer: The tokenizer
        question: The text question to ask
        generation_config: Generation configuration
        model_path: Path to the model (for loading processor if needed)

    Returns:
        Generated response text or None if failed
    """
    try:
        if hasattr(model, "chat"):
            print("  Using v1 model.chat() method for text-only generation...")
            # Use model.chat with None for images (text-only mode)
            response = model.chat(tokenizer, None, question, generation_config, history=None)
            return response
        else:
            print("  Using v2 model.generate() method for text-only generation...")
            # Load processor for v2 models
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

            # Create text-only messages
            messages = [
                {"role": "system", "content": "/no_think"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ]

            # Apply chat template
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process text-only inputs
            inputs = processor(
                text=[prompt],
                images=None,  # No images for text-only
                return_tensors="pt",
            )

            # Move inputs to the same device as the model
            model_device = model.device
            inputs = inputs.to(model_device)

            # Generate response using model.generate
            generated_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **generation_config,
            )

            # Decode the response (trim input tokens like in the working example)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]

    except Exception as e:
        print(f"Text-only generation failed: {e}")
        return None
