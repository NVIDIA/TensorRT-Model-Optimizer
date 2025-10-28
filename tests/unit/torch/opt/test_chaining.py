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

import pytest
import torch
from _test_utils.torch.misc import compare_outputs
from _test_utils.torch.opt.utils import apply_mode_with_sampling
from torchvision.models.mobilenetv2 import InvertedResidual

import modelopt.torch.distill as mtd
import modelopt.torch.nas as mtn
import modelopt.torch.opt as mto
import modelopt.torch.sparsity as mts
from modelopt.torch.utils.distributed import _serialize


def get_model():
    return InvertedResidual(16, 32, 1, 6)


def get_input():
    return torch.randn(1, 16, 8, 8)


def get_kd_mode():
    config = {
        "teacher_model": get_model,
        "criterion": mtd.LogitsDistillationLoss(),
        "loss_balancer": mtd.StaticLossBalancer(),
    }
    return [("kd_loss", config)]


@pytest.mark.parametrize(
    "mode",
    [
        ["autonas"],
        ["autonas", "export_nas"],
        ["autonas", "export_nas", "fastnas"],
        ["autonas", "export_nas", "fastnas", "export_nas"],
        ["autonas", "export_nas", "fastnas", "export_nas", get_kd_mode()],
        ["autonas", "export_nas", "fastnas", "export_nas", get_kd_mode(), "export_student"],
        [
            "autonas",
            "export_nas",
            "fastnas",
            "export_nas",
            "quantize",
            get_kd_mode(),
            "export_student",
        ],
        [get_kd_mode(), "export_student", "fastnas", "export_nas", get_kd_mode(), "export_student"],
        ["quantize"],
        ["fastnas", get_kd_mode(), "export_student", "export_nas"],
        ["sparse_magnitude", get_kd_mode(), "export_student", "export_sparse"],
        ["sparse_magnitude", "quantize", get_kd_mode(), "export_student"],
        ["fastnas", "export_nas", "sparse_magnitude", "quantize", get_kd_mode(), "export_student"],
        ["fastnas", "quantize", get_kd_mode(), "export_student"],
        ["fastnas", "sparse_magnitude", "export_sparse", "export_nas"],
    ],
)
def test_chained_save_restore(mode):
    """Test whether we can save and restore a model."""
    # setup model
    model = get_model()
    model = apply_mode_with_sampling(model, mode)

    # store state
    modelopt_state = mto.modelopt_state(model)

    # restore model
    model2 = get_model()
    model2 = mto.restore_from_modelopt_state(model2, modelopt_state)
    model2.load_state_dict(model.state_dict())

    # compare serialized version since some configs may be objected...
    manager = mto.ModeloptStateManager(model)
    manager2 = mto.ModeloptStateManager(model2)
    assert torch.equal(_serialize(manager.state_dict()), _serialize(manager2.state_dict()))

    # run comparison in eval mode since there might be model randomization in train mode
    model.eval()
    model2.eval()
    dummy_input = get_input()
    output = model(dummy_input)
    output2 = model2(dummy_input)
    compare_outputs(output, output2)


@pytest.mark.parametrize(
    ("mode", "error_msg"),
    [
        (
            ["export_nas"],
            [r"Cannot add export_nas according to the current export stack: deque\(\[.*\]\)."],
        ),
        (
            ["autonas", "fastnas"],
            [r"Cannot add fastnas after autonas! Next modes of autonas are \{.*\}."],
        ),
        (
            ["fastnas", "export_nas", "export_student"],
            [r"Cannot add export_student according to the current export stack: deque\(\[.*\]\)."],
        ),
        (
            ["quantize", "export_nas"],
            [r"Cannot add export_nas according to the current export stack: deque\(\[.*\]\)."],
        ),
        (
            ["quantize", "fastnas"],
            [
                "Cannot add fastnas after quantize! quantize does not allow fastnas to be its next mode."
            ],
        ),
        (
            ["fastnas", get_kd_mode(), "export_nas", "export_student"],
            [r"Cannot add export_nas according to the current export stack: deque\(\[.*\]\)."],
        ),
    ],
)
def test_invalid_chaining(mode, error_msg):
    """Test whether apply_mode will raise an error for this combination of mode(s)."""
    model = get_model()
    with pytest.raises(AssertionError, match=error_msg[0]):
        model = apply_mode_with_sampling(model, mode)


@pytest.mark.parametrize(
    ("mode", "modellike", "expect_exception"),
    [
        ("autonas", True, False),
        ("autonas", False, False),
        (
            [
                (
                    "sparsegpt",
                    {},
                )
            ],
            True,
            False,
        ),
        (
            [
                (
                    "sparsegpt",
                    {},
                )
            ],
            False,
            False,
        ),
        ("quantize", True, False),
        ("quantize", False, False),
    ],
)
def test_model_like_initialization(mode, modellike, expect_exception):
    model = (get_model, (), {}) if modellike else get_model()

    if expect_exception:
        with pytest.raises(Exception):
            model = mto.apply_mode(model, mode=mode, init_state=True)
    else:
        model = mto.apply_mode(model, mode=mode, init_state=True)
        assert isinstance(model, torch.nn.Module)

    modelopt_state = mto.modelopt_state(model)
    model2 = (get_model, (), {}) if modellike else get_model()

    model2 = mto.restore_from_modelopt_state(model2, modelopt_state)
    assert isinstance(model2, torch.nn.Module)


def test_sparse_quantized_module():
    model = get_model()
    modes = ["fastnas", "sparse_magnitude"]

    # apply modes one-by-one with sampling in between to mix up architecture
    model = apply_mode_with_sampling(model, modes)

    # get conv[0][0] since this one can be pruned and sparsified
    conv = model.conv[0][0]
    out_c, in_c, k1, k2 = conv.out_channels, conv.in_channels, *conv.kernel_size
    out_c_og = conv.get_hparam("out_channels").original
    in_c_og = conv.get_hparam("in_channels").original

    # sparsify with a random mask
    mask = torch.rand_like(conv.weight) > 0.5
    conv.set_mask(mask)

    # now check for expected weights to check stacked callback functionality
    weight_og = conv._parameters["weight"]

    # check if pruning works as expected
    assert weight_og.shape == (out_c_og, in_c_og, k1, k2)
    assert conv.weight.shape == (out_c, in_c, k1, k2)

    # check if sparsification on top of that works as expected
    assert torch.equal(conv.weight != 0.0, mask)

    # now try to manually apply expected callbacks to see if everything works
    weight_expected = mask * weight_og[:out_c, :in_c]
    assert torch.equal(conv.weight, weight_expected)

    # now try to export the model, the expected weights should not change
    model = mts.export(model)
    assert torch.equal(conv.weight, weight_expected)

    # check expected weight now...
    assert conv._parameters["weight"] is weight_og, "The weight should still be the original!"
    assert torch.equal(conv.weight, weight_expected)

    # do the final export (now the actual weight is also overwritten)
    model = mtn.export(model)
    assert torch.equal(conv.weight, weight_expected)
    assert torch.equal(conv._parameters["weight"], weight_expected), "Weight should be overwritten!"
