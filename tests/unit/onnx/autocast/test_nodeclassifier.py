# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnxruntime import __version__ as ort_version
from packaging.version import Version

from modelopt.onnx.autocast.logging_config import configure_logging
from modelopt.onnx.autocast.nodeclassifier import (
    DisabledNodeNameRegexRule,
    InitializerRangeRule,
    IORangeRule,
    NodeClassifier,
)

configure_logging("DEBUG")


@pytest.fixture
def test_model():
    # Create a simple model with Add and Mul operations
    node1 = helper.make_node("Add", ["X", "Y"], ["add_out"], name="add_node")
    node2 = helper.make_node("Mul", ["add_out", "Z"], ["output"], name="mul_node")

    # Create graph inputs
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 2])

    # Create initializers
    y_init = numpy_helper.from_array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), name="Y")
    z_init = numpy_helper.from_array(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32), name="Z")

    graph = helper.make_graph(
        [node1, node2],
        "test_model",
        [x],
        [output],
        initializer=[y_init, z_init],
    )

    model = helper.make_model(graph)
    model.opset_import[0].version = 20
    model.ir_version = 10
    return model


def test_disabled_node_name_regex_rule():
    rule = DisabledNodeNameRegexRule([r".*add.*", r".*layer\.1.*"])
    node1 = helper.make_node("Add", ["X", "Y"], ["Z"], name="/test/layer.0/add_1")
    node2 = helper.make_node("Mul", ["A", "B"], ["C"], name="/test/layer.1/mul_2")
    node3 = helper.make_node("Sub", ["D", "E"], ["F"], name="/test/layer.2/sub_1")
    node4 = helper.make_node("Mul", ["G", "H"], ["I"], name="/test/layer.2/mul_2")

    assert rule.check(node1) is True
    assert rule.check(node2) is True
    assert rule.check(node3) is False
    assert rule.check(node4) is False


def test_initializer_range_rule():
    # Create test initializers
    init1 = numpy_helper.from_array(np.array([900], dtype=np.float32))
    init2 = numpy_helper.from_array(np.array([1100], dtype=np.float32))
    init3 = numpy_helper.from_array(np.array([-900], dtype=np.float32))
    init4 = numpy_helper.from_array(np.array([-1100], dtype=np.float32))

    node_to_init_map = {"node1": [init1], "node2": [init2], "node3": [init3], "node4": [init4]}

    rule = InitializerRangeRule(1000, node_to_init_map)

    node1 = helper.make_node("TestOp", [], [], name="node1")
    node2 = helper.make_node("TestOp", [], [], name="node2")
    node3 = helper.make_node("TestOp", [], [], name="node3")
    node4 = helper.make_node("TestOp", [], [], name="node4")

    assert rule.check(node1) is False
    assert rule.check(node2) is True
    assert rule.check(node3) is False
    assert rule.check(node4) is True


def test_output_range_rule():
    reference_outputs = {
        "out1": np.array([9000], dtype=np.float32),
        "out2": np.array([11000], dtype=np.float32),
        "out3": np.array([-9000], dtype=np.float32),
        "out4": np.array([-11000], dtype=np.float32),
    }

    rule = IORangeRule(10000, reference_outputs, node_to_init_map={})

    node1 = helper.make_node("TestOp", [], ["out1"], name="node1")
    node2 = helper.make_node("TestOp", [], ["out2"], name="node2")
    node3 = helper.make_node("TestOp", [], ["out3"], name="node3")
    node4 = helper.make_node("TestOp", [], ["out4"], name="node4")
    node5 = helper.make_node("TestOp", ["out1"], [], name="node5")
    node6 = helper.make_node("TestOp", ["out2"], [], name="node6")
    node7 = helper.make_node("TestOp", ["out3"], [], name="node7")
    node8 = helper.make_node("TestOp", ["out4"], [], name="node8")

    assert rule.check(node1) is False
    assert rule.check(node2) is True
    assert rule.check(node3) is False
    assert rule.check(node4) is True
    assert rule.check(node5) is False
    assert rule.check(node6) is True
    assert rule.check(node7) is False
    assert rule.check(node8) is True


@pytest.mark.skipif(
    Version(ort_version) < Version("1.21.0"), reason="WAR: Requires onnxruntime>=1.21.0"
)
def test_node_classifier(test_model):
    node_to_init_map = {key: [] for key in ["add_node", "mul_node"]}

    # Test with data range constraints
    classifier = NodeClassifier(
        model=test_model,
        node_to_init_map=node_to_init_map,
        data_max=3.0,
    )

    # First set of inputs
    # add_out = X + Y : requires fp32
    fp16_nodes, fp32_nodes = classifier.run(
        calibration_data={"X": np.array([[0.2, 0.3], [0.3, 0.4]], dtype=np.float32)}
    )
    # Compute expected outputs for first set of inputs
    assert "add_node" in fp32_nodes
    assert "mul_node" in fp32_nodes  # add_out is input of mul_node
    assert len(fp16_nodes) + len(fp32_nodes) == 2
    # Test that no node is in both fp16 and fp32 lists
    assert not set(fp16_nodes).intersection(set(fp32_nodes))

    # Second set of inputs
    # all within fp16 range
    fp16_nodes, fp32_nodes = classifier.run(
        calibration_data={"X": np.array([[-0.5, -1.5], [-3, -4]], dtype=np.float32)},
    )
    assert "add_node" in fp16_nodes
    assert "mul_node" in fp16_nodes
    assert len(fp16_nodes) + len(fp32_nodes) == 2
    # Test that no node is in both fp16 and fp32 lists
    assert not set(fp16_nodes).intersection(set(fp32_nodes))


@pytest.mark.skipif(
    Version(ort_version) < Version("1.21.0"), reason="WAR: Requires onnxruntime>=1.21.0"
)
def test_node_classifier_custom_rule(test_model):
    node_to_init_map = {
        "add_node": [numpy_helper.from_array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))],
        "mul_node": [numpy_helper.from_array(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32))],
    }

    from modelopt.onnx.autocast.nodeclassifier import NodeRuleBase

    class CustomRule(NodeRuleBase):
        def _check_inner(self, node):
            # Return True if any initializer contains zeros
            print(node.input)
            return any("X" in input for input in node.input)

    classifier = NodeClassifier(
        model=test_model,
        node_to_init_map=node_to_init_map,
        custom_rule=CustomRule(),
    )

    fp16_nodes, fp32_nodes = classifier.run()
    assert "add_node" in fp32_nodes
    assert "mul_node" in fp16_nodes
    assert len(fp16_nodes) + len(fp32_nodes) == 2
    # Test that no node is in both fp16 and fp32 lists
    assert not set(fp16_nodes).intersection(set(fp32_nodes))


@pytest.mark.skipif(
    Version(ort_version) < Version("1.21.0"), reason="WAR: Requires onnxruntime>=1.21.0"
)
def test_node_classifier_op_types_to_exclude(test_model):
    node_to_init_map = {
        "add_node": [numpy_helper.from_array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))],
        "mul_node": [numpy_helper.from_array(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32))],
    }
    classifier = NodeClassifier(
        model=test_model,
        node_to_init_map=node_to_init_map,
        op_types_to_exclude=["Add"],
    )
    fp16_nodes, fp32_nodes = classifier.run()
    assert "add_node" in fp32_nodes
    assert "mul_node" in fp16_nodes
    assert len(fp16_nodes) + len(fp32_nodes) == 2
    # Test that no node is in both fp16 and fp32 lists
    assert not set(fp16_nodes).intersection(set(fp32_nodes))

    classifier2 = NodeClassifier(
        model=test_model,
        node_to_init_map=node_to_init_map,
        op_types_to_exclude=["Mul"],
    )
    fp16_nodes, fp32_nodes = classifier2.run()
    assert "add_node" in fp16_nodes
    assert "mul_node" in fp32_nodes
    assert len(fp16_nodes) + len(fp32_nodes) == 2
    # Test that no node is in both fp16 and fp32 lists
    assert not set(fp16_nodes).intersection(set(fp32_nodes))
