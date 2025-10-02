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

import modelopt.torch.nas as mtn
from modelopt.torch.nas.demonas import MLP


def test_demonas_convert_and_search():
    model = MLP(mlp_inner=40, hidden_size=10)  # original params = 2*10*40 = 800

    # Convert to search space using DemoNAS algorithm
    converted = mtn.convert(model, mode=[("demonas", {"mlp_inner_choices": [10, 20, 30, 40]})])

    # Train the converted model

    # Run search using DemoNAS: pick inner width under params=600 and trim weights
    searched_model, _ = mtn.search(
        converted,
        constraints={"params": 600},
        dummy_input=None,  # Not used
        config={},
    )


if __name__ == "__main__":
    test_demonas_convert_and_search()
