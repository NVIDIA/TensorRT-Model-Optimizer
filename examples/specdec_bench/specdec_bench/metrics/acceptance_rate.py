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

import json
import os

from .base import Metric


class AcceptanceRate(Metric):
    def __init__(self):
        super().__init__()
        self.prompt_ar = []
        self.name = "acceptance_rate"

    def process_step(self, step_outputs, new_turn=True):
        if new_turn:
            self.prompt_ar.append([])
        for i, beam_output in enumerate(step_outputs["output_ids"]):
            for output_id_iter in beam_output:
                self.prompt_ar[-1].append(len(output_id_iter))

    def _get_lengths(self, turn, lengths):
        for j in turn:
            if j not in lengths:
                lengths[j] = 0
            lengths[j] += 1

    def _process_lengths(self, lengths):
        lengths = dict(sorted(lengths.items(), key=lambda x: x[0]))
        self.out["Acceptance_Length_Histogram"] = lengths
        print("Acceptance Length Histogram")
        print(lengths)
        sum_lengths = sum(lengths.values())
        running_len = sum_lengths
        prev_ratio = 1
        self.out["Conditional_Acceptance_Rate"] = {}
        print("Conditional acceptance rate")
        for k, v in lengths.items():
            print(k, running_len / sum_lengths / prev_ratio)
            self.out["Conditional_Acceptance_Rate"][k] = running_len / sum_lengths / prev_ratio
            prev_ratio = running_len / sum_lengths
            running_len -= v

    def process_final(self, text_outputs):
        i = 0
        lengths = {}
        self.out["Request_AR"] = {}
        while i < len(self.prompt_ar):
            turn_1 = self.prompt_ar[i]
            self.out["Request_AR"][i] = sum(turn_1) / len(turn_1)
            self._get_lengths(turn_1, lengths)
            print(i, self.out["Request_AR"][i])
            i += 1
        average_ar = sum(self.out["Request_AR"].values()) / len(self.out["Request_AR"])
        print("Average AR:", average_ar)
        self.out["Average_AR"] = average_ar
        self._process_lengths(lengths)
        self.write()
        self._format_write_output(text_outputs)

    def clear(self):
        self.prompt_ar = []

    def _format_write_output(self, outputs):
        with open(os.path.join(self.directory, "responses.jsonl"), "w") as outfile:
            for i, messages in enumerate(outputs):
                q_id = i
                out_line = {}
                out_line["question_id"] = q_id
                if messages[0]["role"] == "system":
                    out_line["system_prompt"] = messages[0]["content"]
                q_turns = [c["content"] for c in messages if c["role"] == "user"]
                a_turns = [c["content"] for c in messages if c["role"] == "assistant"]
                out_line["turns"] = q_turns
                out_line["choices"] = [{"index": 0, "turns": a_turns}]
                json.dump(out_line, outfile)
                outfile.write("\n")
