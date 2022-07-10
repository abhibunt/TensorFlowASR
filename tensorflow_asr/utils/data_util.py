# Copyright 2020 Huy Le Nguyen (@usimarit)
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
# limitations under the License.

# tf.data.Dataset does not work well for namedtuple so we are using dict


def create_inputs(
    inputs,
    inputs_length,
    predictions=None,
    predictions_length=None,
) -> dict:
    data = {
        "inputs": inputs,
        "inputs_length": inputs_length,
    }
    if predictions is not None:
        data["predictions"] = predictions
    if predictions_length is not None:
        data["predictions_length"] = predictions_length
    return data


def create_logits(
    logits,
    logits_length,
) -> dict:
    return {"logits": logits, "logits_length": logits_length}


def create_labels(
    labels,
    labels_length,
) -> dict:
    return {
        "labels": labels,
        "labels_length": labels_length,
    }
