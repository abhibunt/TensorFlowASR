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

import tensorflow as tf


class CtcLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        blank=0,
        name=None,
    ):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)
        self.blank = blank

    def call(self, y_true, y_pred):
        return tf.nn.ctc_loss(
            logits=tf.cast(y_pred["logits"], tf.float32),
            logit_length=tf.cast(y_pred["logits_length"], tf.int32),
            labels=tf.cast(y_true["labels"], tf.int32),
            label_length=tf.cast(y_true["labels_length"], tf.int32),
            logits_time_major=False,
            blank_index=self.blank,
            name=self.name,
        )
