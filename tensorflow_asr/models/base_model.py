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

from tensorflow_asr.utils import env_util, file_util


class BaseModel(tf.keras.Model):
    def summary(
        self,
        line_length=127,
        expand_nested=True,
        show_trainable=True,
        **kwargs,
    ):
        super().summary(line_length=line_length, expand_nested=expand_nested, show_trainable=show_trainable, **kwargs)

    def save(
        self,
        filepath,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    ):
        with file_util.save_file(filepath) as path:
            super().save(
                filepath=path,
                overwrite=overwrite,
                include_optimizer=include_optimizer,
                save_format=save_format,
                signatures=signatures,
                options=options,
                save_traces=save_traces,
            )

    def save_weights(
        self,
        filepath,
        overwrite=True,
        save_format=None,
        options=None,
    ):
        with file_util.save_file(filepath) as path:
            super().save_weights(filepath=path, overwrite=overwrite, save_format=save_format, options=options)

    def load_weights(
        self,
        filepath,
        by_name=False,
        skip_mismatch=False,
        options=None,
    ):
        with file_util.read_file(filepath) as path:
            super().load_weights(filepath=path, by_name=by_name, skip_mismatch=skip_mismatch, options=options)

    @property
    def metrics(self):
        if not hasattr(self, "_tfasr_metrics"):
            self._tfasr_metrics = {}
        return list(self._tfasr_metrics.values())

    def add_metric(
        self,
        metric: tf.keras.metrics.Metric,
    ):
        if not hasattr(self, "_tfasr_metrics"):
            self._tfasr_metrics = {}
        self._tfasr_metrics[metric.name] = metric

    def make(self, *args, **kwargs):
        """Custom function for building model (uses self.build so cannot overwrite that function)"""
        raise NotImplementedError()

    def compile(
        self,
        loss,
        optimizer,
        run_eagerly=None,
        ga_steps=None,
        **kwargs,
    ):
        if ga_steps:
            if not isinstance(ga_steps, int) or ga_steps < 0:
                raise ValueError("ga_steps must be integer > 0")
            self.use_ga = True
            self.ga_steps = tf.constant(ga_steps, dtype=tf.int32)
            self.ga_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False, name=f"{self.name}_ga_accum_step")
            self.ga = [
                tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False, name=f"{self.name}_ga_{i}")
                for i, v in enumerate(self.trainable_variables)
            ]
        else:
            self.use_ga = False

        self.use_loss_scale = False
        if not env_util.has_devices("TPU"):
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.get(optimizer), dynamic=True)
            self.use_loss_scale = True

        self.add_metric(metric=tf.keras.metrics.Mean(name="loss", dtype=tf.float32))

        super().compile(optimizer=optimizer, loss=loss, run_eagerly=run_eagerly, **kwargs)

    # -------------------------------- STEP FUNCTIONS -------------------------------------

    def train_step(self, batch):
        """
        Args:
            batch ([tf.Tensor]): a batch of training data

        Returns:
            Dict[tf.Tensor]: a dict of validation metrics with keys are the name of metric

        """
        inputs, y_true = batch

        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            loss = self.loss(y_true, y_pred)
            if self.use_loss_scale:
                scaled_loss = self.optimizer.get_scaled_loss(loss)

        if self.use_loss_scale:
            gradients = tape.gradient(scaled_loss, self.trainable_weights)
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        else:
            gradients = tape.gradient(loss, self.trainable_weights)

        if self.use_ga:  # perform gradient accumulation
            apply = tf.equal(self.ga_acum_step, self.ga_steps)
            tf.cond(apply, lambda: self.ga_acum_step.assign(0), lambda: self.ga_acum_step.assign_add(1))
            for i, g in enumerate(self.ga):
                g.assign_add(gradients[i])
            self.optimizer.apply_gradients(zip(self.ga, self.trainable_variables))
            keep = tf.cond(apply, lambda: tf.constant(0, dtype=tf.float32), lambda: tf.constant(1, dtype=tf.float32))
            for g in self.ga:
                g.assign(g * keep)
        else:
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self._tfasr_metrics["loss"].update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, batch):
        """
        Args:
            batch ([tf.Tensor]: a batch of validation data

        Returns:
            Dict[tf.Tensor]: a dict of validation metrics with keys are the name of metric prefixed with "val_"

        """
        inputs, y_true = batch
        y_pred = self(inputs, training=False)
        loss = self.loss(y_true, y_pred)
        self._tfasr_metrics["loss"].update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, batch):
        """
        Args:
            batch ([tf.Tensor]): a batch of testing data

        Returns:
            [tf.Tensor]: stacked tensor of shape [B, 3] with each row is the text [truth, greedy, beam_search]
        """
        inputs, y_true = batch
        labels = self.text_featurizer.iextract(y_true["labels"])
        greedy_decoding = self.recognize(inputs)
        if self.text_featurizer.decoder_config.beam_width == 0:
            beam_search_decoding = tf.map_fn(lambda _: tf.convert_to_tensor("", dtype=tf.string), labels)
        else:
            beam_search_decoding = self.recognize_beam(inputs)
        return tf.stack([labels, greedy_decoding, beam_search_decoding], axis=-1)

    # -------------------------------- INFERENCE FUNCTIONS -------------------------------------

    def recognize(self, *args, **kwargs):
        """Greedy decoding function that used in self.predict_step"""
        raise NotImplementedError()

    def recognize_beam(self, *args, **kwargs):
        """Beam search decoding function that used in self.predict_step"""
        raise NotImplementedError()

    # ---------------------------------- TFLITE ---------------------------------- #

    def make_tflite_function(
        self,
        *args,
        **kwargs,
    ):
        pass
