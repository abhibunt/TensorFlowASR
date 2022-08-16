"""
Gradient Accummulation for training TF2 custom training loop.
Copy and modified from https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/optimizers/utils.py.
"""

import tensorflow as tf


class GradientAccumulator:
    # We use the ON_READ synchronization policy so that no synchronization is
    # performed on assignment. To get the value, we call .value() which returns the
    # value on the current replica without synchronization.

    def __init__(self, ga_steps, name="ga"):
        self.name = name
        self._gradients = []
        self._accum_step = None
        if ga_steps is None:
            raise ValueError("ga_steps must be defined")
        self._ga_steps = tf.constant(ga_steps, dtype=tf.int64)

    @property
    def step(self):
        """Number of accumulated steps."""
        if self._accum_step is None:
            self._accum_step = tf.Variable(
                tf.constant(0, dtype=tf.int64),
                trainable=False,
                synchronization=tf.VariableSynchronization.ON_READ,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                name=f"{self.name}_accum_step",
            )
        return self._accum_step.value()

    @property
    def total_steps(self):
        return self._ga_steps

    @property
    def is_apply_step(self):
        return tf.equal(self.step, self.total_steps)

    @property
    def gradients(self):
        """The accumulated gradients on the current replica."""
        if not self._gradients:
            raise ValueError("The accumulator's accumulate should be called first to initialize the gradients")
        return tf.cond(  # zeros gradients so that apply_gradient has no effect
            self.is_apply_step,
            lambda: list(gradient.value() for gradient in self._gradients),
            lambda: list(tf.zeros_like(gradient) for gradient in self._gradients),
        )

    def accumulate(self, gradients):
        """Accumulates :obj:`gradients` on the current replica."""
        if not self._gradients:
            _ = self.step  # Create the step variable.
            self._gradients.extend(
                [
                    tf.Variable(
                        tf.zeros_like(gradient),
                        trainable=False,
                        synchronization=tf.VariableSynchronization.ON_READ,
                        aggregation=tf.VariableAggregation.NONE,
                        name=f"{self.name}_{i}",
                    )
                    for i, gradient in enumerate(gradients)
                ]
            )
        if len(gradients) != len(self._gradients):
            raise ValueError("Expected %s gradients, but got %d" % (len(self._gradients), len(gradients)))

        for accum_gradient, gradient in zip(self._gradients, gradients):
            accum_gradient.assign_add(gradient, read_value=False)
        self._accum_step.assign_add(1)

    def reset(self):
        """Resets the accumulated gradients on the current replica."""
        if not self._gradients:
            return
        self._accum_step.assign(0)
        for gradient in self._gradients:
            gradient.assign(tf.zeros_like(gradient), read_value=False)
