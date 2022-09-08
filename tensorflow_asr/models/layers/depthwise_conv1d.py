import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops


class DepthwiseConv1D(tf.keras.layers.DepthwiseConv1D):
    """
    Causal padding supported DepthwiseConv1D
    Slightly modified with input_shape specific for ASR
    """

    def _validate_init(self):  # removed check padding causal
        if self.filters is not None and self.filters % self.groups != 0:
            raise ValueError(
                "The number of filters must be evenly divisible by the number of groups. Received: groups={}, filters={}".format(
                    self.groups, self.filters
                )
            )
        if not all(self.kernel_size):
            raise ValueError("The argument `kernel_size` cannot contain 0(s). Received: %s" % (self.kernel_size,))
        if not all(self.strides):
            raise ValueError("The argument `strides` cannot contains 0(s). Received: %s" % (self.strides,))

    def call(self, inputs):
        # input will be in shape [B, T, E] for channel_last or [B, E, T] for channel_first
        if self._is_causal:
            inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))

        if self.data_format == "channels_last":  # default
            strides = (1,) + self.strides * 2 + (1,)
            spatial_start_dim = 2  # [B, T, 1, E]
        else:
            strides = (1, 1) + self.strides * 2
            spatial_start_dim = 3  # [B, E, T, 1]
        inputs = tf.expand_dims(inputs, spatial_start_dim)
        depthwise_kernel = tf.expand_dims(self.depthwise_kernel, axis=1)  # (kernel_size, 1) across T dimension
        dilation_rate = (1,) + self.dilation_rate

        outputs = tf.nn.depthwise_conv2d(
            inputs,
            depthwise_kernel,
            strides=strides,
            padding=self.padding.upper() if not self._is_causal else "VALID",
            dilations=dilation_rate,
            data_format=conv_utils.convert_data_format(self.data_format, ndim=4),
        )

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        outputs = tf.squeeze(outputs, [spatial_start_dim])

        if self.activation is not None:
            return self.activation(outputs)

        return outputs
