
import tensorflow as tf
from tensorflow.keras.layers import  LayerNormalization


class DynamicLayerNormalization(LayerNormalization):

    def call(self, inputs):
        # TODO(b/229545225): Remove the RaggedTensor check.
        is_ragged = isinstance(inputs, tf.RaggedTensor)
        if is_ragged:
            inputs_lengths = inputs.nested_row_lengths()
            inputs = inputs.to_tensor()
        inputs = tf.cast(inputs, self.compute_dtype)
        # Compute the axes along which to reduce the mean / variance
        input_shape = tf.shape(inputs)
        ndims = input_shape.shape[0]  # Get the number of dimensions dynamically

        # Broadcasting only necessary for norm when the axis is not just
        # the last dimension
        broadcast_shape = [1] * ndims
        for dim in self.axis:
            broadcast_shape[dim] = input_shape[dim]

        def _broadcast(v):
            if (
                    v is not None
                    and len(v.shape) != ndims
                    and self.axis != [ndims - 1]
            ):
                return tf.reshape(v, broadcast_shape)
            return v

        if not self._fused:
            input_dtype = inputs.dtype
            if (
                    input_dtype in ("float16", "bfloat16")
                    and self.dtype == "float32"
            ):
                # If mixed precision is used, cast inputs to float32 so that
                # this is at least as numerically stable as the fused version.
                inputs = tf.cast(inputs, "float32")

            # Calculate the moments on the last axis (layer activations).
            mean, variance = tf.nn.moments(inputs, self.axis, keepdims=True)

            scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

            # Compute layer normalization using the batch_normalization
            # function.
            outputs = tf.nn.batch_normalization(
                inputs,
                mean,
                variance,
                offset=offset,
                scale=scale,
                variance_epsilon=self.epsilon,
            )
            outputs = tf.cast(outputs, input_dtype)
        else:
            # Collapse dims before self.axis, and dims in self.axis

            axis = sorted(self.axis)
            tensor_shape = tf.shape(inputs)
            pre_dim = tf.reduce_prod(tensor_shape[: axis[0]])
            in_dim = tf.reduce_prod(tensor_shape[axis[0] :])
            squeezed_shape = [1, pre_dim, in_dim, 1]
            # This fused operation requires reshaped inputs to be NCHW.
            data_format = "NCHW"

            inputs = tf.reshape(inputs, squeezed_shape)

            # self.gamma and self.beta have the wrong shape for
            # fused_batch_norm, so we cannot pass them as the scale and offset
            # parameters. Therefore, we create two constant tensors in correct
            # shapes for fused_batch_norm and later construct a separate
            # calculation on the scale and offset.
            scale = tf.ones([pre_dim], dtype=self.dtype)
            offset = tf.zeros([pre_dim], dtype=self.dtype)

            # Compute layer normalization using the fused_batch_norm function.
            outputs, _, _ = tf.compat.v1.nn.fused_batch_norm(
                inputs,
                scale=scale,
                offset=offset,
                epsilon=self.epsilon,
                data_format=data_format,
            )

            outputs = tf.reshape(outputs, tensor_shape)

            scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

            if scale is not None:
                outputs = outputs * tf.cast(scale, outputs.dtype)
            if offset is not None:
                outputs = outputs + tf.cast(offset, outputs.dtype)

        # If some components of the shape got lost due to adjustments, fix that.
        outputs = tf.reshape(outputs, input_shape)

        if is_ragged:
            outputs = tf.RaggedTensor.from_tensor(outputs, inputs_lengths)
        return outputs

import numpy as np

# Assuming the DynamicLayerNormalization class is already defined as you provided.

class DynamicLayerNormalizationTest(tf.test.TestCase):

    def test_dynamic_shape_support(self):
        # Create a dynamic input placeholder where the shape is not fully specified
        input_data = tf.keras.Input(shape=(None, 10), dtype=tf.float32)

        layer = DynamicLayerNormalization()

        # Apply the layer to the input
        output = layer(input_data)

        # Initialize model
        model = tf.keras.models.Model(inputs=input_data, outputs=output)

        # Create some test data
        np.random.seed(0)
        test_data = np.random.randn(2, 5, 10).astype(np.float64)  # Batch size = 2, dynamic dimension = 5

        # Check if the layer can be executed and outputs have correct shape
        output_data = model.predict(test_data)
        self.assertAllEqual(output_data.shape, (2, 5, 10))

        # To further check the correctness of computations, provide a simple test case
        # Here, it just validates dimensions, but you should add checks for actual output values.
        expected_mean = np.mean(test_data, axis=-1, keepdims=True)
        expected_std = np.std(test_data, axis=-1, keepdims=True)
        expected_normalized = (test_data - expected_mean) / (expected_std + layer.epsilon)

        # Calculate expected output considering gamma and beta are default (i.e., gamma=1, beta=0)
        self.assertAllClose(output_data, expected_normalized, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    tf.test.main()

