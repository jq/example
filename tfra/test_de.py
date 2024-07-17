import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.platform import test
from tensorflow_recommenders_addons import dynamic_embedding as de

from embedding_lookup import DynamicEmbeddingLookup


class EmbeddingLayerTest(test.TestCase):
    def test_embedding_lookup(self):
        embedding_size = 3
        keys_num = 200
        de_layer = de.keras.layers.Embedding(
           embedding_size=embedding_size,
           value_dtype=tf.float32,
           initializer=tf.keras.initializers.GlorotUniform(),
           name="embedding",
           init_capacity=100,
           trainable=True,
        )
        ids = math_ops.range(keys_num, dtype=dtypes.int64)
        output = de_layer(ids)
        assert output.shape == (keys_num, 3)
        print(f"size : {de_layer.params.size()}")
        prepared_values = array_ops.ones((keys_num, embedding_size), dtype=tf.float32)
        de_layer.params.upsert(ids, prepared_values)
        print(f"new size : {de_layer.params.size()}")
        output = de_layer(ids)
        self.assertAllClose(prepared_values, output, rtol=1e-6, atol=1e-7)



    # def test_de_create(self):
    #     embedding_size = 3
    #     keys_num = 200
    #     de =  DynamicEmbeddingLookup(
    #         name="embedding",
    #         init_capacity=keys_num * 2,
    #         embedding_size=embedding_size,
    #         initializer=tf.keras.initializers.GlorotUniform(),
    #         value_dtype=tf.float32,
    #         trainable=True,
    #     )
    #     ids = math_ops.range(keys_num, dtype=dtypes.int64)
    #     prepared_values = array_ops.ones((keys_num, embedding_size), dtype=tf.float32)
    #     de.upsert(ids, prepared_values)
    #     output = de(ids)
    #     self.assertAllClose(prepared_values, output, rtol=1e-6, atol=1e-7)
    #     assert output.shape == (keys_num, 3)


if __name__ == "__main__":

    test.main()


