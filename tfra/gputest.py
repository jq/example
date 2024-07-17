# -*- coding: utf-8 -*-
import json
import os
import psutil
import gc
import time

import tensorflow as tf
from tensorflow.keras.layers import (Layer, Input, Embedding, Reshape,
                                     Concatenate, Dense, Lambda)
import tensorflow_datasets as tfds
import tensorflow_recommenders_addons as tfra
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.python.platform import tf_logging as logging

import horovod.tensorflow.keras as hvd

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.random.set_seed(12345)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_DETERMINISTIC_OPS"] = "1"


value_dtype_impl = {
    "bfloat16": tf.bfloat16,
    "float16": tf.float16,
    "float32": tf.float16,
}

try:
    from keras.src import mixed_precision
except:
    from keras import mixed_precision
policy = mixed_precision.policy.Policy('mixed_bfloat16')
# mixed_precision.policy.set_global_policy(policy)
print("compute_dtype:", mixed_precision.policy.global_policy().compute_dtype)

hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


def input_fn():
    # 公开数据集，其内部继承tf.datset
    # todo：tfrecord
    ids = tfds.load("movielens/100k-ratings",
                    split="train",
                    data_dir=".",
                    download=False)
    ids = ids.map(
        lambda x: {
            "movie_id": tf.strings.to_number(x["movie_id"], tf.int64),
            "user_id": tf.strings.to_number(x["user_id"], tf.int64),
        })
    ratings = tfds.load("movielens/100k-ratings",
                        split="train",
                        data_dir=".",
                        download=False)
    ratings = ratings.map(lambda x: {"user_rating": x["user_rating"]})
    dataset = tf.data.Dataset.zip((ids, ratings))
    # dataset = dataset.shuffle(1_000_000,
    #                           seed=2021,
    #                           reshuffle_each_iteration=False)
    dataset = dataset.take(1_000_000*4096).cache().repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE).batch(4096)
    return dataset


class DeepLayer(Layer):

    def __init__(self, hidden_dim, layer_num, out_dim):
        self.layers = []
        for i in range(layer_num):
            self.layers.append(Dense(hidden_dim, "relu"))
        self.layers.append(Dense(out_dim, "sigmoid"))
        super(DeepLayer, self).__init__()

    def call(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer(output)
        return output  # (batch, out_dim)


export_dir = "hvd_test_export/"
saved_options = tf.saved_model.SaveOptions(
    namespace_whitelist=['TFRA'],
    experimental_variable_policy=tf.saved_model.experimental.VariablePolicy.
    EXPAND_DISTRIBUTED_VARIABLES)

device = "/job:localhost/replica:0/task:0/GPU:0"
# device = ["/job:localhost/replica:0/task:0/GPU:0", "/job:localhost/replica:0/task:0/GPU:1"]
# device = "/job:localhost/replica:0/task:0/CPU:0"

class EmbLayer(Layer):
    def __init__(self, input_name, is_training, hkv):
        super(EmbLayer, self).__init__()
        saver=None
        saver=tfra.dynamic_embedding.FileSystemSaver(proc_size=hvd.size(), proc_rank=hvd.rank())
        if hkv:
            kv_creator=tfra.dynamic_embedding.HkvHashTableCreator(saver=saver)
        else:
            kv_creator=tfra.dynamic_embedding.CuckooHashTableCreator(saver=saver)
        self.emb = tfra.dynamic_embedding.keras.layers.HvdAllToAllEmbedding(
            embedding_size=8,
            combiner='mean',
            key_dtype=tf.int64,
            value_dtype=value_dtype_impl[mixed_precision.global_policy().compute_dtype],
            initializer=tf.keras.initializers.RandomNormal() if is_training else 0,
            devices=device,
            name=input_name + '_DELayer',
            # kv_creator=None,
            kv_creator=kv_creator
        )

    def call(self, input_tensor):
        return self.emb(input_tensor)

def build_model(is_training=True, hkv=True):

    # 输入层
    input_list = ["movie_id", "user_id"]
    # input_list = ["movie_id"]

    inputs = dict()
    embedding_outs = []

    for input_name in input_list:
        input_tensor = Input(shape=(1,), dtype=tf.int64, name=input_name)
        inputs[input_name] = input_tensor

        #      自定义keras Embedding层（通过继承tfra.dynamic_embedding.layers.Embedding修改）
        embedding_out = EmbLayer(input_name, is_training, hkv)(input_tensor)
        # embedding_out = Embedding(
        # 1000,
        # 16,
        # embeddings_initializer=tf.keras.initializers.Ones(),
        # name=input_name+'_DELayer'
        # )(input_tensor)
        ####################################################################################
        embedding_out = tf.cast(embedding_out, tf.float32)
        embedding_out = Reshape((-1,))(embedding_out)
        embedding_outs.append(embedding_out)

    embeddings_concat = Concatenate(axis=1)(embedding_outs)

    outs = DeepLayer(8, 1, 1)(embeddings_concat)
    # outs = DeepLayer(4096*2, 5, 1)(embeddings_concat)
    outs = Lambda(lambda x: x, name="user_rating")(outs)

    model = tf.keras.Model(inputs=inputs, outputs=outs)
    # model.summary()

    # optimizer = tf.keras.optimizers.Adam(learning_rate=1E-4, amsgrad=False, jit_compile=True)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1E-4, amsgrad=False)

    # optimizer = hvd.DistributedOptimizer(optimizer)

    optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(optimizer,
                                                                 synchronous=True)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=tf.keras.metrics.AUC(num_thresholds=1000,
                                     summation_method='minoring'),
        # jit_compile=True,
        steps_per_execution = 10
    )
    return model

from tensorflow.python.eager import context
from tensorflow.python.framework import config
context.enable_jit_compile_rewrite()
config.set_soft_device_placement(True)

model = build_model(True)
data = input_fn()

callbacks = []

# Horovod: broadcast initial variable states from rank 0 to all other processes.
# This is necessary to ensure consistent initialization of all workers when
# training is started with random weights or restored from a checkpoint.
hvd_callback = tfra.dynamic_embedding.keras.callbacks.DEHvdBroadcastGlobalVariablesCallback(0)
callbacks.append(hvd_callback)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=f'{export_dir}/TFtensorboard',
    profile_batch=(0,10),
)
# callbacks.append(tensorboard_callback)

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
callbacks.append(tfra.dynamic_embedding.keras.callbacks.DEHvdModelCheckpoint('./checkpoint-{epoch}'))

# if hvd.rank() == 0:
#   tf.profiler.experimental.start(f'{export_dir}/TFtensorboard', options=None)
model.fit(
    data,
    batch_size=4096,
    epochs=1,
    # steps_per_epoch=10 // hvd.size(),
    steps_per_epoch=10,
    callbacks=callbacks,
    verbose=1 if hvd.rank() == 0 else 0)
# if hvd.rank() == 0:
#   tf.profiler.experimental.stop()
tfra.dynamic_embedding.keras.models.de_save_model(model, export_dir)

tfra.dynamic_embedding.enable_inference_mode()
export_model = build_model(is_training=False)
# Modify the inference graph to a stand-alone version
from tensorflow.python.saved_model import save as tf_save
# The save_and_return_nodes function is used to overwrite the saved_model.pb file generated by the save_model function and rewrite the inference graph.
tf_save.save_and_return_nodes(obj=export_model,
                              export_dir=export_dir,
                              options=saved_options,
                              experimental_skip_checkpoint=True)


# time.sleep(10)
# print("\n==================Start to Reload and Train==================\n")

tf.keras.backend.clear_session()
model = build_model(hkv=False)
#Unsupport# model = tf.keras.models.load_model(export_dir, compile=False)
model.load_weights(export_dir)
data = input_fn()
model.fit(
    data,
    epochs=1,
    steps_per_epoch=1,
)
tfra.dynamic_embedding.keras.models.de_save_model(model, export_dir)