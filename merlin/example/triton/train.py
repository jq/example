import hierarchical_parameter_server as hps
import os
import numpy as np
import tensorflow as tf
import struct

args = dict()
# model_repo/hps_tf_triton/ should be same as hps_tf_triton.json
base_dir = '/workspace/e/example/triton/model_repo/hps_tf_triton/'
args["gpu_num"] = 1                               # the number of available GPUs
args["iter_num"] = 10                             # the number of training iteration
args["slot_num"] = 5                              # the number of feature fields in this embedding layer
args["embed_vec_size"] = 16                       # the dimension of embedding vectors
args["global_batch_size"] = 1024                  # the globally batchsize for all GPUs
args["max_vocabulary_size"] = 50000
args["vocabulary_range_per_slot"] = [[0,10000],[10000,20000],[20000,30000],[30000,40000],[40000,50000]]
args["dense_dim"] = 10

args["dense_model_path"] = "hps_tf_triton_dense.model"
args["ps_config_file"] = "hps_tf_triton.json"
args["embedding_table_path"] = "hps_tf_triton_sparse_0.model"
args["saved_path"] = "hps_tf_triton_tf_saved_model"
args["np_key_type"] = np.int64
args["np_vector_type"] = np.float32
args["tf_key_type"] = tf.int64
args["tf_vector_type"] = tf.float32


os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args["gpu_num"])))

def generate_random_samples(num_samples, vocabulary_range_per_slot, dense_dim, key_dtype = args["np_key_type"]):
    keys = list()
    for vocab_range in vocabulary_range_per_slot:
        keys_per_slot = np.random.randint(low=vocab_range[0], high=vocab_range[1], size=(num_samples, 1), dtype=key_dtype)
        keys.append(keys_per_slot)
    keys = np.concatenate(np.array(keys), axis = 1)
    dense_features = np.random.random((num_samples, dense_dim)).astype(np.float32)
    labels = np.random.randint(low=0, high=2, size=(num_samples, 1))
    return keys, dense_features, labels

def tf_dataset(keys, dense_features, labels, batchsize):
    dataset = tf.data.Dataset.from_tensor_slices((keys, dense_features, labels))
    dataset = dataset.batch(batchsize, drop_remainder=True)
    return dataset

class TrainModel(tf.keras.models.Model):
    def __init__(self,
                 init_tensors,
                 slot_num,
                 embed_vec_size,
                 dense_dim,
                 **kwargs):
        super(TrainModel, self).__init__(**kwargs)

        self.slot_num = slot_num
        self.embed_vec_size = embed_vec_size
        self.dense_dim = dense_dim
        self.init_tensors = init_tensors
        self.params = tf.Variable(initial_value=tf.concat(self.init_tensors, axis=0))
        self.concat = tf.keras.layers.Concatenate(axis=1, name="concatenate")
        self.fc_1 = tf.keras.layers.Dense(units=256, activation=None,
                                          kernel_initializer="ones",
                                          bias_initializer="zeros",
                                          name='fc_1')
        self.fc_2 = tf.keras.layers.Dense(units=1, activation=None,
                                          kernel_initializer="ones",
                                          bias_initializer="zeros",
                                          name='fc_2')

    def call(self, inputs):
        keys, dense_features = inputs[0], inputs[1]
        embedding_vector = tf.nn.embedding_lookup(params=self.params, ids=keys)
        embedding_vector = tf.reshape(embedding_vector, shape=[-1, self.slot_num * self.embed_vec_size])
        concated_features = self.concat([embedding_vector, dense_features])
        logit = self.fc_2(self.fc_1(concated_features))
        return logit

    def summary(self):
        inputs = [tf.keras.Input(shape=(self.slot_num, ), dtype=args["tf_key_type"]),
                  tf.keras.Input(shape=(self.dense_dim, ), dtype=tf.float32)]
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()

def train(args):
    init_tensors = np.ones(shape=[args["max_vocabulary_size"], args["embed_vec_size"]], dtype=args["np_vector_type"])

    model = TrainModel(init_tensors, args["slot_num"], args["embed_vec_size"], args["dense_dim"])
    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def _train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logit = model(inputs)
            loss = loss_fn(labels, logit)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return logit, loss

    keys, dense_features, labels = generate_random_samples(args["global_batch_size"]  * args["iter_num"], args["vocabulary_range_per_slot"], args["dense_dim"])
    dataset = tf_dataset(keys, dense_features, labels, args["global_batch_size"])
    for i, (keys, dense_features, labels) in enumerate(dataset):
        inputs = [keys, dense_features]
        _, loss = _train_step(inputs, labels)
        print("-"*20, "Step {}, loss: {}".format(i, loss),  "-"*20)

    return model

# the local directory to save the model and config files
triton_model_repo = base_dir #"/hugectr/hps_tf/notebooks/model_repo/hps_tf_triton/"

class InferenceModel(tf.keras.models.Model):
    def __init__(self,
                 slot_num,
                 embed_vec_size,
                 dense_dim,
                 dense_model_path,
                 **kwargs):
        super(InferenceModel, self).__init__(**kwargs)

        self.slot_num = slot_num
        self.embed_vec_size = embed_vec_size
        self.dense_dim = dense_dim
        self.lookup_layer = hps.LookupLayer(model_name = "hps_tf_triton",
                                            table_id = 0,
                                            emb_vec_size = self.embed_vec_size,
                                            emb_vec_dtype = args["tf_vector_type"],
                                            ps_config_file = triton_model_repo + args["ps_config_file"],
                                            global_batch_size = args["global_batch_size"],
                                            name = "lookup")
        self.dense_model = tf.keras.models.load_model(dense_model_path)

    def call(self, inputs):
        keys, dense_features = inputs[0], inputs[1]
        embedding_vector = self.lookup_layer(keys)
        embedding_vector = tf.reshape(embedding_vector, shape=[-1, self.slot_num * self.embed_vec_size])
        logit = self.dense_model([embedding_vector, dense_features])
        return logit

    def summary(self):
        inputs = [tf.keras.Input(shape=(self.slot_num, ), dtype=args["tf_key_type"]),
                  tf.keras.Input(shape=(self.dense_dim, ), dtype=tf.float32)]
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()

def create_and_save_inference_graph(args):
    model = InferenceModel(args["slot_num"], args["embed_vec_size"], args["dense_dim"], args["dense_model_path"])
    model.summary()
    _ = model([tf.keras.Input(shape=(args["slot_num"], ), dtype=args["tf_key_type"]),
               tf.keras.Input(shape=(args["dense_dim"], ), dtype=tf.float32)])
    model.save(args["saved_path"])

def convert_to_sparse_model(embeddings_weights, embedding_table_path, embedding_vec_size):
    os.system("mkdir -p {}".format(embedding_table_path))
    with open("{}/key".format(embedding_table_path), 'wb') as key_file, \
            open("{}/emb_vector".format(embedding_table_path), 'wb') as vec_file:
        for key in range(embeddings_weights.shape[0]):
            vec = embeddings_weights[key]
            key_struct = struct.pack('q', key)
            vec_struct = struct.pack(str(embedding_vec_size) + "f", *vec)
            key_file.write(key_struct)
            vec_file.write(vec_struct)

trained_model = train(args)
weights_list = trained_model.get_weights()
embedding_weights = weights_list[-1] # the last weight is the embedding table
dense_model = tf.keras.Model(trained_model.get_layer("concatenate").input,
                             trained_model.get_layer("fc_2").output)
dense_model.summary()
dense_model.save(args["dense_model_path"])

convert_to_sparse_model(embedding_weights, args["embedding_table_path"], args["embed_vec_size"])
create_and_save_inference_graph(args)

