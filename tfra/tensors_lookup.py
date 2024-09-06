import tensorflow as tf

def concat_tensors(tensors):
    flat_values = []
    row_lengths = []
    non_ragged_tensors = []

    for tensor in tensors:
        if isinstance(tensor, tf.RaggedTensor):
            flat_values.append(tensor.flat_values)
            row_lengths.append(tensor.row_lengths())
        else:
            non_ragged_tensors.append(tensor)
            row_lengths.append(0)
    print(f"flat_values {flat_values} row_lengths {row_lengths} non_ragged_tensors {non_ragged_tensors}")
    print(f"{flat_values + non_ragged_tensors}")
    concatenated = tf.concat(flat_values + non_ragged_tensors, axis=0)
    return concatenated, row_lengths

def concat_embedding(tensors, embeddings, row_lengths):
    offset = 0
    results = []
    for i, tensor in enumerate(tensors):
        if isinstance(tensor, tf.RaggedTensor):
            print(f"tensor {tensor} tensor.flat_values.shape {tensor.flat_values.shape} {tensor.flat_values.shape[0]}")

            count = tensor.flat_values.shape[0]
            ragged_embeddings = tf.RaggedTensor.from_row_lengths(
                embeddings[offset:offset+count], row_lengths[i])
            pooled = tf.reduce_mean(ragged_embeddings, axis=1)
            results.append(pooled)
            offset += count
        else:
            count = tensor.shape[0]
            results.append(embeddings[offset:offset+count])
            offset += count

    return tf.concat(results, axis=0)


def concat_lookup(tensors, embedding_matrix):
    concatenated, row_lengths = concat_tensors(tensors)
    embeddings = tf.nn.embedding_lookup(embedding_matrix, concatenated)
    return concat_embedding(tensors, embeddings, row_lengths)


def independent_lookup(tensors, embedding_matrix):
    results = []
    for tensor in tensors:
        if isinstance(tensor, tf.RaggedTensor):
            embeddings = tf.nn.embedding_lookup(embedding_matrix, tensor.flat_values)
            ragged_embeddings = tf.RaggedTensor.from_row_lengths(embeddings, tensor.row_lengths())
            pooled = tf.reduce_mean(ragged_embeddings, axis=1)
            results.append(pooled)
        else:
            embeddings = tf.nn.embedding_lookup(embedding_matrix, tensor)
            results.append(embeddings)

    return tf.concat(results, axis=0)

def embeddings_equality(embedding1, embedding2):
    are_equal = tf.reduce_all(tf.equal(embedding1, embedding2))
    return are_equal.numpy()

def compare_lookup():
    ragged1 = tf.RaggedTensor.from_row_lengths([1, 2, 3, 4], [2, 2])
    ragged2 = tf.RaggedTensor.from_row_lengths([1, 2], [1, 1])
    tensor1 = tf.constant([[3, 3], [4, 5]])
    tensor2 = tf.constant([[6, 6], [7, 8]])
    embedding_matrix = tf.random.uniform([10, 5], -1, 1)

    tensors = [ragged1, ragged2, tensor1, tensor2]
    result_process_tensors = concat_lookup(tensors, embedding_matrix)
    result_independent_lookup_pool = independent_lookup(tensors, embedding_matrix)

    print(f"Embeddings equality test result: {embeddings_equality(result_process_tensors, result_independent_lookup_pool)}")

compare_lookup()

def get_dataset(batch_size=1):
    import tensorflow_datasets as tfds
    ds = tfds.load("movielens/1m-ratings",
                   split="train",
                   data_dir="~/dataset",
                   download=True)
    def process_features(x):
        # read seq as ragged tensor
        movie_genres_ragged = tf.RaggedTensor.from_tensor(tf.expand_dims(x['movie_genres'], axis=-1), lengths=None)

        return {
            "movie_id": tf.strings.to_number(x["movie_id"], tf.int64),
            "movie_genres": movie_genres_ragged,
            "user_id": tf.strings.to_number(x["user_id"], tf.int64),
            "user_gender": tf.cast(x["user_gender"], tf.int64),
            "user_occupation_label": tf.cast(x["user_occupation_label"], tf.int64),
            "bucketized_user_age": tf.cast(x["bucketized_user_age"], tf.int64),
            "timestamp": tf.cast(x["timestamp"] - 880000000, tf.int64),
        }
    features = ds.map(process_features)
