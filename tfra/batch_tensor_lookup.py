import tensorflow as tf

def concat_tensors(tensors):
    flat_values = []
    row_lengths = []  # This will now be a list of lists

    for tensor in tensors:
        if isinstance(tensor, tf.RaggedTensor):
            flat_values.append(tensor.flat_values)
            row_lengths.append(tensor.row_lengths())  # Append as a sublist
        else:
            flat_values.append(tf.reshape(tensor, [-1]))
            tensor_row_lengths = tf.ones(tf.shape(tensor)[0], dtype=tf.int64)
            row_lengths.append(tensor_row_lengths)
    concatenated = tf.concat(flat_values, axis=0)
    return concatenated, row_lengths

def concat_embedding(tensors, embeddings, row_lengths):
    indices = []
    for length in row_lengths:  # Compute start indices for each segment
        sum =  tf.reduce_sum(length)
        indices.append(sum)

    emb_shape = embeddings.shape[-1]
    print(f"Embeddings shape: {emb_shape} {type(emb_shape)}")

    emb = tf.split(embeddings, indices, axis=0)
    print(f"Sliced embeddings shape: {emb} ")

    for i, tensor in enumerate(tensors):
        print(f"Row shape: {row_lengths[i]}")
        if isinstance(tensor, tf.RaggedTensor):
            orignal = tf.RaggedTensor.from_row_lengths(emb[i], row_lengths[i])
            print(f"orignal: {orignal.shape}")
            # Pool the embeddings
            emb[i] = tf.reduce_mean(orignal, axis=1)
            print(f"Pooled embeddings shape: {emb[i].shape}")
    return tf.concat(emb, axis=0)

def concat_embedding_stride(tensors, embeddings, row_lengths):
    results = []
    indices = [tf.constant(0, dtype=tf.int64)]
    for length in row_lengths:  # Compute start indices for each segment
        sum =  tf.reduce_sum(length)
        print(f"Length: {length} {sum} {indices[-1]} type: {type(sum)}")
        indices.append(indices[-1] + sum)

    emb_shape = embeddings.shape[-1]
    print(f"Embeddings shape: {emb_shape} {type(emb_shape)}")

    for i, tensor in enumerate(tensors):
        # Calculate the start and end indices for the current tensor
        start_index = indices[i]
        end_index = indices[i+1]
        print(f"Start index: {start_index} {end_index} {type(start_index)} {type(end_index)}")
        begin = tf.stack([start_index, 0])
        end = tf.stack([end_index,emb_shape])
        print(f"Begin: {begin} {end}")
        # embedding 10 * 5 slice to
        emb = tf.strided_slice(
            embeddings, begin, end)
        print(f"Sliced embeddings shape: {emb.shape} {row_lengths[i]}")
        if isinstance(tensor, tf.RaggedTensor):
            orignal = tf.RaggedTensor.from_row_lengths(emb, row_lengths[i])
            print(f"orignal: {orignal.shape}")
            # Pool the embeddings
            emb = tf.reduce_mean(orignal, axis=1)
            print(f"Pooled embeddings shape: {emb.shape}")
        results.append(emb)
    r = tf.concat(results, axis=1)
    print(f"Result shape: {r.shape}")
    return r

def concat_embedding_slow(tensors, embeddings, row_lengths):
    offset = 0
    results = []
    for i, tensor in enumerate(tensors):
        if isinstance(tensor, tf.RaggedTensor):
            count = tensor.flat_values.shape[0]
            print(f"Ragged tensor count: {tf.shape(tensor.flat_values)[0]} {tensor.flat_values.shape}")
            ragged_embeddings = tf.RaggedTensor.from_row_lengths(
                embeddings[offset:offset+count], row_lengths[i])
            pooled = tf.reduce_mean(ragged_embeddings, axis=1)
            results.append(pooled)
            offset += count
        else:
            # Directly use the row_lengths for regular tensors to manage the embeddings lookup and reshaping
            count = tf.shape(tensor)[0]  # Sum the lengths to find how many embeddings are needed
            pooled_embeddings = tf.reshape(embeddings[offset:offset+count], [-1, embeddings.shape[-1]])
            results.append(pooled_embeddings)
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
            print(f"Ragged tensor pooled shape: {pooled.shape}")
        else:
            embeddings = tf.nn.embedding_lookup(embedding_matrix, tensor)
            flatten_emb = tf.reshape(embeddings, [tensor.shape[0], -1])
            results.append(flatten_emb)
            print(f"Tensor pooled shape: {embeddings.shape} {flatten_emb.shape}")
    cancatenated_results = tf.concat(results, axis=0)
    print(f"Concatenated results shape: {cancatenated_results.shape}")
    return cancatenated_results

def embeddings_equality(embedding1, embedding2):
    are_equal = tf.reduce_all(tf.equal(embedding1, embedding2))
    return are_equal.numpy()

def compare_eager_lookup():
    ragged1 = tf.RaggedTensor.from_row_lengths([1, 2, 3, 4], [2, 2])
    ragged2 = tf.RaggedTensor.from_row_lengths([1, 2], [1, 1])
    tensor1 = tf.constant([[3], [4]])
    tensor2 = tf.constant([[6], [7]])
    embedding_matrix = tf.random.uniform([10, 5], -1, 1)

    tensors = [ragged1, ragged2, tensor1, tensor2]
    result_process_tensors = concat_lookup(tensors, embedding_matrix)
    result_independent_lookup_pool = independent_lookup(tensors, embedding_matrix)

    print(f"Embeddings equality test result: {embeddings_equality(result_process_tensors, result_independent_lookup_pool)}")
    print(f"Result shape from independent lookup: {result_independent_lookup_pool.shape}")

@tf.function
def fuse(ragged1, ragged2, tensor1, tensor2, embedding_matrix):
    tensors = [ragged1, ragged2, tensor1, tensor2]
    fuse_tensor = concat_lookup(tensors, embedding_matrix)
    non_fuse = independent_lookup(tensors, embedding_matrix)
    return fuse_tensor, non_fuse
# graph lookup
def graph_fuse():
    ragged1 = tf.RaggedTensor.from_row_lengths([1, 2, 3, 4], [2, 2])
    ragged2 = tf.RaggedTensor.from_row_lengths([1, 2], [1, 1])
    tensor1 = tf.constant([[3], [4]])
    tensor2 = tf.constant([[6], [7]])
    embedding_matrix = tf.random.uniform([10, 5], -1, 1)

    fuse_tensor, non_fuse = fuse(ragged1, ragged2, tensor1, tensor2, embedding_matrix)
    embeddings_equality_result = embeddings_equality(fuse_tensor, non_fuse)

    print(f"Result: {fuse_tensor} {embeddings_equality_result}")

graph_fuse()

# for default value
#movie_genres_ragged = tf.RaggedTensor.from_tensor(tf.expand_dims(x['movie_genres'], axis=-1), lengths=None)
# row_lengths = movie_genres_ragged.row_lengths()
# mask = tf.greater(row_lengths, 0)
#
# default_values = tf.fill([tf.shape(row_lengths)[0], 1], tf.cast(21, tf.int64))
# filled_movie_genres = tf.where(tf.expand_dims(mask, 1), movie_genres_ragged.to_tensor(), default_values)
#
# movie_genres_ragged = tf.RaggedTensor.from_tensor(filled_movie_genres)