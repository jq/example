import tensorflow as tf

def process_tensors(ragged1, ragged2, tensor1, tensor2, embedding_matrix):
    flat_ragged1 = ragged1.flat_values
    flat_ragged2 = ragged2.flat_values

    concatenated = tf.concat([flat_ragged1, flat_ragged2, tensor1, tensor2], axis=0)
    embeddings = tf.nn.embedding_lookup(embedding_matrix, concatenated)

    ragged1_embeddings = tf.RaggedTensor.from_row_lengths(embeddings[:ragged1.flat_values.shape[0]], ragged1.row_lengths())
    ragged2_embeddings = tf.RaggedTensor.from_row_lengths(embeddings[ragged1.flat_values.shape[0]:ragged1.flat_values.shape[0]+ragged2.flat_values.shape[0]], ragged2.row_lengths())

    pooled1 = tf.reduce_mean(ragged1_embeddings, axis=1)
    pooled2 = tf.reduce_mean(ragged2_embeddings, axis=1)

    tensor1_embeddings = embeddings[-(tensor1.shape[0] + tensor2.shape[0]):-tensor2.shape[0]]
    tensor2_embeddings = embeddings[-tensor2.shape[0]:]
    concatenated_final = tf.concat([pooled1, pooled2, tensor1_embeddings, tensor2_embeddings], axis=0)

    return concatenated_final

def independent_lookup_pool(ragged1, ragged2, tensor1, tensor2, embedding_matrix):
    embeddings_ragged1 = tf.nn.embedding_lookup(embedding_matrix, ragged1.flat_values)
    embeddings_ragged2 = tf.nn.embedding_lookup(embedding_matrix, ragged2.flat_values)
    embeddings_tensor1 = tf.nn.embedding_lookup(embedding_matrix, tensor1)
    embeddings_tensor2 = tf.nn.embedding_lookup(embedding_matrix, tensor2)

    pooled_ragged1 = tf.reduce_mean(tf.RaggedTensor.from_row_lengths(embeddings_ragged1, ragged1.row_lengths()), axis=1)
    pooled_ragged2 = tf.reduce_mean(tf.RaggedTensor.from_row_lengths(embeddings_ragged2, ragged2.row_lengths()), axis=1)

    concatenated_embeddings = tf.concat([pooled_ragged1, pooled_ragged2, embeddings_tensor1, embeddings_tensor2], axis=0)

    return concatenated_embeddings

ragged1 = tf.RaggedTensor.from_row_lengths([1, 2, 3, 4], [2, 2])
ragged2 = tf.RaggedTensor.from_row_lengths([1, 2], [1, 1])
tensor1 = tf.constant([3, 4, 5])
tensor2 = tf.constant([6, 7, 8])
embedding_matrix = tf.random.uniform([10, 5], -1, 1)

result_process_tensors = process_tensors(ragged1, ragged2, tensor1, tensor2, embedding_matrix)
result_independent_lookup_pool = independent_lookup_pool(ragged1, ragged2, tensor1, tensor2, embedding_matrix)

print("Result from process_tensors:")
print(result_process_tensors)
print("Result from independent_lookup_pool:")
print(result_independent_lookup_pool)

def embeddings_equality(embedding1, embedding2):
    are_equal = tf.reduce_all(tf.equal(embedding1, embedding2))
    return are_equal

print(f"Embeddings equality test result: {embeddings_equality(result_process_tensors, result_independent_lookup_pool)}")

