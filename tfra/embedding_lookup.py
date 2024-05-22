from __future__ import annotations

from typing import Any

import abc

import tensorflow as tf
from tensorflow_recommenders_addons import dynamic_embedding as de


class IEmbeddingLookup(abc.ABC):
    """Abstract class for embedding."""

    @abc.abstractmethod
    def __call__(self, ids: tf.Tensor) -> tf.Tensor:
        """Call the embedding layer."""
        raise NotImplementedError

    @abc.abstractmethod
    def safe_embedding_lookup_sparse(
            self,
            sparse_ids: tf.SparseTensor | tf.RaggedTensor,
            sparse_weights: tf.SparseTensor | tf.RaggedTensor | None,
            combiner: str,
    ) -> tf.Tensor:
        """Lookup embedding with safe handling of missing values."""
        raise NotImplementedError

    @abc.abstractmethod
    def upsert(keys: tf.Tensor, values: tf.Tensor):
        """Upsert the embedding table with the given ids and values."""
        raise NotImplementedError

    @abc.abstractmethod
    def restrict(self, num_reserved, **kwargs):
        """Restrict the embedding table to the given number of reserved keys."""
        raise NotImplementedError

    @abc.abstractmethod
    def remove(self, keys, name=None):
        """Remove the keys from the embedding table."""
        raise NotImplementedError

    @abc.abstractmethod
    def clear(self, name=None):
        """Clear the embedding table."""
        raise NotImplementedError

    @abc.abstractmethod
    def export(self, name=None):
        """Returns tensors of all keys and values in the table.

        Args:
          name: A name for the operation (optional).

        Returns:
          A pair of tensors with the first tensor containing all keys and the
            second tensors containing all values in the table.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def size(self, index=None, name=None):
        """Compute the number of elements in the index-th table of this Variable.

        If index is none, the total size of the Variable wil be return.

        Args:
          index: The device index of table (optional)
          name: A name for the operation (optional).

        Returns:
          A scalar tensor containing the number of elements in this Variable.
        """
        raise NotImplementedError


class DynamicEmbeddingLookup(IEmbeddingLookup):
    """Embedding layer with dynamic embedding."""

    def __init__(self,
                 name: str,
                 init_capacity: int,
                 embedding_size: int,
                 initializer: Any,
                 value_dtype: tf.DType,
                 trainable: bool,
                 ):
        self.embedding = de.keras.layers.Embedding(
            embedding_size=embedding_size,
            value_dtype=value_dtype,
            initializer=initializer,
            name=name,
            init_capacity=init_capacity,
            trainable=trainable,
        )

    def __call__(self, ids: tf.Tensor) -> tf.Tensor:
        """Call the embedding layer."""
        return self.embedding(ids)

    def safe_embedding_lookup_sparse(
            self,
            sparse_ids: tf.SparseTensor | tf.RaggedTensor,
            sparse_weights: tf.SparseTensor | tf.RaggedTensor | None,
            combiner: str,
    ) -> tf.Tensor:
        """Lookup embedding with safe handling of missing values."""
        embedding_weights = self.embedding
        # TFRA use key to compute partition, can't use the TF partition strategy
        return de.safe_embedding_lookup_sparse(
            embedding_weights, sparse_ids, sparse_weights, combiner=combiner)

    def upsert(self, keys: tf.Tensor, values: tf.Tensor)-> None:
        """Upsert the embedding table with the given ids and values."""
        self.embedding.params.upsert(keys, values)

    def restrict(self, num_reserved, **kwargs)-> None:
        """Restrict the embedding table to the given number of reserved keys."""
        self.embedding.params.restrict(num_reserved, **kwargs)

    def remove(self, keys, name=None)-> None:
        """Remove the keys from the embedding table."""
        self.embedding.params.remove(keys, name=name)

    def clear(self, name=None)-> None:
        """Clear the embedding table."""
        self.embedding.params.clear(name=name)

    def export(self, name=None)-> (tf.tensor, tf.tensor):
        """Export the embedding table."""
        self.embedding.params.export(name=name)

    def size(self, index=None, name=None)-> tf.Tensor:
        """Compute the number of elements in the index-th table of this Variable.
        """
        return self.embedding.params.size(index=index, name=name)