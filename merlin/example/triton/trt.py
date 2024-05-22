# Build TF-TRT SavedModel
from tensorflow.python.compiler.tensorrt import trt_convert as trt

import tensorflow as tf
from train import generate_random_samples, args

# Release the occupied GPU memory by TensorFlow and Keras
from numba import cuda
cuda.select_device(0)
cuda.close()

ORIGINAL_MODEL_PATH = "model_repo/hps_tf_triton/1/model.savedmodel"
NEW_MODEL_PATH = "model_repo/hps_tf_triton/2/model.savedmodel"

# Instantiate the TF-TRT converter
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=ORIGINAL_MODEL_PATH,
    precision_mode=trt.TrtPrecisionMode.FP32
)

# Convert the model into TRT compatible segments
trt_func = converter.convert()
converter.summary()

keys, dense_features, _ = generate_random_samples(args["global_batch_size"], args["vocabulary_range_per_slot"], args["dense_dim"])
keys  = tf.convert_to_tensor(keys)
dense_features = tf.convert_to_tensor(dense_features)
def input_fn():
    yield [keys, dense_features]

converter.build(input_fn=input_fn)
converter.save(output_saved_model_dir=NEW_MODEL_PATH)