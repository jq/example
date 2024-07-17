import pkg_resources
from tensorflow_recommenders_addons import dynamic_embedding as de

import tensorflow as tf

if __name__ == "__main__":
    tf.load_library("/usr/local/Caskroom/miniconda/base/envs/r/lib/python3.9/site-packages/tensorflow_recommenders_addons/dynamic_embedding/core/_math_ops.so")
    print(pkg_resources.get_distribution("tensorflow"))
    print(pkg_resources.get_distribution("tfra-nightly"))
    pkg_resources.get_distribution("tfra-nightly")

