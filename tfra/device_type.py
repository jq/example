from enum import IntEnum


def device():
    from tensorflow.python.framework import device as tf_device
    dv = tf_device.DeviceSpec.from_string(None).device_type
    print(dv)
def device_type():
    import tensorflow as tf

    # 检查CUDA是否可用
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("CUDA and GPUs are available.")
        try:
            # Restrict TensorFlow to only use the first GPU
            for gpu in gpus:
                # 获取GPU的显存信息
                memory_details = tf.config.experimental.get_memory_info(gpu.name)
                print(f"{gpu.name}: Memory = {memory_details['current'] / 1e9} GB")  # 显示当前使用的显存大小
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print("CUDA is not available or no GPUs detected.")


class Hkv(IntEnum):
    LRU = 0
    LFU = 1
    EPOCHLRU = 2
    EPOCHLFU = 3
    CUSTOMIZED = 4


print(Hkv.LRU)
print(Hkv.LFU.value)
