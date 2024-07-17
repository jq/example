import tensorflow as tf

# 获取 TensorFlow 的库文件目录
print("Library directory:", tf.sysconfig.get_lib())

# 获取 TensorFlow 的包含目录（头文件）
print("Include directory:", tf.sysconfig.get_include())

# 获取编译时的链接标志
print("Link flags:", tf.sysconfig.get_link_flags())

# 获取编译时的编译标志
print("Compile flags:", tf.sysconfig.get_compile_flags())


# 获取 tf.float32 类型的大小（以字节为单位）
size_in_bytes = tf.dtypes.float32.size

print(f"Size of tf.float32 in bytes: {size_in_bytes}")
