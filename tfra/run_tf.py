import tensorflow as tf

# 获取 TensorFlow 的库文件目录
print("Library directory:", tf.sysconfig.get_lib())

# 获取 TensorFlow 的包含目录（头文件）
print("Include directory:", tf.sysconfig.get_include())

# 获取编译时的链接标志
print("Link flags:", tf.sysconfig.get_link_flags())

# 获取编译时的编译标志
print("Compile flags:", tf.sysconfig.get_compile_flags())
