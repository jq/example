import tensorflow as tf

# 禁用 Eager Execution
tf.compat.v1.disable_eager_execution()

@tf.function
def compute_and_print(y):
    result = y + 1  # 添加一个简单的计算
    tf.print("Y value + 1:", result)
    return result

# 创建一个图
graph = tf.Graph()
with graph.as_default():
    # 创建一个张量作为输入
    y = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)

    # 调用函数并获取结果张量
    result_tensor = compute_and_print(y)

    # 使用 Session 运行图并获取结果
    with tf.compat.v1.Session() as sess:
        # 这将执行图，包括计算和打印操作
        result = sess.run(result_tensor)
        print("Result from function:", result)
