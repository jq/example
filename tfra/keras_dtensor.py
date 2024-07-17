import tensorflow as tf
from tensorflow.experimental import dtensor

# 创建一个分布策略
devices = ["CPU:0"]
mesh = dtensor.create_mesh([("batch", 1)], devices)

# 使用 DTensor 创建模型层
inputs = tf.keras.Input(shape=(10,))
dense = dtensor.DTensorDense(32, activation='relu', mesh=mesh, mesh_axes=(None, "batch"))
outputs = dense(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 生成数据并分布到不同的设备上
x_train = dtensor.copy_to_mesh(tf.random.uniform([64, 10]), mesh, layout=(None, "batch"))
y_train = dtensor.copy_to_mesh(tf.random.uniform([64, 32]), mesh, layout=(None, "batch"))

# 使用 fit 方法训练模型
model.fit(x_train, y_train, epochs=10)
