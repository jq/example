import jax.numpy as jnp
from jax.lax import stop_gradient
from jax.numpy.linalg import eigh

# 创建数据
x = jnp.array([1, 2, 3, 2, 3, 2, 1])

# 生成邻接矩阵
A = x[:, None] == x[None, :]
A = stop_gradient(A)  # 防止JAX尝试对布尔矩阵求导

# 计算度矩阵D
D = jnp.diag(jnp.sum(A, axis=1))

# 计算拉普拉斯矩阵L
L = D - A

# 计算特征值和特征向量
evals, evecs = eigh(L)

# 分析特征值确定去重元素
num_unique = jnp.sum(jnp.abs(evals) < 0.5)
unique_values = jnp.unique(x)

print(f"特征值: {evals}")
print(f"特征向量: \n{evecs}")
print(f"唯一值数量: {num_unique}")
print(f"去重后的值: {unique_values}")
