import tensorflow as tf
from tensorflow import keras

class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = keras.layers.Dense(5, activation='relu', input_shape=(2,))
        self.dense2 = keras.layers.Dense(2, activation='relu')

    @tf.function
    def preprocess(self, inputs):
        return inputs + 10

    @tf.function
    def postprocess(self, inputs):
        return inputs * 5

    def call(self, inputs):
        x = self.preprocess(inputs)
        x = self.dense1(x)
        x = self.postprocess(x)
        return self.dense2(x)


model = MyModel()
#model.build((None, 2)) this is similar to input_shape=(2,)
test_input = tf.random.normal([1, 2])
model(test_input)

path_to_save = '/tmp/my_model'
tf.keras.models.save_model(model, path_to_save)


loaded_model = tf.keras.models.load_model(path_to_save, custom_objects={'MyModel': MyModel})

test_input = tf.constant([[1.0, 2.0]])

output = loaded_model(test_input)

print("Output of the loaded model:", output.numpy())
