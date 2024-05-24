import os
import matplotlib.pyplot as plt
import numpy as np
import keras
import PIL
import tensorflow as tf

from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers 
from tensorflow.keras.models import load_model

# Loading the model

@keras.saving.register_keras_serializable()
class SelfAttention(layers.Layer):
    def __init__(self, units, W=None, U=None, V=None, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units)
        self.U = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, inputs):
        query = self.W(inputs)
        key = self.U(inputs)
        value = inputs

        score = tf.nn.tanh(query + key)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        context_vector = attention_weights * value
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

    def get_config(self):
        base_config = super(SelfAttention, self).get_config()
        config = {
            "units": keras.saving.serialize_keras_object(self.units),
            "W": keras.saving.serialize_keras_object(self.W),
            "U": keras.saving.serialize_keras_object(self.U),
            "V": keras.saving.serialize_keras_object(self.V),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        units_conf = config.pop("units")
        W_conf = config.pop("W")
        U_conf = config.pop("U")
        V_conf = config.pop("V")

        units = keras.saving.deserialize_keras_object(units_conf)
        W = keras.saving.deserialize_keras_object(W_conf)
        U = keras.saving.deserialize_keras_object(U_conf)
        V = keras.saving.deserialize_keras_object(V_conf)
        return cls(units, W, U, V, **config)

custom_objects = {"SelfAttention": SelfAttention}
with keras.saving.custom_object_scope(custom_objects):
    model = load_model('./models/flower_detectionv4.keras')

print(model.summary())
    
class_names = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']

def load_img(img_path):
    img = Image.open(img_path).resize((200,200))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(img_path):
    img_array = load_img(img_path)
    prediction = model.predict(img_array)
    probabilities = prediction[0]
    predicted_class = np.argmax(prediction, axis=1)[0]
    probs = []
    for _, prob in enumerate(probabilities):
        # print(f'{class_names[i]}: {prob*100:.2f}%')
        probs.append(prob*100)

    return predicted_class, class_names[predicted_class], probs




