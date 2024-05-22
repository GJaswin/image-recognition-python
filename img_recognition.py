import os
import matplotlib.pyplot as plt
import numpy as np
import keras
import PIL
import tensorflow as tf

from PIL import Image
from tensorflow.keras.preprocessing import image

# Loading the model
model = tf.keras.models.load_model('flower_detection.keras')
    
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

    for i, prob in enumerate(probabilities):
        print(f'{class_names[i]}: {prob*100:.2f}%')

    return class_names[predicted_class]

img_path = 'lot.jpg'
flower = predict(img_path)
print(f'The flower is: {flower}')
