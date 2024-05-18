import os
import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras import layers 
from tensorflow.keras import Sequential 
from tensorflow.keras.utils import to_categorical

# Checking TensorFlow version
print(tf.__version__)

# Directory containing the flower dataset
data_dir = 'flower_images'

# List of flower classes (subdirectories)
classes = os.listdir(data_dir)
num_classes = len(classes)

# Initialize lists to store images and labels
X = []
y = []

# Load images and labels
for i, cls in enumerate(classes):
    class_dir = os.path.join(data_dir, cls)
    images = os.listdir(class_dir)
    for img_name in images:
        img_path = os.path.join(class_dir, img_name)
        img = load_img(img_path, target_size=(200, 200))  # Load high-resolution images
        img_array = img_to_array(img)
        X.append(img_array)
        y.append(i)  # Assign label to image

# Convert lists to arrays
X = np.array(X)
y = np.array(y)

# One-hot encode labels
y = to_categorical(y, num_classes=num_classes)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


num_samples_per_class = 5
plt.figure(figsize=(15, 10))

for i, cls in enumerate(classes):
    class_indices = np.where(np.argmax(y_train, axis=1) == i)[0]
    sampled_indices = np.random.choice(class_indices, num_samples_per_class, replace=False)
    for j, idx in enumerate(sampled_indices):
        plt.subplot(len(classes), num_samples_per_class, i * num_samples_per_class + j + 1)
        plt.imshow(X_train[idx].astype(np.uint8))
        plt.title(cls)
        plt.axis('off')

plt.show()

class_distribution_train = {cls: np.sum(np.argmax(y_train, axis=1) == i) for i, cls in enumerate(classes)}
print("Training Set Class Distribution:", class_distribution_train)

plt.bar(class_distribution_train.keys(), class_distribution_train.values())
plt.xlabel('Classes')
plt.ylabel('Number of Samples')
plt.title('Training Set Class Distribution')
plt.xticks(rotation=45)
plt.show()

class_distribution_test = {cls: np.sum(np.argmax(y_test, axis=1) == i) for i, cls in enumerate(classes)}
print("Testing Set Class Distribution:", class_distribution_test)

plt.bar(class_distribution_test.keys(), class_distribution_test.values())
plt.xlabel('Classes')
plt.ylabel('Number of Samples')
plt.title('Testing Set Class Distribution')
plt.xticks(rotation=45)
plt.show()

# Preprocessing
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

X_train_flattened = X_train.reshape(X_train.shape[0], -1)
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train_flattened)

X_test_flattened = X_test.reshape(X_test.shape[0], -1)
X_test_pca = pca.transform(X_test_flattened)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA Explained Variance Ratio')
plt.show()

# Creating the Model

model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Training and assessing the model
history = model.fit(X_train, y_train, epochs=22, 
                    validation_data=(X_test, y_test), 
                    batch_size=16)

model.save('flower_detection2.keras')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')

keras.backend.clear_session()
