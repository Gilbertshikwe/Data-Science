import tensorflow as tf
from tensorflow.keras import layers, models #type:ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type:ignore
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image #type:ignore

# Define the CNN model
def create_model(input_shape=(150, 150, 3), num_classes=3):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Prepare the data
batch_size = 2  # Reduced batch size due to very small dataset
img_height, img_width = 150, 150

data_dir = '/home/gilbert/MyCodes/DataScience data/fruit_dataset'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    seed=123
)

# Print dataset information
print(f"Training samples: {train_generator.samples}")
print(f"Classes: {train_generator.class_indices}")

# Create and train the model
model = create_model()

# Calculate steps per epoch
steps_per_epoch = max(1, train_generator.samples // batch_size)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('cnn.png')

# Function to use the model to classify a new image
def predict_fruit(img_path, model):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    prediction = model.predict(img_array)
    class_names = list(train_generator.class_indices.keys())
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    print(f"The image is most likely a {predicted_class} with {confidence:.2f}% confidence")

# Example usage
predict_fruit('/home/gilbert/MyCodes/DataScience data/fruit_dataset/apple/apple1.jpeg', model)
predict_fruit('/home/gilbert/MyCodes/DataScience data/fruit_dataset/banana/banana1.jpeg', model)
predict_fruit('/home/gilbert/MyCodes/DataScience data/fruit_dataset/orange/orange1.jpeg', model)

