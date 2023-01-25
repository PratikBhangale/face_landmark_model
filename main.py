import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define the directories
train_dir = 'D:\\Datasets\\Final Project Datasets\\3\\fer2013\\train'
test_dir = 'D:\\Datasets\\Final Project Datasets\\3\\fer2013\\validation'

# Data Augmenting
train_datagen = ImageDataGenerator(rescale=1 / 255,
                                   vertical_flip=False,
                                   horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1 / 255,
                                  vertical_flip=False,
                                  horizontal_flip=False)

# Loading and cropping Images
training_generator = train_datagen.flow_from_directory(
    train_dir,
    class_mode='categorical',
    target_size=(250, 250),
    batch_size=70,
    color_mode='grayscale'
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    class_mode='categorical',
    target_size=(250, 250),
    batch_size=10,
    color_mode='grayscale'
)

# Model Definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(58, (3, 3), activation='relu', input_shape=(250, 250, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (2, 2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

# Compiling the Model
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print(model.summary())

# Fitting the data and starting the training
history = model.fit(training_generator, epochs=15, validation_data=validation_generator, verbose=1)

# Saving Model as Keras model
model.save('Face_Landmark_1.h5')

# Plotting the Results...
acc = history.history['accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.title('Training accuracy')
plt.show()

plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Validation accuracy')
plt.show()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training loss')
plt.show()

plt.plot(epochs, val_loss, 'b', label='Training Loss')
plt.title('Validation loss')
plt.show()
