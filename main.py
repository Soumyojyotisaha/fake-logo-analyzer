import numpy as np
import tensorflow as tf
from keras import layers, Sequential, utils
import os
import matplotlib.pyplot as plt

train_ds = tf.keras.utils.image_dataset_from_directory(
  "Images",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(70, 70),
  batch_size=32)

val_ds = utils.image_dataset_from_directory(
  "Images",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(70, 70),
  batch_size=32)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = Sequential()
model.add(layers.Rescaling(1./255))
model.add(layers.Conv2D(70, (3, 3), activation='relu', input_shape=(70, 70, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(140, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(140, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(70, activation='relu'))
model.add(layers.Dense(2))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(train_ds, epochs=30, 
                    validation_data=val_ds,callbacks=[cp_callback])

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
plt.savefig("model_accuracy.png")

test_loss, test_acc = model.evaluate(val_ds, verbose=2)
print(test_acc)
