import streamlit as st
from PIL import Image
import tensorflow as tf
from keras import layers, models
import numpy as np
import os

# Function to load and compile the model
def load_model():
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(70, 70, 3)),
        layers.Conv2D(70, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(140, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(140, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(70, activation='relu'),
        layers.Dense(2, activation='softmax')  # softmax activation for multi-class classification
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # from_logits should be False
                  metrics=['accuracy'])

    checkpoint_path = "training_1/cp.ckpt"
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
    else:
        # Load training data and train the model
        train_ds = tf.keras.utils.image_dataset_from_directory(
            "Images",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(70, 70),
            batch_size=32
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            "Images",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(70, 70),
            batch_size=32
        )

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            verbose=1
        )

        model.fit(train_ds, epochs=30, validation_data=val_ds, callbacks=[cp_callback])

    return model

# Function to preprocess the uploaded image
def preprocess_image(image):
    try:
        # Convert to RGB if not already in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize the image to 70x70 pixels
        image = image.resize((70, 70))
        
        return image
    except Exception as e:
        raise ValueError("Error preprocessing image:", e)

# Function to make prediction
def predict_image(image):
    model = load_model()
    image_array = np.asarray(image)
    if image_array.shape != (70, 70, 3):
        raise ValueError("Invalid image format or size. Please make sure the image is RGB and resize it to 70x70 pixels.")
    image_array = image_array / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    results = model.predict(image_array)[0]

    if results[0] > results[1]:
        return "Fake"
    else:
        return "Real"

# Streamlit UI
def main():
    st.title("Fake Logo Detection")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Open the uploaded image
            image = Image.open(uploaded_file)
            
            # Preprocess the image
            preprocessed_image = preprocess_image(image)
            
            # Display the preprocessed image
            st.image(preprocessed_image, caption='Uploaded Image (Resized to 70x70)', use_column_width=True)

            if st.button('Predict'):
                prediction = predict_image(preprocessed_image)
                st.write(f"Prediction: {prediction}")
        
        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error("Error processing image:", e)

if __name__ == "__main__":
    main()
