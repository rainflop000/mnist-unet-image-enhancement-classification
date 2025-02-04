import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import random

# Data Preprocessing Functions
def low_resolution(image):
    height, width = image.shape
    reduced_quality = tf.image.resize(image[..., tf.newaxis], (height // 2, width // 2))
    original_size = tf.image.resize(reduced_quality, (height, width))
    return tf.squeeze(original_size).numpy()

# U-Net Model Definition
def unet_model(output_channels, num_classes):
    inputs = Input(shape=[28, 28, 1])

    # Encoder
    conv1 = Conv2D(32, 3, activation="relu", padding="same")(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation="relu", padding="same")(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(128, 3, activation="relu", padding="same")(pool2)

    # Decoder
    up1 = Conv2DTranspose(64, 2, strides=(2, 2), activation="relu", padding="same")(conv3)
    concat1 = concatenate([up1, conv2], axis=-1)
    conv4 = Conv2D(64, 3, activation="relu", padding="same")(concat1)

    up2 = Conv2DTranspose(32, 2, strides=(2, 2), activation="relu", padding="same")(conv4)
    concat2 = concatenate([up2, conv1], axis=-1)
    conv5 = Conv2D(32, 3, activation="relu", padding="same")(concat2)

    output_image = Conv2D(output_channels, 1, activation="sigmoid", padding="same", name="output_image")(conv5)

    # Fully connected layers for classification
    x = Flatten()(conv3)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)
    logits = Dense(num_classes, activation="softmax", name="output_class")(x)

    model = Model(inputs=inputs, outputs=[output_image, logits])

    model.compile(optimizer="adam",
                  loss={"output_image": "binary_crossentropy", "output_class": "categorical_crossentropy"},
                  metrics={"output_image": "accuracy", "output_class": "accuracy"})

    return model

# Training Loss Plotting
def plot_loss(model_history, test_losses):
    loss = model_history.history["loss"]
    val_loss = model_history.history["val_loss"]
    test_loss = [t[0] for t in test_losses]
    epochs = range(len(loss))

    plt.figure()
    plt.plot(epochs, loss, "r+-", label="Training loss")
    plt.plot(epochs, val_loss, "b*-", label="Validation loss")
    plt.plot(epochs, test_loss, "go-", label="Testing loss")
    plt.title("Training, Validation, and Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

# Display Results
def display_results(reconstructed_image, class_scores):
    plt.imshow(reconstructed_image.squeeze(), cmap="gray")
    plt.title("Reconstructed Image")
    plt.axis("off")
    plt.show()

    print("Class-wise scores (softmax probabilities):")
    for i, score in enumerate(class_scores):
        print(f"Number {i}: {score * 100:.2f}%")

# Callback for Testing Loss Tracking
class DisplayCallback(Callback):
    def __init__(self, test_data, test_labels):
        super().__init__()
        self.test_data = test_data
        self.test_labels = test_labels
        self.test_losses = []

    def on_epoch_end(self, epoch, logs=None):
        test_loss = self.model.evaluate(self.test_data, {"output_image": self.test_data, "output_class": self.test_labels}, verbose=0)
        self.test_losses.append(test_loss)

def main():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    num_classes = 10
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

    low_res_train_images = np.array([low_resolution(img) for img in train_images]) / 255.0
    low_res_test_images = np.array([low_resolution(img) for img in test_images]) / 255.0

    low_res_train_images = low_res_train_images[..., tf.newaxis]
    low_res_test_images = low_res_test_images[..., tf.newaxis]

    val_images = low_res_train_images[:5000]
    val_labels = train_labels[:5000]

    OUTPUT_CLASSES = 1
    model = unet_model(output_channels=OUTPUT_CLASSES, num_classes=num_classes)

    model.summary()

    EPOCHS = 10
    BATCH_SIZE = 64
    test_loss_callback = DisplayCallback(low_res_test_images, test_labels)

    model_history = model.fit(
        low_res_train_images, {"output_image": low_res_train_images, "output_class": train_labels},
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(val_images, {"output_image": val_images, "output_class": val_labels}),
        verbose=2,
        callbacks=[test_loss_callback]
    )

    plot_loss(model_history, test_loss_callback.test_losses)

    random_index = random.randint(0, low_res_test_images.shape[0] - 1)
    single_image = low_res_test_images[random_index:random_index + 1]
    reconstructed_image, class_scores = model.predict(single_image)
    display_results(reconstructed_image[0], class_scores[0])

if __name__ == "__main__":
    main()