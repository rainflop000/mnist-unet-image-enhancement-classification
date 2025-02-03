import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np

class DisplayCallback(Callback):
    def __init__(self, test_data):
        super().__init__()
        self.test_data = test_data
        self.test_losses = []

    def on_epoch_end(self, epoch, logs=None):
        test_loss = self.model.evaluate(self.test_data, self.test_data, verbose=0)
        self.test_losses.append(test_loss if isinstance(test_loss, float) else test_loss[0])

def low_resolution(image):
    height, width = image.shape
    # Resize image to lower resolution and then back to the original size
    reduced_quality = tf.image.resize(image[..., tf.newaxis], (height // 2, width // 2))
    original_size = tf.image.resize(reduced_quality, (height, width))
    return tf.squeeze(original_size).numpy()

def unet_model(output_channels):
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

    output_layer = Conv2D(output_channels, 1, activation="sigmoid", padding="same")(conv5)

    model = Model(inputs=inputs, outputs=output_layer)

    model.compile(optimizer="adam",
                 loss="binary_crossentropy",
                 metrics=["accuracy"])

    return model

def plot_loss(model_history, test_losses):
    loss = model_history.history["loss"]
    val_loss = model_history.history["val_loss"]
    epochs = range(len(loss))

    plt.figure()
    plt.plot(epochs, loss, "r+-", label="Training loss")
    plt.plot(epochs, val_loss, "b*-", label="Validation loss")
    plt.plot(epochs, test_losses, "go", label="Testing loss")
    plt.title("Training, Validation, and Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

def main():
    # Load dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Generate low-resolution versions
    # Initialize empty lists to store the processed images
    print("Processing training images...")
    low_res_train_images_list = []
    for img in train_images:
        low_res_img = low_resolution(img)
        low_res_train_images_list.append(low_res_img)
    low_res_train_images = np.array(low_res_train_images_list)

    print("Processing test images...")
    low_res_test_images_list = []
    for img in test_images:
        low_res_img = low_resolution(img)
        low_res_test_images_list.append(low_res_img)
    low_res_test_images = np.array(low_res_test_images_list)

    # Normalize data
    low_res_train_images = low_res_train_images / 255.0
    low_res_test_images = low_res_test_images / 255.0

    # Add channel dimension
    low_res_train_images = low_res_train_images[..., tf.newaxis]
    low_res_test_images = low_res_test_images[..., tf.newaxis]

    # Split data into train, validation, and test sets
    val_images = low_res_train_images[:5000]
    val_labels = low_res_train_images[:5000]

    # Create U-Net model
    OUTPUT_CLASSES = 1
    model = unet_model(output_channels=OUTPUT_CLASSES)

    # Print model summary
    model.summary()

    # Train model
    EPOCHS = 10
    BATCH_SIZE = 64

    test_loss_callback = DisplayCallback(low_res_test_images)

    model_history = model.fit(
        low_res_train_images, low_res_train_images,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(val_images, val_labels),
        verbose=1,
        callbacks=[test_loss_callback]
    )

    # Plot training, validation, and testing loss
    plot_loss(model_history, test_loss_callback.test_losses)

if __name__ == "__main__":
    main()