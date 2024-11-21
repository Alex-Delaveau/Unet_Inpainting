import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from config.config import Config
from data.dataset import split_dataset
import matplotlib.pyplot as plt

def add_black_hole(image):
    """Adds a fixed black mask at the center of the image."""
    h, w = (256, 256)  # Image dimensions
    mh, mw = (96, 96)  # Mask dimensions

    # Calculate the fixed position for the mask
    x_start = (w - mw) // 2
    y_start = (h - mh) // 2

    # Create a copy of the image
    image_with_hole = tf.identity(image)

    # Apply the black mask by setting the specified region to zero
    image_with_hole = tf.tensor_scatter_nd_update(
        image_with_hole,
        tf.reshape(
            tf.stack(
                tf.meshgrid(
                    tf.range(y_start, y_start + mh),
                    tf.range(x_start, x_start + mw),
                    indexing='ij'
                ),
                axis=-1
            ),
            [-1, 2]
        ),
        tf.zeros([mh * mw, 3])  # Updates with zero (black mask)
    )

    return image_with_hole



def add_random_noise_hole(image):
    """Adds a fixed mask at the center with random noise inside."""
    image_size = (256, 256)
    mask_size = (96, 96)

    h, w = image_size
    mh, mw = mask_size

    # Calculate the fixed position for the mask
    x_start = (w - mw) // 2
    y_start = (h - mh) // 2

    # Generate random noise for the mask area
    noise = tf.random.uniform((mh, mw, 3), 0, 1)  # Random noise in [0, 1]

    # Create the mask for the hole
    image_with_hole = tf.identity(image)
    image_with_hole = tf.tensor_scatter_nd_update(
        image_with_hole,
        indices=tf.reshape(
            tf.stack(
                tf.meshgrid(
                    tf.range(y_start, y_start + mh),
                    tf.range(x_start, x_start + mw),
                    indexing='ij'
                ),
                axis=-1
            ),
            [-1, 2]
        ),
        updates=tf.reshape(noise, [-1, 3])
    )

    return image_with_hole

def load_image(image_path, img_size=(256, 256), mask_size=(96, 96)):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    img = cv2.resize(img, img_size)
    img = img / 255.0  # Normalize to [0, 1]

    # Add fixed hole
    image_with_hole = add_black_hole(img)

    return img, image_with_hole

def load_image(path):
    """Loads and preprocesses a single image."""
    path = tf.strings.as_string(path)  # Ensure path is treated as a string
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # Decode JPEG images
    img = tf.image.resize(img, (256, 256))  # Resize to desired input size
    img = img / 255.0  # Normalize to [0, 1]
    return img


def inference_on_image(image_path):
    # Load the trained model
    model = load_model('/home/alex/enseirb/unet_face/unet.keras')

    # Load and preprocess the image
    img = load_image(image_path)
    image_with_hole = add_black_hole(img)

    # Add batch dimension
    image_with_hole_batch = np.expand_dims(image_with_hole, axis=0)


    # Perform inference
    prediction = model.predict(image_with_hole_batch)


    # Post-process the prediction if necessary
    prediction = np.squeeze(prediction, axis=0)  # Remove batch dimension
    prediction = (prediction * 255).astype(np.uint8)  # Convert to uint8

    # Save or display the base image, image with hole and prediction in the same window
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Base image")
    plt.axis('off')
    
    # get rid of batch dimmension from image_with_hole
    image_with_hole_batch = np.squeeze(image_with_hole_batch, axis=0)
    plt.subplot(1, 3, 2)
    plt.imshow(image_with_hole_batch)
    plt.title("Image with hole")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(prediction)
    plt.title("Prediction")
    plt.axis('off')

    #save the image
    plt.savefig('/home/alex/enseirb/unet_face/prediction.jpg')


if __name__ == "__main__":
    # Test the whole model
    # test_model()

    # Perform inference on a single image
    inference_on_image('/home/alex/enseirb/unet_face/dataset/ffhq_resized/splitted_images/folder_0/image_15.jpg')