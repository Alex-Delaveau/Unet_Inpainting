import numpy as np
import cv2
from tensorflow.keras.models import load_model
from config.config import Config
from data.dataset import split_dataset

def load_image(image_path, img_size=(256, 256), mask_size=(96, 96)):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    img = cv2.resize(img, img_size)
    img = img / 255.0  # Normalize to [0, 1]

    # Add fixed hole
    image_with_hole = img.copy()
    h, w, _ = image_with_hole.shape
    mh, mw = mask_size
    x_start = (w - mw) // 2
    y_start = (h - mh) // 2
    image_with_hole[y_start:y_start+mh, x_start:x_start+mw] = 0

    return img, image_with_hole

def test_model():
    # Load the trained model
    model = load_model('/home/alex/enseirb/Unet_Inpainting/saved_model/best_unet_model.keras')

    # Load test dataset
    _, _, test_dataset = split_dataset(image_dir=Config.IMAGES_PATH, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=12, shuffle=True)

    # Evaluate the model on the test dataset
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc}")

def inference_on_image(image_path):
    # Load the trained model
    model = load_model('/home/alex/enseirb/Unet_Inpainting/saved_model/best_unet_model.keras')

    # Load and preprocess the image
    img, image_with_hole = load_image(image_path)
    image_with_hole = np.expand_dims(image_with_hole, axis=0)  # Add batch dimension

    # Perform inference
    prediction = model.predict(image_with_hole)

    # Post-process the prediction if necessary
    prediction = np.squeeze(prediction, axis=0)  # Remove batch dimension

    # Save or display the prediction
    cv2.imwrite('/home/alex/enseirb/Unet_Inpainting/inference_result.png', prediction * 255)

if __name__ == "__main__":
    # Test the whole model
    test_model()

    # Perform inference on a single image
    inference_on_image('/home/alex/enseirb/Unet_Inpainting/dataset/169_b57b3732.jpg')