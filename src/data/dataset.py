import tensorflow as tf
import numpy as np
import cv2
import os

class InpaintingDataset(tf.keras.utils.Sequence):
    def __init__(self, image_dir, batch_size, img_size=(256, 256), mask_size=(96, 96), shuffle=True):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.batch_size = batch_size
        self.img_size = img_size
        self.mask_size = mask_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, images_with_holes = self.__data_generation(batch_indexes)
        return np.array(images_with_holes), np.array(images)  # (input, target)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_indexes):
        images, images_with_holes = [], []
        for i in batch_indexes:
            image = self.load_image(self.image_paths[i])

            # Create a copy with a fixed hole
            image_with_hole = self.add_fixed_hole(image)

            # Append to lists
            images.append(image)
            images_with_holes.append(image_with_hole)

        return images, images_with_holes

    def load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0  # Normalize to [0, 1]
        return img

    def add_fixed_hole(self, image):
        # Copy the image
        image_with_hole = image.copy()

        # Calculate center position
        h, w, _ = image_with_hole.shape
        mh, mw = self.mask_size
        x_start = (w - mw) // 2
        y_start = (h - mh) // 2
        
        # Apply the fixed mask (e.g., fill with black or another constant value)
        image_with_hole[y_start:y_start+mh, x_start:x_start+mw] = 0
        
        return image_with_hole

