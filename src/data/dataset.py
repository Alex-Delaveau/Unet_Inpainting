import os
import numpy as np
import tensorflow as tf
import cv2
from data.augment import DataTransformer

class InpaintingDataset(tf.keras.utils.Sequence):
    def __init__(self, image_paths, batch_size, img_size=(256, 256), mask_size=(96, 96), shuffle=True):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.mask_size = mask_size
        self.shuffle = shuffle
        self.data_transformer = DataTransformer()
        self.on_epoch_end()

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, images_with_holes = self.__data_generation(batch_indexes)
        augmented_images = []
        augmented_images_with_holes = []
        for image, image_with_holes in zip(images, images_with_holes):
            aug_image, aug_image_with_holes = self.data_transformer.augment(image, image_with_holes)
            augmented_images.append(aug_image)
            augmented_images_with_holes.append(aug_image_with_holes)
        
        return np.array(augmented_images_with_holes), np.array(augmented_images)  # (input, target)


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_indexes):
        images, images_with_holes = [], []
        for idx in batch_indexes:
            image_path = self.image_paths[idx]
            image = self.load_image(image_path)
            image_with_holes = self.add_fixed_hole(image)
            images.append(image)
            images_with_holes.append(image_with_holes)
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

def split_dataset(image_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, batch_size=32, img_size=(256, 256), mask_size=(96, 96), shuffle=True):
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    np.random.shuffle(image_paths)
    
    train_size = int(len(image_paths) * train_ratio)
    val_size = int(len(image_paths) * val_ratio)
    
    train_paths = image_paths[:train_size]
    val_paths = image_paths[train_size:train_size + val_size]
    test_paths = image_paths[train_size + val_size:]
    
    train_dataset = InpaintingDataset(train_paths, batch_size, img_size, mask_size, shuffle)
    val_dataset = InpaintingDataset(val_paths, batch_size, img_size, mask_size, shuffle)
    test_dataset = InpaintingDataset(test_paths, batch_size, img_size, mask_size, shuffle=False)
    
    return train_dataset, val_dataset, test_dataset