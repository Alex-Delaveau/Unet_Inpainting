import tensorflow as tf
import numpy as np
import os
from glob import glob
from config.config import Config
import kagglehub
class InpaintingDataset:
    def __init__(self, image_paths, img_size, mask_size, batch_size, shuffle=True, augment=False, use_black_mask=False):
        self.image_paths = image_paths
        self.img_size = img_size
        self.mask_size = mask_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.use_black_mask = use_black_mask  # Flag to toggle black mask usage

    def load_image(self, path):
        """Loads and preprocesses a single image."""
        path = tf.strings.as_string(path)  # Ensure path is treated as a string
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)  # Decode JPEG images
        img = tf.image.resize(img, self.img_size)  # Resize to desired input size
        img = img / 255.0  # Normalize to [0, 1]
        return img

    def add_black_hole(self, image):
        """Adds a fixed black mask at the center of the image."""
        h, w = self.img_size
        mh, mw = self.mask_size

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



    def add_random_noise_hole(self, image):
        """Adds a fixed mask at the center with random noise inside."""
        h, w = self.img_size
        mh, mw = self.mask_size

        # Calculate the fixed position for the mask
        x_start = (w - mw) // 2
        y_start = (h - mh) // 2

        # Generate random noise for the mask area
        noise = tf.random.uniform((mh, mw, 3), 0, 1)  # Random noise in [0, 1]

        # Apply the noise mask
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

    def augment_image(self, image, image_with_hole):
        """Applies the same random augmentations to the image and its masked version."""
        # Generate a random seed for consistent transformations
        seed = tf.random.uniform(shape=(), maxval=1000000, dtype=tf.int32)

        # Apply random horizontal flip with the same seed
        image = tf.image.stateless_random_flip_left_right(image, seed=[seed, 0])
        image_with_hole = tf.image.stateless_random_flip_left_right(image_with_hole, seed=[seed, 0])

        # Apply random brightness with the same seed
        seed += 1  # Increment the seed
        image = tf.image.stateless_random_brightness(image, max_delta=0.1, seed=[seed, 0])
        image_with_hole = tf.image.stateless_random_brightness(image_with_hole, max_delta=0.1, seed=[seed, 0])

        # Clip pixel values to the valid range [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        image_with_hole = tf.clip_by_value(image_with_hole, 0.0, 1.0)

        return image, image_with_hole

    def preprocess(self, path):
        """Loads an image, applies a mask (black or random noise), and optionally applies augmentation."""
        image = self.load_image(path)
        if self.use_black_mask:
            image_with_hole = self.add_black_hole(image)
        else:
            image_with_hole = self.add_random_noise_hole(image)

        if self.augment:
            image, image_with_hole = self.augment_image(image, image_with_hole)

        return image_with_hole, image

    def to_tf_dataset(self):
        """Converts the dataset into a tf.data.Dataset pipeline."""
        print(self.image_paths)
        dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.image_paths))

        dataset = dataset.map(
            lambda x: self.preprocess(x),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset


def load_kaggle_dataset(kaggle_dataset_id):
    """Download the Kaggle dataset and retrieve image file paths."""
    print(f"Downloading dataset from Kaggle: {kaggle_dataset_id}...")
    dataset_path = kagglehub.dataset_download(kaggle_dataset_id)
    print(f"Dataset downloaded to: {dataset_path}")

    # Ensure the dataset path exists
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    # Retrieve all image file paths recursively
    image_paths = glob(os.path.join(dataset_path, '**', '*.jpg'), recursive=True)
    if not image_paths:
        raise ValueError(f"No images found in the dataset path: {dataset_path}")

    print(f"Found {len(image_paths)} images in the dataset.")
    return image_paths

def split_dataset(kaggle_dataset_id, train_ratio=Config.TRAIN_RATIO, val_ratio=Config.VALID_RATIO, test_ratio=Config.TEST_RATIO, batch_size=Config.BATCH_SIZE):
    image_paths = load_kaggle_dataset(kaggle_dataset_id)
    np.random.shuffle(image_paths)
    
    train_size = int(len(image_paths) * train_ratio)
    val_size = int(len(image_paths) * val_ratio)
    
    # train_paths = image_paths[:train_size]
    # val_paths = image_paths[train_size:train_size + val_size]
    # test_paths = image_paths[train_size + val_size:]

    train_paths = image_paths[:100]
    print(train_paths)
    val_paths = image_paths[:100]
    test_paths = image_paths[:100]

    train_dataset = InpaintingDataset(
        train_paths, Config.INPUT_SIZE, Config.MASK_SIZE, batch_size, shuffle=True, augment=True, use_black_mask=True
    ).to_tf_dataset()

    val_dataset = InpaintingDataset(
        val_paths, Config.INPUT_SIZE, Config.MASK_SIZE, batch_size, shuffle=False, augment=False, use_black_mask=True
    ).to_tf_dataset()

    test_dataset = InpaintingDataset(
        test_paths, Config.INPUT_SIZE, Config.MASK_SIZE, batch_size, shuffle=False, augment=False
    ).to_tf_dataset()
    
    return train_dataset, val_dataset, test_dataset


