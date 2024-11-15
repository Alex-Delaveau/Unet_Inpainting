import tensorflow as tf

class Random90DegreeRotation(tf.keras.layers.Layer):
    def __init__(self):
        super(Random90DegreeRotation, self).__init__()

    def call(self, x):
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        return tf.image.rot90(x, k)

class DataTransformer:
    def __init__(self):
        self.augmenter = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            Random90DegreeRotation(),
            tf.keras.layers.RandomZoom(0.2),
        ])

    def augment(self, image, image_with_holes):
        # Concatenate images along the channel dimension
        combined = tf.concat([image, image_with_holes], axis=-1)
        
        # Apply augmentations
        augmented_combined = self.augmenter(combined)
        
        # Split the augmented images back into separate images
        augmented_image = augmented_combined[..., :3]
        augmented_image_with_holes = augmented_combined[..., 3:]
        
        return augmented_image.numpy(), augmented_image_with_holes.numpy()