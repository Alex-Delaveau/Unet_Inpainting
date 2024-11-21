from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import tensorflow as tf




def perceptual_loss(target, prediction, vgg=vgg):

    # Extract features from intermediate layers
    layer_names = ["block1_conv1", "block2_conv1", "block3_conv1"]
    feature_extractor = Model(inputs=vgg.input, outputs=[vgg.get_layer(name).output for name in layer_names])

    # Compute features
    target_features = feature_extractor(target)
    prediction_features = feature_extractor(prediction)

    # Loss is the L2 difference of the features
    perceptual_loss = tf.add_n([tf.reduce_mean(tf.square(tf.subtract(f1, f2))) 
                                for f1, f2 in zip(target_features, prediction_features)])
    return perceptual_loss

def combined_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))  # Pixel-wise loss
    perceptual = perceptual_loss(y_true, y_pred, vgg)          # Perceptual loss
    return mse_loss + 0.1 * perceptual

