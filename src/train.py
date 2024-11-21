from tensorflow.keras import losses, callbacks
import matplotlib.pyplot as plt
# from data.dataset import split_dataset
from data.optimized_data import split_dataset
from config.config import Config
from model.unet import build_unet
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import tensorflow as tf
from model.metrics import combined_loss

# def get_datasets():
#     train_dataset, val_dataset, test_dataset = split_dataset(kaggle_dataset_id=Config.DATASET_BASE_PATH, shuffle=True)
#     return train_dataset, val_dataset, test_dataset

"""Compute perceptual loss using VGG16 features."""
vgg = VGG16(weights="imagenet", include_top=False)
vgg.trainable = False

def get_optimized_datasets():
    train_dataset, val_dataset, test_dataset = split_dataset(kaggle_dataset_id=Config.DATASET_BASE_PATH)
    return train_dataset, val_dataset, test_dataset

def visualize_first_batch(dataset, save_path=None):
    print(dataset.__len__())
    first_batch = next(iter(dataset))
    first_image_with_hole = first_batch[0][0]
    first_base_image = first_batch[1][0]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(first_base_image)
    ax[0].set_title('Base Image')
    ax[0].axis('off')
    ax[1].imshow(first_image_with_hole)
    ax[1].set_title('Image with Hole')
    ax[1].axis('off')
    


    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

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


def main():
    train_dataset, val_dataset, test_dataset = get_optimized_datasets()
    visualize_first_batch(train_dataset, save_path='first_batch.png')

    model = build_unet(input_shape=Config.INPUT_SHAPE)
    model.summary()

    

    # Define callbacks
    checkpoint = callbacks.ModelCheckpoint(
        'saved_model/best_unet_model.keras', 
        save_best_only=True,
        save_weights_only=False,  
        monitor='val_loss', 
        mode='min', 
        verbose=1
    )

    model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy', 'AUC'])

    model.fit(
        train_dataset, 
        validation_data=val_dataset, 
        epochs=Config.EPOCHS, 
        callbacks=[checkpoint]
    )


if __name__ == "__main__":
    main()