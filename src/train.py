from tensorflow.keras import losses, callbacks
import matplotlib.pyplot as plt
from data.dataset import split_dataset
from config.config import Config
from model.unet import build_unet

def get_datasets():
    train_dataset, val_dataset, test_dataset = split_dataset(image_dir=Config.IMAGES_PATH, shuffle=True)
    return train_dataset, val_dataset, test_dataset

def visualize_first_batch(dataset, save_path='first_batch.png'):
    # Get the first batch
    print(dataset.__len__())
    images_with_holes, images = dataset[0]

    

    # Plot the first two images with and without the mask
    fig, axes = plt.subplots(5, 2, figsize=(10, 10))
    for i in range(5):
        axes[i, 0].imshow(images_with_holes[i])
        axes[i, 0].set_title(f"Image {i+1} with Mask")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(images[i])
        axes[i, 1].set_title(f"Image {i+1} without Mask")
        axes[i, 1].axis('off')

    plt.savefig(save_path)
    plt.close()

def main():
    train_dataset, val_dataset, test_dataset = get_datasets()
    # visualize_first_batch(train_dataset, save_path='first_batch.png')

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

    model.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy', 'AUC'])

    model.fit(
        train_dataset, 
        validation_data=val_dataset, 
        epochs=Config.EPOCHS, 
        callbacks=[checkpoint]
    )


if __name__ == "__main__":
    main()