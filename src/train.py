from tensorflow.keras import losses
import matplotlib.pyplot as plt
import numpy as np
from data.dataset import split_dataset
from config.config import Config
from model.unet import build_unet

def get_datasets():
    train_dataset, val_dataset, test_dataset = split_dataset(image_dir=Config.IMAGES_PATH, shuffle=True)
    return train_dataset, val_dataset, test_dataset

def main():
    train_dataset, val_dataset, test_dataset = get_datasets()

    model = build_unet(input_shape=Config.INPUT_SHAPE)
    model.summary()

    model.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])

    model.fit(train_dataset, validation_data=val_dataset, epochs=Config.EPOCHS)
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc}")


if __name__ == "__main__":
    main()