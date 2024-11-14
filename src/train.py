from data.dataset import InpaintingDataset
from config.config import Config
import os

def get_dataset():
    print("Get data")
    print(os.listdir(os.path.curdir))
    return InpaintingDataset(image_dir=Config.IMAGES_PATH, batch_size=32)
    

def main():
    print("main")
    dataset = get_dataset()
    print(dataset.__len__())


if __name__ == "__main__":
    main()