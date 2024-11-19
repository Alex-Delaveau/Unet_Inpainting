import os

from cycler import V

class Config:
    #Dataset config
    DATASET_BASE_PATH='dataset/used_data/'
    IMAGES_PATH=os.path.join(DATASET_BASE_PATH, 'images')

    #Input config
    INPUT_SIZE=(256, 256)
    INPUT_SHAPE=(256, 256, 3)
    MASK_SIZE=(92, 92)

    #Training config
    TRAIN_RATIO=0.8
    VALID_RATIO=0.1
    TEST_RATIO=0.1

    #Model config
    EPOCHS=100
    BATCH_SIZE=12
    
