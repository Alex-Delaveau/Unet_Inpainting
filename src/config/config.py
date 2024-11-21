import os

class Config:
    #Dataset config
    DATASET_BASE_PATH='xhlulu/flickrfaceshq-dataset-nvidia-resized-256px'

    #Input config
    INPUT_SIZE=(256, 256)
    INPUT_SHAPE=(256, 256, 3)
    MASK_SIZE=(92, 92)

    #Training config
    TRAIN_RATIO=0.8
    VALID_RATIO=0.2
    TEST_RATIO=0

    #Model config
    EPOCHS=100
    BATCH_SIZE=12
    
