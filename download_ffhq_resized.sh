#!/bin/bash

# Define download path
DOWNLOAD_PATH=~/dataset/archive.zip
UNZIP_DIR=~/dataset/ffhq_resized

# Step 1: Download the dataset
echo "Downloading the dataset..."
curl -L -o $DOWNLOAD_PATH \
https://www.kaggle.com/api/v1/datasets/download/xhlulu/flickrfaceshq-dataset-nvidia-resized-256px

# Step 2: Check if the download was successful
if [ -f "$DOWNLOAD_PATH" ]; then
    echo "Download completed. Unzipping the file..."
    
    # Create a directory for unzipping the dataset
    mkdir -p $UNZIP_DIR
    
    # Step 3: Unzip the file
    unzip -o $DOWNLOAD_PATH -d $UNZIP_DIR
    
    echo "Unzipping completed. Dataset is available at $UNZIP_DIR."
else
    echo "Download failed. Please check your internet connection or Kaggle API setup."
    exit 1
fi
