#!/bin/bash

###############################################################################################################

# Build testing set to evaluate the YOLO model.

###############################################################################################################

# Parameters
export SRC_EVAL=./src/eval
export DATA_DIR=./data

###############################################################################################################

# Create file gathering the labels and anchors of the test dataset samples
python ${SRC_EVAL}/gather_labels_anchors.py \
	--dataset ${DATASET_PATH} \
	--image_format ${IMG_FORMAT} \
	--output_file ${DATA_DIR}/labels_anchors.txt

###############################################################################################################

# Prepare test data
python ${SRC_UTILS}/prepare_data.py \
	--image_dir ${DATASET_PATH} \
	--output_folder ${INPUT_FOLDER} \
	--image_format ${IMG_FORMAT} \
	--image_list ${INPUT_LIST}



