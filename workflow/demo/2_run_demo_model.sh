#!/bin/bash

###############################################################################################################

# Run the application on the Alveo U280 board.

###############################################################################################################

# Parameters
export MODEL_NAME=yolov4_leaky_spp_m
export CLASSES=./model/coco_classes.txt
export SRC_APP=src/app
export INPUT_FOLDER=./data/test_images
export OUTPUT_FOLDER=./output/0_demo
export OUTPUT_IMAGES=${OUTPUT_FOLDER}/images
export LOG_FILE=${OUTPUT_FOLDER}/output.txt

###############################################################################################################

echo "-----------------------------------------"
echo "RUNNING APP ON ALVEO ${ALVEO_MODEL}"
echo "-----------------------------------------" 

# Run app
./${SRC_APP}/yolov3 ${MODEL_NAME} ${INPUT_FOLDER} ${OUTPUT_IMAGES} ${LOG_FILE} ${CLASSES} ${ALVEO_MODEL}

echo "-----------------------------------------"
echo "RUN COMPLETE"
echo "-----------------------------------------" 
