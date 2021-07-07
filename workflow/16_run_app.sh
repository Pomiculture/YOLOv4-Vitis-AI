#!/bin/bash

###############################################################################################################

# Run the application on the Alveo U280 board.

###############################################################################################################

# Create output folder
mkdir -p ${ALVEO_OUTPUT}

echo "-----------------------------------------"
echo "RUNNING APP ON ALVEO ${ALVEO_MODEL}"
echo "-----------------------------------------" 

# Run app
./${SRC_APP}/yolov3 ${MODEL_NAME} ${INPUT_FOLDER} ${OUTPUT_IMAGES} ${LOG_FILE} ${CLASSES} ${ALVEO_MODEL}

echo "-----------------------------------------"
echo "RUN COMPLETE"
echo "-----------------------------------------" 
