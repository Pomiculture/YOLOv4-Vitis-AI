#!/bin/bash

###############################################################################################################

# Run the Darknet model on a sample image.
# Source : https://github.com/AlexeyAB/darknet

###############################################################################################################

# Move testing set to AlexeyAB Darknet folder
sudo cp ${INPUT_FOLDER} /workspace/external_tools/darknet_AlexeyAB/data -r
sudo cp ${INPUT_LIST}  /workspace/external_tools/darknet_AlexeyAB/ -r

###############################################################################################################

# Move to AlexeyAB Darknet application folder
cd ./external_tools/darknet_AlexeyAB

# Create results folder
mkdir results

###############################################################################################################

# Run Darknet model over a testing set
./app/darknet detector valid \
	${DATASET}.data \
	/workspace/${DARKNET_MODEL_DIR}/${MODEL_NAME}.cfg \
	/workspace/${DARKNET_MODEL_DIR}/${MODEL_NAME}.weights

###############################################################################################################

# Test on a sample image
#./app/darknet detector test \
#	${DATASET}.data \
#	/workspace/${DARKNET_MODEL_DIR}/${MODEL_NAME}.cfg \
#	/workspace/${DARKNET_MODEL_DIR}/${MODEL_NAME}.weights \
#	data/test.jpg \
#	-thresh 0.3

###############################################################################################################

# Remove unwanted output file
rm bad.list

# Copy results to output folder
sudo cp results/. /workspace/${SOFTWARE_DIR} -r
echo "The results can be found in folder /workspace/${SOFTWARE_DIR}"

# Go back to root directory
cd /workspace
