#!/bin/bash

###############################################################################################################

# Create the [model].prototxt file to the folder.

###############################################################################################################

echo "-----------------------------------------"
echo "CREATING THE PROTOTXT FILE '${PROTOTXT_DIR}/${MODEL_NAME}.prototxt'"
echo "-----------------------------------------"

python ${SRC_UTILS}/create_prototxt.py \
	--model_name ${MODEL_NAME} \
	--output_layers ${OUTPUT_NODE_NAMES} \
	--conf_threshold ${CONF_THRESHOLD} \
	--nms_threshold ${NMS_THRESHOLD} \
	--classes_path ${CLASSES} \
	--anchors_path ${ANCHORS} \
	--output_folder ${PROTOTXT_DIR}

echo "-----------------------------------------"
echo "PROTOTXT FILE CREATED"
echo "-----------------------------------------"
