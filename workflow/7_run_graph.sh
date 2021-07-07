#!/bin/bash

###############################################################################################################

# Run the tensorFlow graph to check its performances.

###############################################################################################################

# Parameters
export OUTPUT_FILE=./output/graph_eval/output.txt

###############################################################################################################

# Run the graph
python ${SRC_EVAL}/run_graph.py \
	--graph ${TF_MODEL_DIR}/${TF_FROZEN_GRAPH} \
	--input ${INPUT_NODE_NAME} \
	--outputs ${OUTPUT_NODE_NAMES} \
	--anchors ${ANCHORS} \
	--classes ${CLASSES} \
	--det_thresh ${CONF_THRESHOLD} \
	--nms_thresh ${NMS_THRESHOLD} \
	--dataset ${INPUT_FOLDER} \
	--img_format ${IMG_FORMAT} \
	--results ${GRAPH_OUTPUT}/output.txt



