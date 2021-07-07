#!/bin/bash

###############################################################################################################

# Evaluate the performances of the graph.

###############################################################################################################

echo "-----------------------------------------"
echo " EVALUATING THE mAP score.."
echo "-----------------------------------------"

# Evaluate mAP score
python ${SRC_EVAL}/eval.py \
	--results_file ${GRAPH_OUTPUT}/output.txt \
	--gt_file ${DATA_DIR}/labels_anchors.txt \
	--detection_thresh ${CONF_THRESHOLD} \
	--iou_thresh ${NMS_THRESHOLD}

echo "-----------------------------------------"
echo "EVALUATION COMPLETE"
echo "-----------------------------------------"
