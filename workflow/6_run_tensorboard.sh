#!/bin/bash

###############################################################################################################

# Open the TensorBoard with a wb browser to visualize the graph of the network.

###############################################################################################################

echo "-----------------------------------------"
echo "VISUALIZE INFERENCE TENSORFLOW GRAPH"
echo "-----------------------------------------"

python ${SRC_TENSORBOARD}/open_tensorboard.py \
	    --graph      ${TF_MODEL_DIR}/${MODEL_NAME}_frozen.pb \
	    --log_dir 	 ${TF_MODEL_DIR}/tb_logs \
	    --port    	 6006
