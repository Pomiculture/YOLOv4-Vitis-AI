#!/bin/bash

###############################################################################################################

# Get the name and shape of the input tensor(s), and the name of the ouput tensor(s).
# Source : https://newbedev.com/given-a-tensor-flow-model-graph-how-to-find-the-input-node-and-output-node-names

###############################################################################################################

python ${SRC_UTILS}/get_io_tensors.py \
	    --graph     ${TF_MODEL_DIR}/${TF_FROZEN_GRAPH}
