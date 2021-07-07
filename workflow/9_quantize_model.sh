#!/bin/bash

###############################################################################################################

# Create a set of image files to be used in the calibration phase of quantization.
# Launch the Vitis AI quantizer for TensorFlow to convert the floating-point frozen graph 
# (32-bit floating-point weights and activation values) to a fixed-point integer (8-bit integer - INT8) model.
# The calibration dataset is a subset of the training dataset containing 100 to 1000 images.
# In the quantize calibration process, only a small set of unlabeled images are required to analyze the distribution of activations.

###############################################################################################################        

run_quant() {

	# Create calibration dataset (pick random set of different images)			
	python ${SRC_CALIBRATION}/build_calib_data.py  \
		--image_dir=${DATASET_PATH} \
		--output_folder=${CALIB_DATASET} \
		--image_list=${CALIB_IMAGE_LIST} \
		--image_format=${IMG_FORMAT} \
		--nb_images=${NB_IMAGES}

	# Display the quantizer version being used
	vai_q_tensorflow --version                                         

	# Move to calibration directory
	cd ${SRC_CALIBRATION} 

	# Quantize
	vai_q_tensorflow quantize \
		--input_frozen_graph /workspace/${TF_MODEL_DIR}/${TF_FROZEN_GRAPH} \
		--input_fn           image_input_fn.calib_input \
		--output_dir         ${QUANT} \
		--input_nodes        ${INPUT_NODE_NAME} \
		--output_nodes       ${OUTPUT_NODE_NAMES} \
		--input_shapes       ${INPUT_SHAPE} \
		--calib_iter         ${NB_ITERATIONS} \
		--gpu                ${CUDA_VISIBLE_DEVICES}    

	# Return to workspace
	cd /workspace
            
}

###############################################################################################################

quant() {
	echo "Quantizing frozen graph..."
	echo "-----------------------------------------"
	echo "QUANTIZE STARTED.."
	echo "-----------------------------------------"

	# Reset quantization folder
	rm -rf ${QUANT}                                                       
	mkdir -p ${QUANT} 

	# Run quantization
	run_quant                                                          

	echo "-----------------------------------------"
	echo "QUANTIZE COMPLETE"
	echo "-----------------------------------------"

	echo "Two files are generated in the output directory : "
	echo "- quantize_eval_model.pb is for targeting DPUCZDX8G implementations."
	echo "- deploy_model.pb is used for evaluation and as input for the VAI compiler for most DPU architectures, like DPUCAHX8H, DPUCAHX8L, and DPUCADF8H)."
}

###############################################################################################################

# Quantize graph
quant

