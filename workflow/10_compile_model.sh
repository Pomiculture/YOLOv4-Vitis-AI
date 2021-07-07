#!/bin/bash
# Launches the Vitis AI compiler for TensorFlow to compile the quantized model into an .xmodel file for the Alveo U280 accelerator card.

###############################################################################################################

# Function for compilation (XIR based compiler)				#TODO : adapt shape
run_compile() {
	vai_c_tensorflow \
		--frozen_pb  ${QUANT}/quantize_eval_model.pb \
		--arch       ${ARCH} \
		--output_dir ${COMPILE}/${MODEL_NAME} \
		--net_name   ${MODEL_NAME} \
		--options "{'mode':'normal','save_kernel':'', 'input_shape':'1,416,416,3'}"                	               
}

compile() {

	echo "-----------------------------------------"
	echo "COMPILE FOR TARGET ALVEO ${ALVEO_MODEL} STARTED.."
	echo "-----------------------------------------"

	# Reset compile folder
	rm -rf ${COMPILE}
	mkdir -p ${COMPILE}

	# Compile quantized model
	run_compile                                                   

	echo "-----------------------------------------"
	echo "COMPILE  FOR TARGET ALVEO ${ALVEO_MODEL} COMPLETE"
	echo "-----------------------------------------"

}

###############################################################################################################

# Compile
compile
