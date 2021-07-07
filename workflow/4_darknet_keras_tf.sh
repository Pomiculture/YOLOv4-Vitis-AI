#!/bin/bash

###############################################################################################################

# Convert Darknet model to TensorFlow frozen model.
# Source : https://github.com/david8862/keras-YOLOv3-model-set

###############################################################################################################

echo "-----------------------------------------"
echo "RESETTING ${TF_MODEL_DIR} FOLDER CONTENT"
echo "-----------------------------------------"

# Reset output directory
rm -rf ${TF_MODEL_DIR} 
mkdir -p ${TF_MODEL_DIR} 

echo "-----------------------------------------"
echo "RESET COMPLETE"
echo "-----------------------------------------"

###############################################################################################################

echo "-----------------------------------------"
echo "CONVERTING DARKNET MODEL TO KERAS"
echo ""
echo "Inputs :"
echo "- Darknet model : ${DARKNET_MODEL_DIR}/${MODEL_NAME}.cfg"
echo "- Darknet weights : ${DARKNET_MODEL_DIR}/${MODEL_NAME}.weights"
echo ""
echo "Output :"
echo "- Keras model : ${TF_MODEL_DIR}/${MODEL_NAME}.h5"
echo "-----------------------------------------"

# Convert Darknet model description (.cfg) and Darknet weights (.weights) into a Keras model (.h5)
python ./external_tools/keras-YOLOv3-model-set/tools/model_converter/convert.py \
	--yolo4_reorder ${DARKNET_MODEL_DIR}/${MODEL_NAME}.cfg \
	${DARKNET_MODEL_DIR}/${MODEL_NAME}.weights \
	${TF_MODEL_DIR}/${MODEL_NAME}.h5                                     

###############################################################################################################

echo "-----------------------------------------"
echo "CONVERTING TENSORFLOW MODEL TO KERAS"
echo ""
echo "Input :"
echo "- Keras model : ${TF_MODEL_DIR}/${MODEL_NAME}.h5"
echo ""
echo "Output :"
echo "- TensorFlow frozen model : ${TF_MODEL_DIR}/${TF_FROZEN_GRAPH}.pb"
echo "-----------------------------------------"

# Convert Keras model (.h5) to a frozen TensorFlow graph (.pb)
python ./external_tools/keras-YOLOv3-model-set/tools/model_converter/keras_to_tensorflow.py \
	--input_model ${TF_MODEL_DIR}/${MODEL_NAME}.h5 \
	--output_model ${TF_MODEL_DIR}/${TF_FROZEN_GRAPH}

echo "-----------------------------------------"
echo "CONVERSION COMPLETE"
echo "-----------------------------------------"
