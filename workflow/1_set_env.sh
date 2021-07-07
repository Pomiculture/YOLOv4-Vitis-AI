#!/bin/bash

###############################################################################################################

# Set shared environment variables.

###############################################################################################################

echo "-----------------------------------------"
echo "ACTIVATING ANACONDA TENSORFLOW ENVIRONMENT"
echo "-----------------------------------------"

# Activate the Conda virtual environment for Vitis AI with TensorFlow framework
conda activate vitis-ai-tensorflow                                     

echo "-----------------------------------------"
echo "ANACONDA ENVIRONMENT ACTIVATED"
echo "-----------------------------------------"

###############################################################################################################

# Hardware specs
export ALVEO_MODEL=U280
export DPU_CONFIG=DPUCAHX8H
export DPU_ARCHIVE=alveo_xclbin-1.3.0
export DPU_FREQ=14E300M

###############################################################################################################

# Model specs
export MODEL_TYPE=yolov4 			#yolov3
export MODEL_NAME=${MODEL_TYPE}_apple 		#${MODEL_TYPE}_coco
export DATASET=apples 				#coco
export INPUT_NODE_NAME=image_input                                                            
export OUTPUT_NODE_NAMES=conv2d_93/BiasAdd,conv2d_101/BiasAdd,conv2d_109/BiasAdd    #conv2d_58/BiasAdd,conv2d_66/BiasAdd,conv2d_74/BiasAdd   
export INPUT_WIDTH=416
export INPUT_HEIGHT=416
export NB_CHANNELS=3
export INPUT_SHAPE=?,${INPUT_WIDTH},${INPUT_HEIGHT},${NB_CHANNELS}   
export MODEL_SPECS=./model/specs
export CLASSES=${MODEL_SPECS}/${DATASET}_classes.txt
export ANCHORS=${MODEL_SPECS}/yolov4_anchors.txt

###############################################################################################################

# Dataset and ouput data     
export DATASET_PATH=./data/dataset_${DATASET}                 
export INPUT_FOLDER=./data/test_images
export INPUT_LIST=./data/test_list.txt
export OUTPUT_FOLDER=./output
export ALVEO_OUTPUT=${OUTPUT_FOLDER}/3_alveo
export OUTPUT_IMAGES=${ALVEO_OUTPUT}/images
export LOG_FILE=${ALVEO_OUTPUT}/output.txt

###############################################################################################################

# Graph output
export GRAPH_OUTPUT=${OUTPUT_FOLDER}/2_graph

# Darknet output
export SOFTWARE_DIR=${OUTPUT_FOLDER}/1_darknet_software

###############################################################################################################

# Convert Darknet model to TensorFlow
export DARKNET_MODEL_DIR=./model/darknet
export TF_MODEL_DIR=./model/build/tensorflow    
export SRC_TENSORBOARD=./src/tensorboard
export SRC_UTILS=./src/utils

###############################################################################################################

# Freeze graph
export FREEZE=./model/build/tensorflow_freeze

export TF_FROZEN_GRAPH=${MODEL_NAME}_frozen.pb

###############################################################################################################

# Calibrate		
export NB_ITERATIONS=10         				# TODO 100 - 1000 images
export BATCH_SIZE=10
export NB_IMAGES=180
export IMG_FORMAT=jpg          
export CALIB_DATASET=/workspace/data/calib_dataset
export CALIB_IMAGE_LIST=/workspace/data/calib_image_list.txt

export CUDA_VISIBLE_DEVICES=0       

# Quantize model
export QUANT=/workspace/model/build/quantize  
export SRC_CALIBRATION=./src/calibration

###############################################################################################################

# Compile model
export COMPILE=./model/build/compile
export ARCH=/opt/vitis_ai/compiler/arch/${DPU_CONFIG}/${ALVEO_MODEL}/arch.json 

###############################################################################################################

# Prototxt
export PROTOTXT_DIR=./model/prototxt
export CONF_THRESHOLD=0.6
export NMS_THRESHOLD=0.5

###############################################################################################################

# App
export SRC_APP_DIR=./src/app
export SRC_APP=src/app
export FILE_PATH=${SRC_APP_DIR}/yolov3

