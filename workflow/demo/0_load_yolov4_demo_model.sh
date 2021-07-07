#!/bin/bash

###############################################################################################################

# Run the application on the Alveo U280 board.
# Source : https://github.com/Xilinx/Vitis-AI/blob/master/models/AI-Model-Zoo/model-list/dk_yolov4_coco_416_416_60.1G_1.3/model.yaml 
# (board: u50lv9e & u50lv10e & u280)

###############################################################################################################

# Name of archive to download
export ARCHIVE_NAME=yolov4_leaky_spp_m-u50lv-u280-r1.3.1

# Alveo model
export ALVEO_MODEL=U280
# DPU ARCHIVE
export DPU_ARCHIVE=alveo_xclbin-1.3.0
# DPU frequency
export DPU_FREQ=14E300M

###############################################################################################################

# Download model
wget https://www.xilinx.com/bin/public/openDownload?filename=${ARCHIVE_NAME}.tar.gz -O ${ARCHIVE_NAME}.tar.gz

# Untar the model package
tar -xzvf ${ARCHIVE_NAME}.tar.gz
	
# Remove archive
rm ${ARCHIVE_NAME}.tar.gz
