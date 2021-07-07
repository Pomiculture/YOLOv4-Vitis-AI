#!/bin/bash

###############################################################################################################

# Launch the Vitis AI Docker image for CPU.

###############################################################################################################

# Parameters
export PROCESSOR_TYPE=cpu
export VITIS_DOCKER_IMAGE_VERSION=latest

###############################################################################################################

# Move to Docker files directory
cd ./workflow/docker/

echo "-----------------------------------------"
echo "RUNNING VITIS AI DOCKER IMAGE FOR ${PROCESSOR_TYPE^^}"
echo "-----------------------------------------"

# Run Vitis AI container (for CPU)
source ./docker_run.sh xilinx/vitis-ai-${PROCESSOR_TYPE}:${VITIS_DOCKER_IMAGE_VERSION}  

# Move to workspace root directory
cd ./../../
