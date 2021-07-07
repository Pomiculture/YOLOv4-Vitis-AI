#!/bin/bash

###############################################################################################################

# Run the Vitis AI YOLOv4 pre-compiled model from Model Zoo as a demo, with COCO dataset.
# First, launch the Vitis AI Docker image for CPU : source ./workflow/0_start_docker_cpu.sh

###############################################################################################################

# Path to the demo bash scripts
export PATH_TO_DEMO=workflow/demo

###############################################################################################################

# Compile App
source ./workflow/11_compile_app.sh

# Load the pre-compiled YOLOv4 model from Model Zoo
source ./${PATH_TO_DEMO}/0_load_yolov4_demo_model.sh

# Deploy the model to be run by the accelerator card
source ./${PATH_TO_DEMO}/1_deploy_demo_model.sh

# Load the overlay for Alveo U280
source ./workflow/15_load_overlay.sh

# Run the application
./${PATH_TO_DEMO}/2_run_demo_model.sh
