#!/bin/bash

###############################################################################################################

# Run the whole Vitis AI workflow.
# First, launch the Vitis AI Docker image for CPU : source ./workflow/0_start_docker_cpu.sh

###############################################################################################################

# Set useful environment variables
source ./workflow/1_set_env.sh

# Build testing set
source ./workflow/2_build_test_set.sh

###############################################################################################################

# Run Darknet model
source ./workflow/3_run_darknet.sh

###############################################################################################################

# Convert Darknet model to TensorFlow frozen graph
source ./workflow/4_darknet_keras_tf.sh

# Display name of input and output nodes
source ./workflow/5_get_io_tensors.sh

# Visualize the model with TensorBoard
source ./workflow/6_run_tensorboard.sh

# Run TensorFlow graph
source ./workflow/7_run_graph.sh

# Evaluate TensorFlow graph
source ./workflow/8_eval.sh

###############################################################################################################

# Quantize model
source ./workflow/9_quantize_model.sh

# Compile model
source ./workflow/10_compile_model.sh

# Compile App
source ./workflow/11_compile_app

# Create prototxt file
source ./workflow/12_create_prototxt.sh

###############################################################################################################

# Add prototxt file to target folder
source ./workflow/13_add_prototxt.sh

# Deploy model
source ./workflow/14_deploy_model.sh

# Load the overlay for U280
source ./workflow/15_load_overlay.sh

# Run the application
source ./workflow/16_run_app.sh

# Measure mAP score
source ./workflow/17_eval_score.sh
