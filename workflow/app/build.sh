#!/bin/bash

###############################################################################################################

# Build the application with the required libraries.

###############################################################################################################

# Aliases
export CXX=${CXX:-g++}

###############################################################################################################

# Build source code
$CXX -std=c++17 -I. -o ${FILE_PATH} ${FILE_PATH}.cpp \
-lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lvitis_ai_library-yolov3  -pthread -lglog -lvitis_ai_library-model_config
