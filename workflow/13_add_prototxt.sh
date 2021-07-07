#!/bin/bash

###############################################################################################################

# Add the [model].prototxt file to the folder to deploy on the hardware target.

###############################################################################################################

# Move file
sudo cp ${PROTOTXT_DIR}/${MODEL_NAME}.prototxt ${COMPILE}/${MODEL_NAME}

echo "-----------------------------------------"
echo "COPYING FILE '${PROTOTXT_DIR}/${MODEL_NAME}.prototxt'"
echo "TO PATH ' ${COMPILE}/${MODEL_NAME}'"
echo "-----------------------------------------"
