#!/bin/bash

###############################################################################################################

# Deploy the model by copying it to the path '/usr/share/vitis_ai_library/models/' 
# where the Vitis AI models are located.

###############################################################################################################

# Deploy model
sudo mkdir /usr/share/vitis_ai_library/models
sudo cp -r ${COMPILE}/${MODEL_NAME} /usr/share/vitis_ai_library/models/

echo "-----------------------------------------"
echo "DEPLOY MODEL DIRECTORY '${COMPILE}/${MODEL_NAME}'"
echo "COPIED TO PATH '/usr/share/vitis_ai_library/models/'"
echo "-----------------------------------------"
