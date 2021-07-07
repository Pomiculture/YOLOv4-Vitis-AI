# Copyright 2020 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.  

import os
import cv2
import numpy as np

###############################################################################################################

# Get the environment variables (calibration dataset, image names)
calib_image_dir = os.environ['CALIB_DATASET'] 	         
calib_image_list = os.environ['CALIB_IMAGE_LIST'] 	           
calib_batch_size = int(os.environ['BATCH_SIZE'])
input_node=os.environ['INPUT_NODE_NAME']
input_width=int(os.environ['INPUT_WIDTH'])
input_height=int(os.environ['INPUT_HEIGHT'])
size = (input_width, input_width)

###############################################################################################################

def preprocess(image):
	"""
	Resize the image to fit the model input size.
	Normalize image from [0:255] pixel values to the range [0:1].
	"""
	# Resize the image to match the model requirements
	image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
	# Set the values to float type
	image  = np.asarray(image)
	image = image.astype(np.float32)
	# Scale image
	return image / 255.0							# TODO : vraiment resize ?


###############################################################################################################

def calib_input(iter):
	"""
	Input of the Yolo algorithm for calibration, using a batch of images.
	"""
	images = []
	# Read content of the calibration image list
	line = open(calib_image_list).readlines()
	# Run a batch
	for index in range(0, calib_batch_size):
		# Get the image name to process
		curline = line[iter * calib_batch_size + index]
		calib_image_name = curline.strip()
		# Open the corresponding image file
		filename = os.path.join(calib_image_dir, calib_image_name)
		image = cv2.imread(filename)
		# Check whether the image is empty
		if image is None :
			raise TypeError("Image {} is empty.".format(filename))
		# Resize and normalize image
		image = preprocess(image)
		# Append image to list of inputs
		images.append(image)
		print("Iteration number : {} and index number {} and  file name  {} ".format(iter, index, filename))
	# Link input images to the input node name
	return {input_node: images}
